from enum import Enum
import numpy as np
import uuid
from typing import Callable
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn  import Sequential,Conv2d ,BatchNorm2d,ReLU,Sigmoid,AdaptiveAvgPool2d,Linear,LayerNorm,Dropout,TransformerEncoder,ModuleList,TransformerEncoderLayer,MultiheadAttention
from torch.nn.modules.linear  import NonDynamicallyQuantizableLinear
from torch.nn.functional import relu
import botbowl
from botbowl.ai.layers import *
from botbowl.ai.env import EnvConf
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper
from botbowl import OutcomeType, Game
import botbowl.web.server as server





class ConfigParams(Enum):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_steps = 200000 # 12000000
    num_processes = 1 #4 # was 5
    steps_per_update = 40 #10  # 50 # 1000
    batch_size = 5 #32  # 512  # generally the size should be the 2^n, the bigger it is the higher the exploration, but slower learning
    batch_size_bc = 5  # 32  # 512  # generally the size should be the 2^n, the bigger it is the higher the exploration, but slower learning
    steps_per_update_bc = 5
    buffer_size = 10000 #12000  #Todo rework how buffer works (each step should be individual entry to the list)
    multiple_updates = 8
    multiple_steps = 2
    learning_rate = 5e-3  # 5e-6
    gamma = 0.999
    tau = 0.001 # 0.75 worked well# 0.01 # setting this higher helped talk abt this in paper
    gradient_clip = 1.0 # 1.0 for pure BC #1.5 # was 10
    q_regularization = 0  # 0.005 #0.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.05  # without it the gradient gets too big and model starts getting worse after the while
    log_interval = 25
    save_interval = 50
    reset_steps = 5000  # The environment is reset after this many steps it gets stuck
    patience = 20
    selfplay_window = 5
    selfplay_save_steps = 1000 #int(num_steps / 100000)
    selfplay_swap_steps = selfplay_save_steps
    ppcg = True
    env_size = 11  # Options are 1,3,5,7,11
    env_name = f"botbowl-{env_size}"
    env_conf = EnvConf(size=env_size)#, pathfinding=True)  # pathfinndng=False
    selfplay = False  #todo make it work
    exp_id = str(uuid.uuid1())
    model_dir = f"models/{env_name}/"
    model_path = f"models/{env_name}/bc_network1.nn"  #bc_baseline.nn
    agent_path = f"models/{env_name}/best.nn"  # bc_model.nn"
    num_hidden_nodes = 128
    num_layers = 8
    num_heads = 16
    freeze_steps = 400000
    

class SparseActionHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.head = nn.Linear(input_dim, action_dim)
        nn.init.orthogonal_(self.head.weight, gain=0.01)  # Critical initialization
        nn.init.zeros_(self.head.bias)

    def forward(self, x, action_mask=None):
        logits = self.head(x)
        if action_mask is not None:
            # Numerical stability: avoid -inf
            logits = logits.masked_fill(~action_mask, -1e8)
        return logits

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=40, max_w=60):
        super().__init__()
        self.pe_h = nn.Parameter(torch.randn(1, max_h, d_model))
        self.pe_w = nn.Parameter(torch.randn(1, max_w, d_model))
        
    def forward(self, x, H, W):
        pe_h = self.pe_h[:, :H, :]
        pe_w = self.pe_w[:, :W, :]
        pe_grid = pe_h.unsqueeze(2) + pe_w.unsqueeze(1)
        pe_grid = pe_grid.view(1, H*W, -1)
        return x + pe_grid

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        attn_weights = self.attn(x)
        return x * attn_weights

class TransformerPolicy(nn.Module):
    def __init__(self, spatial_input_shape, num_non_spatial_features, action_dim, 
                 hidden_dim=ConfigParams.num_hidden_nodes.value, num_shared_layers=ConfigParams.num_layers.value, num_branch_layers=ConfigParams.num_layers.value, num_heads=8):
        super().__init__()
        C, H, W = spatial_input_shape
        
        # Shared Feature Extractor (CNN part)
        self.spatial_processor = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            AttentionConv(128, 128),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Shared Token Projections
        self.token_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.non_spatial_proj = nn.Sequential(
            nn.Linear(num_non_spatial_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Shared Transformer Backbone (common encoder layers)
        self.shared_transformer = self.build_transformer(hidden_dim, num_heads, num_shared_layers)
        
        # Separate transformer branches for actor and critic (additional layers)
        self.actor_transformer = self.build_transformer(hidden_dim, num_heads, num_branch_layers)
        self.critic_transformer = self.build_transformer(hidden_dim, num_heads, num_branch_layers)
        
        # Output Heads
        self.policy_head = SparseActionHead(hidden_dim, action_dim)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def build_transformer(self, hidden_dim, num_heads, num_layers):
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
    
    def forward(self, spatial_input, non_spatial_input, action_mask=None):
        batch_size = spatial_input.shape[0]
        
        # CNN-based feature extraction
        spatial_feats = self.spatial_processor(spatial_input)
        spatial_tokens = self.token_proj(spatial_feats).flatten(2).permute(0, 2, 1)
        
        non_spatial = self.non_spatial_proj(non_spatial_input.view(batch_size, -1)).unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate tokens: [CLS token, non-spatial token, spatial tokens]
        x = torch.cat([cls_tokens, non_spatial, spatial_tokens], dim=1)
        
        # Shared transformer processing
        x = self.shared_transformer(x)
        
        # Branching into actor and critic paths
        actor_features = self.actor_transformer(x)
        critic_features = self.critic_transformer(x)
        
        logits = self.policy_head(actor_features[:, 0], action_mask)
        value = self.value_head(critic_features[:, 0])
        
        return value, logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    def freeze_non_transformer(self):
        # Freeze all parameters except transformer layers
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze transformer layers
        for module in [self.shared_transformer, self.actor_transformer, self.critic_transformer]:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    def act(self, spatial_inputs, non_spatial_input, action_mask):
        # Get raw logits from the network
        value, logits = self(spatial_inputs, non_spatial_input, action_mask)
        if len(action_mask.shape) == 1:
                action_mask = torch.reshape(action_mask, (1, -1))
        logits[~action_mask] = -1e10
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Sample actions from the masked distribution
        dist = Categorical(logits=log_probs)
        actions = dist.sample()
        
        return value, actions
    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, action_mask):
        # Get masked logits from forward
        value, logits = self(spatial_inputs, non_spatial_input, action_mask)

        # Stable probability calculation
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        # Correct entropy calculation
        probs = log_probs.exp()
        entropy = -(log_probs * probs).sum(dim=-1).mean()

        return action_log_probs, value, entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        value, logits = self(spatial_input, non_spatial_input, action_mask)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        return value, probs, log_probs
    def get_action_log_probs(self, spatial_input, non_spatial_input, action_mask):
        value, probs, log_probs = self.get_action_probs(spatial_input, non_spatial_input,action_mask)
        return value, log_probs


class BotActor(botbowl.Agent):
    env: BotBowlEnv
    BOT_ID = 'BBot'

    def __init__(self, name='BBot', env_conf: EnvConf = EnvConf(size=11, pathfinding=True),
                 scripted_func: Callable[[Game], Optional[Action]] = None,
                 filename=ConfigParams.model_path.value):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)

        self.scripted_func = scripted_func
        self.action_queue = []

        # MODEL
        # self.policy = CNNPolicy()  # For testing games
        # self.policy.load_state_dict(torch.load(filename))
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_shape = spatial_obs.shape
        num_non_spatial = non_spatial_obs.shape[0]

        self.device = ConfigParams.device.value
        self.policy = TransformerPolicy(
            spatial_input_shape=spatial_shape,
            num_non_spatial_features=num_non_spatial,
            action_dim=len(action_mask)
        )
        self.policy.load_state_dict(torch.load(filename, map_location=self.device, weights_only=False))
        self.policy.eval()
        self.policy.to(self.device)
        self.end_setup = False

    def new_game(self, game, team):
        '''
        Called when creating a new game.
        '''
        self.own_team = team
        self.opp_team = game.get_opp_team(team)
        self.is_home = team == game.state.home_team

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def sample(self, state):
        self.eps_greedy = self.eps_greedy * self.eps_decay
        if np.random.randn() < self.eps_greedy:
            return self.env.action_space.sample()

        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)[np.newaxis]).float().to(self.device)
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)[np.newaxis]).float().to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        _, actions = self.policy.act(spatial_obs, non_spatial_obs, action_mask)

        return actions

    def act(self, game):
        '''
        Called for every game step in order to determine the action to take.
        '''
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)[np.newaxis]).float().to(self.device)
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)[np.newaxis]).float().to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        _, actions = self.policy.act(spatial_obs, non_spatial_obs, action_mask)
        # Create action from output
        action_idx = actions[0]

        action_objects = self.env._compute_action(action_idx.item(), flip=not self.is_home)

        # Return action to the framework
        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        '''
        Called when the game ends.
        '''
        winner = game.get_winning_team()
        if winner is None:
            print("It's a draw")
        elif winner == self.own_team:
            print(f'I ({self.name}) won.')
            print(self.own_team.state.score, '-', self.opp_team.state.score)
        else:
            print(f'I ({self.name}) lost.')
            print(self.own_team.state.score, '-', self.opp_team.state.score)

    @staticmethod
    def register_bot(path=ConfigParams.model_path.value):
        """
        Adds the bot to the registered bots if not already done.
        """
        if BotActor.BOT_ID.lower() not in botbowl.list_bots():
            botbowl.register_bot(BotActor.BOT_ID, lambda name: BotActor(name=name, model_path=path))


if __name__ == '__main__':



    def _make_my_a2c_bot(name, env_size=11):
        return BotActor(name=name,
                        env_conf=EnvConf(size=11),
                        filename=ConfigParams.model_path.value)


    botbowl.register_bot('my-a2c-bot', _make_my_a2c_bot)

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    wins = 0
    draws = 0
    n = 10
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):

        if is_home:
            away_agent = botbowl.make_bot('random')
            home_agent = botbowl.make_bot('my-a2c-bot')
        else:
            away_agent = botbowl.make_bot('my-a2c-bot')
            home_agent = botbowl.make_bot("random")
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i + 1))
        game.init()
        print("Game is over")

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score

    print(f"Home/Draws/Away: {wins}/{draws}/{n - wins - draws}")
    print(f"Home TDs per game: {tds_home / n}")
    print(f"Away TDs per game: {tds_away / n}")


