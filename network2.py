from typing import Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from botbowl.ai.layers import *
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper

# Architecture
class ConfigParams(Enum):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = f"models/botbowl-11/bc_network2.nn"
    gradient_clip = 1.0

model_name = '260d8284-9d44-11ec-b455-faffc23fefdb'
env_name = f'botbowl-11'
model_filename = f"models/botbowl-11/bc_network2.nn"
log_filename = f"logs/{env_name}/{env_name}.dat"

def make_env(env_conf, ppcg=False):
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    return env

env = make_env(EnvConf(size=11))
spat_obs, non_spat_obs, action_mask = env.reset()
spatial_obs_space = spat_obs.shape
non_spatial_obs_space = non_spat_obs.shape[0]
action_space = len(action_mask)

class ResidualBlock(nn.Module):
    def __init__(self, kernels=[128, 128]):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out += residual
        return out

    def reset_parameters(self, relu_gain):
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)


class ChannelAttention(nn.Module):
    def __init__(self, kernels=[128, 64, 17], spatial_shape=spatial_obs_space):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=kernels[0]+spatial_shape[0],
                               out_channels=kernels[1], kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=kernels[1], out_channels=kernels[1], kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=kernels[1], out_channels=kernels[2], kernel_size=3, stride=1, padding='same')
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(1024, 64)
        self.linear3 = nn.Linear(1024, 17)

    def forward(self, x):#, latent_space):
        # out = torch.cat(F.leaky_relu(self.conv0(x[0])), F.sigmoid(F.leaky_relu(self.linear0(latent_space))))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear1(x[1])))
        out = torch.multiply(F.leaky_relu(self.conv1(x[0])), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear2(x[1])))
        out = torch.multiply(F.leaky_relu(self.conv2(out)), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear3(x[1])))
        out = torch.multiply(F.leaky_relu(self.conv3(out)), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        return out

    def reset_parameters(self, relu_gain):
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)
        self.linear3.weight.data.mul_(relu_gain)



class TransformerPolicy2(nn.Module):

    def __init__(self, spatial_input_shape, num_non_spatial_features, hidden_nodes=1024, kernels=[128,128], action_dim=action_space):
        super(TransformerPolicy2    , self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_input_shape[0], out_channels=kernels[0], kernel_size=4, stride=1, padding="same")
        # Residual block layers
        # self.conv_res = self._make_layer(ResidualBlock, kernels, 4)
        self.conv_res0 = ResidualBlock()
        self.conv_res1 = ResidualBlock()
        self.conv_res2 = ResidualBlock()
        self.conv_res3 = ResidualBlock()
        # Non-spatial input stream
        self.linear0 = nn.Linear(num_non_spatial_features, hidden_nodes)
        self.linear1 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = nn.Linear(hidden_nodes, hidden_nodes)
        # Concatenated stream
        self.linear4 = nn.Linear(1024 + 128 * spatial_input_shape[1] * spatial_input_shape[2], hidden_nodes)
        self.linear5 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear6 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear7 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_out = nn.Linear(hidden_nodes, 1)
        # Convolutions with channel attention
        # self.conv_ch = self._make_layer(ChannelAttention, [128, 64, 17], 1)
        self.conv_ch = ChannelAttention([128, 64, 17])
        # self.linear_actor = nn.Linear(1024, 25)
        self.linear_actor = nn.Linear(1024, 24)

        # Linear layers
        # stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        # stream_size += hidden_nodes
        # self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, action_dim)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(relu_gain)
        #self.conv_res.weight.data.mul_(relu_gain)
        #for res in self.conv_res:  # reset residual blocks
        #   res.reset_parameters(relu_gain)
        self.conv_res0.reset_parameters(relu_gain)
        self.conv_res1.reset_parameters(relu_gain)
        self.conv_res2.reset_parameters(relu_gain)
        self.conv_res3.reset_parameters(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)
        self.linear3.weight.data.mul_(relu_gain)
        self.linear4.weight.data.mul_(relu_gain)
        self.linear5.weight.data.mul_(relu_gain)
        self.linear6.weight.data.mul_(relu_gain)
        self.linear7.weight.data.mul_(relu_gain)
        #for att in self.conv_ch:  # reset channel attention
        #   att.reset_parameters(relu_gain)
        self.conv_ch.reset_parameters(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers

        x1 = self.conv1(spatial_input)
        # x1 = self.conv_res(x1)  # todo check that, yields too big results
        x1 = self.conv_res0(x1)
        x1 = self.conv_res1(x1)
        x1 = self.conv_res2(x1)
        x1 = self.conv_res3(x1)
        x1_cat = torch.cat((x1, spatial_input), dim=1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = F.leaky_relu(self.linear0(non_spatial_input))
        x2 = F.leaky_relu(self.linear1(x2))
        x2 = F.leaky_relu(self.linear2(x2))
        x2 = F.leaky_relu(self.linear3(x2))

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        x3 = F.leaky_relu(self.linear4(concatenated))
        x3 = F.leaky_relu(self.linear5(x3))
        x3 = F.leaky_relu(self.linear6(x3))
        x3 = F.leaky_relu(self.linear7(x3))

        # Convolution with channel attention
        # x3 needs to be resized due to the difference in sizes of spatial and non-spatial obs spaces
        x4 = self.conv_ch((x1_cat, x3))
        x4_flatten = x4.flatten(start_dim=1)
        x5 = self.linear_actor(x3)
        actor = torch.cat((x4_flatten, x5), dim=1)

        # Output streams
        value = self.linear_out(x3)
        #value = self.critic(x7)
        # actor = self.actor(x6)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        # In rare cases, multinomial can  sample an action with p=0, so let's avoid that
        for i, action in enumerate(actions):
            correct_action = action
            while not action_mask[i][correct_action]:
                correct_action = action_probs[i].multinomial(1)
            actions[i] = correct_action
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs
    def get_action_log_probs(self, spatial_input, non_spatial_input, action_mask=None):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        log_probs = F.log_softmax(actions, dim=1)
        return values, log_probs

class A2CAgent(Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 scripted_func: Callable[[Game], Optional[Action]] = None,
                 filename=model_filename):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)

        self.scripted_func = scripted_func
        self.action_queue = []

        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_shape = spatial_obs.shape
        num_non_spatial = non_spatial_obs.shape[0]

        # MODEL
        self.device = ConfigParams.device.value
        self.policy = TransformerPolicy2(
            spatial_input_shape=spatial_shape,
            num_non_spatial_features=num_non_spatial,
            action_dim=len(action_mask)
        )
        self.policy.load_state_dict(torch.load(filename, map_location=self.device, weights_only=False))
        self.policy.eval()
        self.end_setup = False

    def new_game(self, game, team):
        pass

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game

        spatial_obs, non_spatial_obs, action_mask = map(A2CAgent._update_obs, self.env.get_state())
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        _, actions = self.policy.act(
            Variable(spatial_obs.float()),
            Variable(non_spatial_obs.float()),
            Variable(action_mask))

        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx)

        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        pass


def main():
    # Register the bot to the framework
    def _make_my_a2c_bot(name, env_size=11):
        return A2CAgent(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename=model_filename)
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

        print("Starting game", (i+1))
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

    print(f"Home/Draws/Away: {wins}/{draws}/{n-wins-draws}")
    print(f"Home TDs per game: {tds_home/n}")
    print(f"Away TDs per game: {tds_away/n}")


if __name__ == "__main__":
    main()