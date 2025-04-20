import numpy as np
import torch
from torch.autograd import Variable
from typing import Callable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from botbowl.ai.layers import *
import botbowl.web.server as server

#from network import TransformerPolicy, ConfigParams
#from network2 import TransformerPolicy2, ConfigParams
from network3 import TransformerPolicy3, ConfigParams
from Data.scripted_bot import ScriptedBot
import botbowl.web.server as server


# @ray.remote
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
        # self.policy = TransformerPolicy(
        # spatial_input_shape=spatial_shape,
        # num_non_spatial_features=num_non_spatial,
        # action_dim=len(action_mask)
        # )
        # self.policy = TransformerPolicy2(
        #     spatial_input_shape=spatial_shape,
        #     num_non_spatial_features=num_non_spatial,
        #     action_dim=len(action_mask)
        # )
        self.policy = TransformerPolicy3(
            spatial_input_shape=spatial_shape,
            num_non_spatial_features=num_non_spatial,
            action_dim=len(action_mask)
        )
        self.policy.load_state_dict(torch.load(filename, map_location=self.device,weights_only=False))
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
        print(f"Selected action: {action_idx}")
        print(f"Is action valid? {action_mask[action_idx]}")

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

