from typing import Callable
import itertools

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv, RewardWrapper, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper

from Data.uji_bot import UjiBot
from network import BotActor
from network2 import A2CAgent
from network3 import A2CAgent_3
from examples.scripted_bot_example import MyScriptedBot


def register_bots():
    UjiBot.register_bot()

    # Register network2 bot
    def _make_network2_bot(name, env_size=11):
        return A2CAgent(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename="models/botbowl-11/bc_network2.nn")

    botbowl.register_bot('network2', _make_network2_bot)

    # Register network1 bot
    def _make_network1_bot(name, env_size=11):
        return BotActor(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename="models/botbowl-11/bc_network1.nn")

    botbowl.register_bot('network1', _make_network1_bot)

    # Register network3 bot
    def _make_network3_bot(name, env_size=11):
        return A2CAgent_3(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename="models/botbowl-11/bc_network3.nn")

    botbowl.register_bot('network3', _make_network3_bot)


    # The random bot is assumed to be provided by the framework (no registration needed)


def play_game(home_bot_name: str, away_bot_name: str, game_id: int, home_team, away_team, config, arena, ruleset):
    home_agent = botbowl.make_bot(home_bot_name)
    away_agent = botbowl.make_bot(away_bot_name)
    game = botbowl.Game(game_id, home_team, away_team, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print(f"Starting game {game_id + 1}: Home: {home_bot_name} vs Away: {away_bot_name}")
    game.init()
    print("Game is over")

    # Determine winner: returns the agent object or None if draw.
    winner = game.get_winner()
    # Get team scores (touchdowns)
    home_td = game.get_agent_team(home_agent).state.score
    away_td = game.get_agent_team(away_agent).state.score
    return winner, home_td, away_td


def main():
    register_bots()

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home_team = botbowl.load_team_by_filename("human", ruleset)
    away_team = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # List of bot names to compete in the tournament
    bot_names = ["network1", "network2", "network3", "scripted", "random", "ujiBot"]

    # Initialize results dictionary for tracking wins, draws, and total touchdowns
    results = {bot: {"wins": 0, "draws": 0, "losses": 0, "td": 0} for bot in bot_names}

    games_per_pair = 10
    game_id = 0

    # Create all unique pairs of bots
    for bot1, bot2 in itertools.combinations(bot_names, 2):
        print(f"\n--- Tournament matchup: {bot1} vs {bot2} ---")
        for i in range(games_per_pair):
            # Alternate home and away roles for fairness
            if i % 2 == 0:
                home_bot = bot1
                away_bot = bot2
            else:
                home_bot = bot2
                away_bot = bot1

            winner, home_td, away_td = play_game(home_bot, away_bot, game_id, home_team, away_team, config, arena,
                                                 ruleset)
            game_id += 1

            # Record touchdowns for both bots (accumulate score)
            results[home_bot]["td"] += home_td
            results[away_bot]["td"] += away_td

            # Update win/draw/loss stats based on game outcome
            if winner is None:
                results[home_bot]["draws"] += 1
                results[away_bot]["draws"] += 1
                print("Result: Draw")
            else:
                # Identify winning bot by matching the winner's name
                # Assumption: The bot instance's name attribute matches the registered name.
                if winner.name == home_bot:
                    results[home_bot]["wins"] += 1
                    results[away_bot]["losses"] += 1
                    print(f"Result: {home_bot} wins")
                else:
                    results[away_bot]["wins"] += 1
                    results[home_bot]["losses"] += 1
                    print(f"Result: {away_bot} wins")

    # Print tournament summary
    print("\n=== Tournament Summary ===")
    for bot in bot_names:
        total_games = results[bot]["wins"] + results[bot]["draws"] + results[bot]["losses"]
        avg_td = results[bot]["td"] / total_games if total_games > 0 else 0
        print(f"Bot: {bot} | Wins: {results[bot]['wins']} | Draws: {results[bot]['draws']} | "
              f"Losses: {results[bot]['losses']} | Avg TD per game: {avg_td:.2f}")


if __name__ == "__main__":
    main()
