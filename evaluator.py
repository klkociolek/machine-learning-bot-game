from botbowl import EnvConf, BotBowlEnv, register_bot, OutcomeType, Game
from botbowl.ai import RandomBot

from Data.uji_bot import UjiBot
from actor import BotActor
import botbowl.web.server as server
import botbowl
from Data.scripted_bot import ScriptedBot

def register_a2c(botname,agent_path):
    def _make_my_a2c_bot(name, env_size=11):
        return BotActor(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename=agent_path)

    register_bot(botname, _make_my_a2c_bot)

if __name__ == '__main__':
    agent_paths = "models/botbowl-11/bc_baseline.nn"
    register_a2c("bc_bot",agent_path=agent_paths)
    ScriptedBot.register_bot()
    UjiBot.register_bot()
    host = "127.0.0.1"
    botbowl.register("random",RandomBot)
    server.start_server(host=host, debug=True, use_reloader=False, port=1234)