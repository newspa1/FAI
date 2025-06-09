import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.sleepcat_player import setup_ai as sleepcat_ai
from agents.shower_player import setup_ai as shower_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=baseline6_ai())
config.register_player(name="Sleepcat", algorithm=sleepcat_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
# game_result = start_poker(config, verbose=1)
# print(json.dumps(game_result, indent=4))
# print(game_result)
# print("======")

win_num_p1 = 0.0
win_num_p2 = 0.0
for i in range (100):
    print(f"round {i}")
    game_result = start_poker(config, verbose=1)
    if game_result["players"][0]["stack"] > 1000:
        win_num_p1 += 1.0
    if game_result["players"][1]["stack"] > 1000:
        win_num_p2 += 1.0

print(f"p1: Winning rate = {win_num_p1 / (win_num_p1 + win_num_p2)}")
print(f"p2: Winning rate = {win_num_p2 / (win_num_p1 + win_num_p2)}")
