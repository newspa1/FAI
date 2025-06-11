import json
import copy
from game.game import setup_config, start_poker
from agents.sleepcat_player import setup_ai as sleepcat_ai
from agents.steve_player import setup_ai as steve_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
import concurrent.futures

# prepare base config
base_config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
base_config.register_player(name="p1", algorithm=baseline7_ai())
base_config.register_player(name="Sleepcat", algorithm=sleepcat_ai())

# for i in range(100):  
#     print(f"Collecting game {i}")
#     start_poker(base_config, verbose=1)

def run_single_game(i):
    print(f"round {i}")
    config = copy.deepcopy(base_config)
    game_result = start_poker(config, verbose=0)

    p1_win = 1.0 if game_result["players"][0]["stack"] > 1000 else 0.0
    p2_win = 1.0 if game_result["players"][1]["stack"] > 1000 else 0.0
    return p1_win, p2_win

# run in parallel
num_games = 100
win_num_p1 = 0.0
win_num_p2 = 0.0

with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(run_single_game, i) for i in range(num_games)]
    for future in concurrent.futures.as_completed(futures):
        p1_win, p2_win = future.result()
        win_num_p1 += p1_win
        win_num_p2 += p2_win

total = win_num_p1 + win_num_p2
print(f"p1: Winning rate = {win_num_p1 / total if total > 0 else 0}")
print(f"p2: Winning rate = {win_num_p2 / total if total > 0 else 0}")