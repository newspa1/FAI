from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from utils.monte_carlo_simulation import MonteCarloSimulator

class SleepcatPlayer(
    BasePokerPlayer
):
    def __init__(self):
        super(SleepcatPlayer, self).__init__()
        self.seat = None
        self.initial_stack = None
        self.max_round = None
        self.small_blind_amount = None
        self.big_blind_amount = None

        self.current_stack = None

        self.preflop_winning_rate_table = {
            "AA": 0.85, "AK": 0.68, "AQ": 0.67, "AJ": 0.66, "AT": 0.66, "A9": 0.64, "A8": 0.63, "A7": 0.63, "A6": 0.62, "A5": 0.62, "A4": 0.61, "A3": 0.60, "A2": 0.59,
            "KK": 0.83, "KQ": 0.64, "KJ": 0.64, "KT": 0.63, "K9": 0.61, "K8": 0.60, "K7": 0.59, "K6": 0.58, "K5": 0.58, "K4": 0.57, "K3": 0.56, "K2": 0.55,
            "QQ": 0.80, "QJ": 0.61, "QT": 0.61, "Q9": 0.59, "Q8": 0.58, "Q7": 0.56, "Q6": 0.55, "Q5": 0.55, "Q4": 0.54, "Q3": 0.53, "Q2": 0.52,
            "JJ": 0.78, "JT": 0.59, "J9": 0.57, "J8": 0.56, "J7": 0.54, "J6": 0.53, "J5": 0.52, "J4": 0.51, "J3": 0.50, "J2": 0.50,
            "TT": 0.75, "T9": 0.56, "T8": 0.54, "T7": 0.53, "T6": 0.51, "T5": 0.49, "T4": 0.49, "T3": 0.48, "T2": 0.47,
            "99": 0.72, "98": 0.53, "97": 0.51, "96": 0.50, "95": 0.48, "94": 0.46, "93": 0.46, "92": 0.45,
            "88": 0.69, "87": 0.50, "86": 0.49, "85": 0.47, "84": 0.45, "83": 0.43, "82": 0.43,
            "77": 0.67, "76": 0.48, "75": 0.46, "74": 0.45, "73": 0.43, "72": 0.41,
            "66": 0.64, "65": 0.46, "64": 0.44, "63": 0.42, "62": 0.40,
            "55": 0.61, "54": 0.44, "53": 0.43, "52": 0.41,
            "44": 0.58, "43": 0.42, "42": 0.40,
            "33": 0.55, "32": 0.39,
            "22": 0.51,
        }
        
        self.is_win = False
        self.is_all_in = False
        self.mc_simulator = MonteCarloSimulator(num_simulations=500)

        self.opp_prev_action = None

    def declare_action(self, valid_actions, hole_card, round_state):
        if not self.is_win:
            self.is_win = self.__is_win(round_state["round_count"], self.seat)
        
        # If I fold, the enemy can win with fold
        if self.__is_win(round_state["round_count"] + 1, 1 - self.seat):
            return self.__all_in(valid_actions)
            
        if self.is_all_in:
            return self.__all_in(valid_actions)
        
        if self.is_win:    
            return self.__fold(valid_actions)
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        if round_state["street"] == "preflop":
            return self.preflop_action(valid_actions, hole_card, round_state) # action returned here is sent to the poker engine
        else:
            return self.postflop_action(valid_actions, hole_card, round_state)
        
        
        # community = [Card.from_str(c) for c in round_state["community_card"]]
        # hole = [Card.from_str(c) for c in hole_card]
        # hand_info = HandEvaluator.gen_hand_rank_info(hole, community)
        
        
    
    def postflop_action(self, valid_actions, hole_card, round_state):
        win_rate = self.mc_simulator.estimate_win_rate(hole_card, round_state["community_card"])
        if win_rate >= 0.6:
            self.is_all_in = True
            return self.__all_in(valid_actions)
        elif win_rate >= 0.5:
            return self.__raise(valid_actions, self.big_blind_amount * 5)
        else:
            # If my hole card is bad and enemy call, I call, too.
            if self.__is_button(round_state) and valid_actions[1]["amount"] == 0:
                return self.__call(valid_actions)
            else:
                if valid_actions[1]["amount"] == 0:
                    return self.__call(valid_actions)
                else:
                    return self.__fold(valid_actions)
        
        
    def preflop_action(self, valid_actions, hole_card, round_state):
        hole_obj = [Card.from_str(card) for card in hole_card]
        cards_rank = sorted([card.rank for card in hole_obj], reverse=True)
        hole_str = "".join([Card.RANK_MAP[r] for r in cards_rank])
        
        # print(hole_str)
        # if hole_str in strong_pairs:
        #     self.is_all_in = True
        #     return self.__all_in(valid_actions)
        # elif hole_str in medium_pairs or hole_str in strong_broadways or hole_card[0][0] == hole_card[1][0]:
        #     return self.__call(valid_actions)
        # else:
        #     return self.__fold(valid_actions)
        
        # print(self.preflop_winning_rate_table[hole_str])
        if self.preflop_winning_rate_table[hole_str] >= 0.65:
            self.is_all_in = True
            return self.__raise(valid_actions, self.big_blind_amount * 5)
        elif self.preflop_winning_rate_table[hole_str] >= 0.55:
            return self.__call(valid_actions)
        else:
            return self.__fold(valid_actions)
        
        
    
    def receive_game_start_message(self, game_info):
        seats = game_info["seats"]
        if seats[0]["uuid"] == self.uuid:
            self.seat = 0
        else:
            self.seat = 1
        
        self.initial_stack = game_info["rule"]["initial_stack"]
        self.max_round = game_info["rule"]["max_round"]
        
        self.small_blind_amount = game_info["rule"]["small_blind_amount"]
        self.big_blind_amount = self.small_blind_amount * 2

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.current_stack = seats[self.seat]["stack"]
        self.is_win = False
        self.is_all_in = False

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def __is_win(self, round_count, seat):
        big_blind_count = small_blind_count = (self.max_round - round_count) // 2
        if not round_count & 1:
            if seat == 0:
                big_blind_count += 1
            else:
                small_blind_count += 1
        
        all_fold_cost = self.big_blind_amount * big_blind_count + self.small_blind_amount * small_blind_count
        stack = self.current_stack if seat == self.seat else 2 * self.initial_stack - self.current_stack
        if stack - self.initial_stack > all_fold_cost:
            return True
        
        return False
        
    def __is_button(self, round_state):
        return self.seat == round_state["dealer_btn"]
    
    def __raise(self, valid_actions, amount):
        raise_action_info = valid_actions[2]
        action = raise_action_info["action"]
        
        min_amount = raise_action_info["amount"]["min"]
        max_amount = raise_action_info["amount"]["max"]
        amount = max(min_amount, min(amount, max_amount))
        
        return action, amount
    
    def __call(self, valid_actions):
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def __fold(self, valid_actions):
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
        return action, amount

    def __all_in(self, valid_actions):
        call_action_info = valid_actions[2]
        action, amount = call_action_info["action"], call_action_info["amount"]['max']
        if amount <= 0:
            return self.__call(valid_actions)
        return action, amount
        
def setup_ai():
    return SleepcatPlayer()
