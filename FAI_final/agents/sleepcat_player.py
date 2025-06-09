from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from monte_carlo_simulation import MonteCarloSimulator

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

        self.is_win = False
        self.is_all_in = False
        
        self.must_all_in = False

        self.mc_simulator = MonteCarloSimulator(num_simulations=500)

    def declare_action(self, valid_actions, hole_card, round_state):
        if not self.is_win:
            self.is_win = self.__is_win(round_state["round_count"], self.seat)
        
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
        # print(round_state)
        
        win_rate = self.mc_simulator.estimate_win_rate(hole_card, round_state["community_card"])
        print(win_rate)
        
        # if self.current_stack > self.initial_stack:
        #     if win_rate >= 0.45:
        #         return self.__raise(valid_actions, self.big_blind_amount * 10)
        #     else:
        #         # If my hole card is bad and enemy call, I call, too.
        #         if self.__is_button(round_state) and valid_actions[1]["amount"] == 0:
        #             return self.__call(valid_actions)
        #         else:
        #             if valid_actions[1]["amount"] == 0:
        #                 return self.__call(valid_actions)
        #             else:
        #                 return self.__fold(valid_actions)
        
        
        if win_rate >= 0.6:
            self.is_all_in = True
            return self.__all_in(valid_actions)
        elif win_rate >= 0.5:
            return self.__raise(valid_actions, self.big_blind_amount * 10)
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
        strong_pairs = ["AA", "KK", "QQ", "JJ", "TT"]
        medium_pairs = ["99", "88", "77", "66", "55"]
        strong_broadways = ["AK", "AQ", "AJ", "AT", "KQ", "KJ", "KT", "QJ", "QT"]
        
        hole_obj = [Card.from_str(card) for card in hole_card]
        cards_rank = sorted([card.rank for card in hole_obj], reverse=True)
        hole_str = "".join([Card.RANK_MAP[r] for r in cards_rank])
        
        if hole_str in strong_pairs:
            self.is_all_in = True
            return self.__all_in(valid_actions)
        elif hole_str in medium_pairs or hole_str in strong_broadways or hole_card[0][0] == hole_card[1][0]:
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
        if self.current_stack - self.initial_stack > all_fold_cost:
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
        return action, amount
        
def setup_ai():
    return SleepcatPlayer()
