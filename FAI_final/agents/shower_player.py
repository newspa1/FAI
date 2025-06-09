import game.visualize_utils as U
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
from game.engine.card import Card


type_list = [     
    "TWOPAIR", 
    "THREECARD",
    "STRAIGHT",
    "FLASH",
    "FULLHOUSE",
    "FOURCARD",
    "STRAIGHTFLASH",
    ]

class shower(BasePokerPlayer): 
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def __init__(self):
        super().__init__()
        self.num = -1
        self.round  = 0
        self.street = ""
        self.stack = 1000

    def find_player_num(self, game_info):
        for i in range(2):
            if game_info['seats'][i]['uuid'] == self.uuid:
                self.num = i
                break
        
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        gen_hole = gen_cards(hole_card)
        gen_comm = gen_cards(round_state['community_card'])
        #print("")
        #print(f"hand card : {hole_card + round_state['community_card']}")
        myresult = hand_card( gen_cards(hole_card), gen_cards(round_state['community_card']))
        mytype = myresult['hand']['strength']
        #print(f"hole card: {hole_card}")
        #print(f"comm card: {round_state['community_card']}")
        mylarge = myresult['hand']['high']
        #print(myresult)
        #print("")
        
        if self.stack >= 1150 - ((self.round-1)*15/2) :
            fold_action_info = valid_actions[0]
            action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif (mytype in type_list) and mylarge >= 12:
            if self.street == "preflop" :
                call_action_info = valid_actions[1]
                action = call_action_info["action"]
                amount = call_action_info["amount"]

            elif self.street == "turn" or self.street == "flop":
                raise_action_info = valid_actions[2]
                action = raise_action_info["action"]
                amount = raise_action_info["amount"]["min"]   
            else: 
                raise_action_info = valid_actions[2]
                action = raise_action_info["action"]
                amount = raise_action_info["amount"]["max"] * (mylarge /14)
        else:
            fold_action_info = valid_actions[0]
            action, amount = fold_action_info["action"], fold_action_info["amount"]

        if self.stack < 850 + ((self.round)*15/2+5):
            raise_action_info = valid_actions[2]
            action = raise_action_info["action"]
            amount = raise_action_info["amount"]["max"]

        if amount == -1 and action == valid_actions[2]["action"]:
            call_action_info = valid_actions[1]
            action = call_action_info["action"]
            amount = call_action_info["amount"]
        elif amount > self.stack:
            call_action_info = valid_actions[1]
            action = call_action_info["action"]
            amount = call_action_info["amount"]
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        if self.num == -1:
            self.find_player_num(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round = round_count

    def receive_street_start_message(self, street, round_state):
        self.street = street

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        #print(winners)
        self.stack = round_state['seats'][self.num]['stack']
        #print(f"mystack is {round_state['seats'][self.num]['stack']}")
        #print(hand_info)

def hand_card(hole_card, community_card):
    handcard = _fill_community_card(community_card, used_card = hole_card + community_card)
    result = HandEvaluator.gen_hand_rank_info(hole_card, handcard)
    return result

def _fill_community_card(base_cards, used_card):
    need_num = 5 - len(base_cards)
    return base_cards + _pick_unused_card(need_num, used_card)

def _pick_unused_card(card_num, used_card):
    used = [card.to_id() for card in used_card]
    unused = [card_id for card_id in range(1, 53) if card_id not in used]
    choiced = random.sample(unused, card_num)
    return [Card.from_id(card_id) for card_id in choiced]

def gen_cards(cards_str):
    return [Card.from_str(s) for s in cards_str]

def setup_ai():
    return shower()
