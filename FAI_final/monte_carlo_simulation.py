import random
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
from game.engine.deck import Deck

class MonteCarloSimulator:
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        print("Initial montecarlo")
        
    def estimate_win_rate(self, hole_cards, community_cards):
        # for _ in range(self.num_simulations):
        win_num = 0
        deck = Deck()
        
        for _ in range(self.num_simulations):
            sim_opp_hole = self.draw_unknown_cards(deck, hole_cards + community_cards, 2)
            sim_opp_hole_cards = [card_to_str(card) for card in sim_opp_hole]
            
            community = [Card.from_str(card) for card in community_cards]
            
            sim_community = community
            if len(sim_community) != 5:
                sim_community += self.draw_unknown_cards(deck, hole_cards + community_cards + sim_opp_hole_cards, 5 - len(community))
            
            hole = [Card.from_str(card) for card in hole_cards]
            my_hand = HandEvaluator.gen_hand_rank_info(hole, sim_community)
            opp_hand = HandEvaluator.gen_hand_rank_info(sim_opp_hole, sim_community)
            
        
            if (my_hand["hand"]["strength"] > opp_hand["hand"]["strength"] or 
                my_hand["hand"]["strength"] == opp_hand["hand"]["strength"] and my_hand["hand"]["high"] > opp_hand["hand"]["high"]):
                win_num += 1
        
        win_rate = win_num / self.num_simulations
        return win_rate
    
    def draw_unknown_cards(self, deck: Deck, known_cards, num_draw):
        deck.shuffle()
        known_cards_obj = [Card.from_str(card) for card in known_cards]
        drawn_cards = []
        for card in deck.deck:
            if not any(card.rank == known_card.rank and card.suit == known_card.suit for known_card in known_cards_obj):
                drawn_cards.append(card)
                if len(drawn_cards) == num_draw:
                    break
        return drawn_cards

def card_to_str(card: Card):
    return Card.SUIT_MAP[card.suit] + Card.RANK_MAP[card.rank]