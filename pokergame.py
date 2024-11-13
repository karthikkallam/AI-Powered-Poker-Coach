import random
from collections import Counter
import torch

RANKS = '2 3 4 5 6 7 8 9 T J Q K A'.split()
SUITS = 'hearts diamonds clubs spades'.split()
STATE_SIZE = 15  # Number of input features for the state representation

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f'{self.rank} of {self.suit}'

    def rank_value(self):
        return RANKS.index(self.rank)

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(self.cards)

    def deal(self, num=1):
        return [self.cards.pop() for _ in range(num)]

class Player:
    def __init__(self, name, stack=100):
        self.name = name
        self.hand = []
        self.stack = stack
        self.current_bet = 0
        self.is_folded = False

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def reset_hand(self):
        self.hand = []

    def show_hand(self, hide_hand=False):
        if hide_hand and self.name == 'AI':
            return '[Hidden]'
        return ', '.join(str(card) for card in self.hand)

    def place_bet(self, amount):
        if amount <= self.stack:
            self.current_bet += amount
            self.stack -= amount
            return True
        return False

    def clear_bet(self):
        self.current_bet = 0

class PokerGame:
    def __init__(self, players, ai_model=None):
        self.players = [Player(name) for name in players]
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.ai_model = ai_model
        self.action_log = []
        self.total_games = 0
        self.ai_wins = 0
        self.total_chips_won = 0

    def reset(self):
        """Reset the game state for a new game."""
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        for player in self.players:
            player.reset_hand()
            player.current_bet = 0
            player.is_folded = False
        self.deal_hole_cards()  # Start a new game with fresh hole cards

    def deal_hole_cards(self):
        for player in self.players:
            player.receive_cards(self.deck.deal(2))

    def deal_community_cards(self, num):
        self.community_cards.extend(self.deck.deal(num))

    def show_community_cards(self):
        return ', '.join(str(card) for card in self.community_cards)

    def get_state(self, player):
        """Creates a game state representation for the AI model."""
        hand = [card.rank_value() for card in player.hand]
        hand += [0] * (2 - len(hand))  # Pad if hand is not full

        community = [card.rank_value() for card in self.community_cards]
        community += [0] * (5 - len(community))  # Pad if fewer than 5 community cards

        current_bet = player.current_bet
        stack = player.stack
        pot_size = self.pot

        state = hand + community + [current_bet, stack, pot_size]
        state += [0] * (STATE_SIZE - len(state))  # Ensure state vector has 15 elements

        return state

    def play_game(self):
        """Orchestrates the entire game flow."""
        self.reset()  # Reset the game before starting
        self.deal_hole_cards()  # Deal initial hole cards

        # Pre-flop betting
        self.betting_round(is_pre_flop=True)

        # Flop (3 community cards)
        self.deal_community_cards(3)
        self.betting_round()

        # Turn (1 community card)
        self.deal_community_cards(1)
        self.betting_round()

        # River (1 community card)
        self.deal_community_cards(1)
        self.betting_round()

        # Determine the winner
        winner = self.determine_winner()
        self.update_metrics(winner)

        return winner.name, self.pot

    def handle_action(self, player, action, amount=0):
        """Handles player action and updates the game state accordingly."""
        if action == 'bet':
            player.place_bet(amount)
            self.pot += amount
        elif action == 'call':
            call_amount = self.get_call_amount(player)
            player.place_bet(call_amount)
            self.pot += call_amount
        elif action == 'raise':
            player.place_bet(amount)
            self.pot += amount
        elif action == 'fold':
            player.is_folded = True

    def get_call_amount(self, player):
        """Calculate the call amount for the player."""
        max_bet = max(p.current_bet for p in self.players)
        return max(0, max_bet - player.current_bet)

    def ai_action(self, player, is_pre_flop):
        """AI decision-making based on game state."""
        state = self.get_state(player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.ai_model(state_tensor)

        action = q_values.argmax().item()

        # Dynamically determine bet/raise amounts
        bet_amount = self.calculate_dynamic_bet(state)
        raise_amount = self.calculate_dynamic_raise(state)

        if action == 0:  # Bet
            self.handle_action(player, 'bet', bet_amount)
        elif action == 1:  # Call
            call_amount = self.get_call_amount(player)
            self.handle_action(player, 'call', call_amount)
        elif action == 2:  # Raise
            self.handle_action(player, 'raise', raise_amount)
        elif action == 3:  # Fold
            self.handle_action(player, 'fold')

    def calculate_dynamic_bet(self, state):
        """Calculate dynamic bet amount based on AI model."""
        hand_strength = self.evaluate_hand(state)
        return int(hand_strength / 100 * self.players[1].stack)  # Bet proportionally to hand strength

    def calculate_dynamic_raise(self, state):
        """Calculate dynamic raise amount based on AI model."""
        hand_strength = self.evaluate_hand(state)
        return int(hand_strength / 75 * self.players[1].stack)  # Raise more aggressively with stronger hands

    def determine_winner(self):
        """Determines the winner based on hand evaluation."""
        active_players = [player for player in self.players if not player.is_folded]
        hand_ranks = {player: self.evaluate_hand(player.hand + self.community_cards) for player in active_players}
        winner = max(hand_ranks, key=hand_ranks.get)
        return winner

    def update_metrics(self, winner):
        """Update win rate and chips won metrics after each game."""
        self.total_games += 1
        if winner.name == 'AI':
            self.ai_wins += 1
        self.total_chips_won += self.pot

    def evaluate_hand(self, hand):
        """Placeholder for hand evaluation logic."""
        return random.randint(1, 100)  # Dummy evaluation for demonstration
