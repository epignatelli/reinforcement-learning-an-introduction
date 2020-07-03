import numpy as np
import random


ACTIONS = [
    True,   # hit
    False,  # stick   
    ]

DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # [A, 2, 3, 4, 5, 6, 7, 8, 9, J, Q, K]
THETA = 1e3

class Blackjack:
    """
    
    """
    def __init__(self):
        self.values = np.zeros((32, 11, 2))
        self.reset()
    
    def deal(self, n=1):
        """
        The deck is considered infinite (cards are replaced), so there is no check on what cards to draw.
        """
        return random.choices(DECK, k=n)
    
    def score(self, cards):
        s = sum(cards)
        # usable ace
        if 1 in cards and s + 10 <= 21:
            s += 10
        return s
    
    def is_natural(self, cards):
        """
        Check if the hand is a natural, and Ace and a Figure or a 10.
        """
        return 1 in self.player and 10 in self.player and len(cards) == 2
    
    def get_observation(self):
        """
        An observation is a tuple with the available information we use to make the decision.
        In this case, the sum of our cards, the sum of the dealer's visible cards (exclude the first),
        and whether we have a usable ace
        """
        return (int(sum(self.player)), int(self.dealer[1]), int(1 in self.player))
    
    def reset(self):
        self.player = self.deal(2)
        self.dealer = self.deal(2)
        return self.get_observation(), False
    
    def step(self, action):
        # check natural win, player never hits
        if self.is_natural(self.player) and not self.is_natural(self.dealer):
            return self.get_observation(), +1, True
            
        if action:  # hit
            self.player += self.deal()
            if self.score(self.player) > 21:  # bust
                return self.get_observation(), -1, True
        else:  # stick, dealer's turn
            while self.score(self.dealer) < 17:
                self.dealer += self.deal()
            if self.score(self.dealer) > 21:
                return self.get_observation(), +1, True
            elif self.score(self.dealer) == self.score(self.player):
                return self.get_observation(), 0, True
        return self.get_observation(), 0, False
    
    def render(self):
        print("Player: {}, Dealer: {}".format(self.player, self.dealer))
    
def mc_policy_evaluation(env, policy):
    delta = float("inf")
    while delta > THETA:
        # run episode
        obs, done = env.reset()
        while not done:
            obs, reward, done = env.step(policy[obs])
        old_value = self.values[obs]
        self.values[obs]
            
    
if __name__ == "__main__":
    policy = np.ones((32, 11, 2))  # always hits
    policy[20:] = 0  # stick if score is above 20
    
    env = Blackjack()
    values = mc_policy_evaluation(env, policy)