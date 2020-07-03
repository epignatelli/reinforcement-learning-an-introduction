import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


ACTIONS = [
    True,   # hit
    False,  # stick   
    ]

DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # [A, 2, 3, 4, 5, 6, 7, 8, 9, J, Q, K]


class Blackjack:
    """
    
    """
    def __init__(self):
        self.values = np.zeros((32, 11, 2))
        self.reset()
        
    def __str__(self):
        return "Player: {}, Dealer: {}".format(self.player, self.dealer)
    
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
        return sorted(cards) == [1, 10]
    
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
        p_score = self.score(self.player)
        d_score = self.score(self.dealer)
        reward = 1 if p_score > d_score else 0 if p_score == d_score else -1
        return self.get_observation(), reward, True
    
    def render(self):
        xticklabels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        yticklabels = range(12, 22)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(self.values[12:22, 1:, 0], cmap="gray", vmin=-1, vmax=1, xticklabels=xticklabels, yticklabels=yticklabels, annot=True, ax=ax[0])
        ax[0].set_title("No usable Ace")
        ax[0].set_xlabel("Dealer showing")
        ax[0].set_ylabel("Player showing")
        sns.heatmap(self.values[12:22, 1:, 1], cmap="gray", vmin=-1, vmax=1, xticklabels=xticklabels, yticklabels=yticklabels, annot=True, ax=ax[1])
        ax[1].set_title("Usable Ace")
        ax[1].set_xlabel("Dealer showing")
        ax[1].set_ylabel("Player showing")
        plt.show()
    
def mc_policy_evaluation(env, policy, first_visit=True):
    counts = np.ones_like(env.values) * 1e-6
    for k in range(10000):
        # run episode
        obs, done = env.reset()
        while not done:
            obs, reward, done = env.step(policy[obs])
#             print(obs)
        if counts[obs] > 0:
            env.values[obs] += reward
        counts[obs] += 1
    return env.values / counts
    
if __name__ == "__main__":
    # default policy
    policy = np.ones((32, 11, 2))  # always hits
    policy[20:] = 0  # stick if score is above 20
    
    # policy iteration
    env = Blackjack()
    env.values = mc_policy_evaluation(env, policy)
    env.render()