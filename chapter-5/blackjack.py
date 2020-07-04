import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


ACTIONS = [
    True,   # hit
    False,  # stick
]

# [A, 2, 3, 4, 5, 6, 7, 8, 9, J, Q, K]
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class Blackjack:
    """
    A Blackjack environment designed for Reinforcement Learning applications,
    tailored on the Monte Carlo Value Estimation from the book:
    Reinforcement Learning: An Introduction
    http://incompleteideas.net/book/code/blackjack1.lisp
    """

    def __init__(self):
        self.values = np.zeros((32, 11, 2))
        self.reset()

    def __str__(self):
        return "Player: {}, Dealer: {}".format(self.player, self.dealer)

    def deal(self, n=1):
        """
        Draws a random card from the deck
        The deck is considered infinite (cards are replaced), so there is no check on what cards to draw.

        Args:
            n (int): The number of cards to deal

        Returns:
            (List[int]): The list of cards drawn from the deck
        """
        return random.choices(DECK, k=n)

    def score(self, cards):
        """
        Computes the score of the current hand
        The Ace is considered 10 if the player doesn't go bust, 1 otherwise
        Args:
            (List[int]): The list of cards in hand

        Returns:
            The sum of the cards.
        """
        s = sum(cards)
        # usable ace
        if 1 in cards and s + 10 <= 21:
            s += 10
        return s

    def get_observation(self):
        """
        An observation is a tuple with the available information we use to make the decision.
        In this case, the sum of our cards, the sum of the dealer's visible cards (exclude the first),
        and whether we have a usable ace

        Returns:
            (Tuple[int, int, int]): A tuple with the three elements used for the observation:
            The score of the player's hand, the card of the dealer facing up, and whether the player has a usable ace
        """
        return (self.score(self.player), int(self.dealer[1]), int(1 in self.player))

    def reset(self):
        """
        Deals another, new hand of blackjack.
        """
        self.player = self.deal(2)
        self.dealer = self.deal(2)
        return self.get_observation(), False

    def step(self, action):
        """
        Plays a blackjack hand, given an action of the player
        """
        # player plays
        if action:  # hit
            self.player += self.deal()
            if self.score(self.player) > 21:  # bust?
                return self.get_observation(), -1., True
            else:  # if the player hits, the dealers doesn't play
                # the episode does not end, the player may want to hit again
                return self.get_observation(), 0., False
        else:  # stick
            pass
        player_score = self.score(self.player)

        # dealer's turn
        while self.score(self.dealer) < 17:  # draw until we reach at least 17
            self.dealer += self.deal()
        dealer_score = self.score(self.dealer)
        if dealer_score > 21:  # bust?
            return self.get_observation(), +1., True

        # outcome
        if player_score > dealer_score:
            return self.get_observation(), +1., True
        elif player_score == dealer_score:
            return self.get_observation(), 0., True
        else:
            return self.get_observation(), -1., True

    def render(self):
        xticklabels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        yticklabels = range(12, 22)
        fig, ax = plt.subplots(2, 1, figsize=(7, 25))
        sns.heatmap(self.values[12:22, 1:, 0], cmap="RdBu", vmin=-1, vmax=1,
                    xticklabels=xticklabels, yticklabels=yticklabels, annot=True, ax=ax[0])
        ax[0].set_title("V(π) with no usable Ace")
        ax[0].set_xlabel("Dealer showing")
        ax[0].set_ylabel("Player sum")
        sns.heatmap(self.values[12:22, 1:, 1], cmap="RdBu", vmin=-1, vmax=1,
                    xticklabels=xticklabels, yticklabels=yticklabels, annot=True, ax=ax[1])
        ax[1].set_title("V(π) with usable Ace")
        ax[1].set_xlabel("Dealer showing")
        ax[1].set_ylabel("Player sum")
        plt.show()


def mc_policy_evaluation(env, policy, iterations=10000, first_visit=True):
    counts = np.ones_like(env.values) * 1e-9
    for k in range(iterations):
        # run episode
        print("MC iteration {}/{}\t".format(k + 1, iterations), end="\r")
        obs, done = env.reset()
        while not done:
            obs, reward, done = env.step(policy[obs])
            # print(env, obs, reward, done)
            if not (first_visit and counts[obs] == 0):
                env.values[obs] += reward
            counts[obs] += 1
    print()
    return env.values / counts


if __name__ == "__main__":
    # default policy
    policy = np.ones((32, 11, 2))  # always hits
    policy[20:] = 0  # stick if score is above 20

    # policy iteration, 10k iterations
    env = Blackjack()
    env.values = mc_policy_evaluation(env, policy, 10000)
    env.render()

    # policy iteration, 500k iterations
    env2 = Blackjack()
    env2.values = mc_policy_evaluation(env2, policy, 500000)
    env2.render()
