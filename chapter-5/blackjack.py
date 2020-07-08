# MIT License

# Copyright (c) 2020 Eduardo Pignatelli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
    tailored on the Monte Carlo Value Estimation illustrated in the book
    Reinforcement Learning: An Introduction, R. S. Sutton, A. G. Barto, 2nd Edition, 2018.
    Original code from Rich available at http://incompleteideas.net/book/code/blackjack1.lisp
    """

    def __init__(self):
        self.values = np.zeros(self.observation_space)
        self.reset()

    def __str__(self):
        return "Player: {}, Dealer: {}".format(self.player, self.dealer)

    @property
    def observation_space(self):
        return (32, 11, 2)

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
            else:  # if the player hits, the dealer doesn't play
                # the episode does not end, the player may want to hit again
                return self.get_observation(), 0., False
        else:  # stick
            # dealer's turn now
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
