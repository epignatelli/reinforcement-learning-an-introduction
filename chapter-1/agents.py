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

import random
import copy
import numpy


class RandomPlayer:
    def __init__(self, marker):
        self.marker = str(marker)

    @staticmethod
    def get_action(state):
        """
        Given the current state of the board, selects a random move from all the available moves
        """
        pos = random.randint(0, 8)
        while state[pos] != " ":
            pos = random.randint(0, 8)
        return pos, None


class EGreedyPlayer:
    def __init__(self, marker, init_value=0.5, e_greedy=0., step_size=0.5,
                 decrement=0.9,
                 decrement_each=1000,
                 learn=True):
        self.marker = str(marker)
        self.learn = learn
        self.e_greedy = e_greedy
        self.step_size = step_size
        self.init_value = init_value
        self.decrement = decrement
        self.decrement_each = decrement_each

        # we store each state of the board as a dict whose key is the hash of the list
        # and the value is its state value
        self.state_value = {}
        self.n_plays = 0
        return

    def train(self):
        """
        Enables updates of the state-value table while playing
        """
        self.learn = True

    def eval(self):
        """
        Disables updates of the state-value table while playing
        """
        self.learn = False

    def encode(self, state):
        """
        Encodes the state of the board into a single string
        Args:
            state (List[str]): The board state as a list of strings: the player markers
        """
        return "".join(state)

    def set_value(self, state, value):
        """
        Updates the state-value of the given state in the table
        Args:
            state (List[str]): The board state as a list of strings: the player markers
            value (float): the probability of winning in that state
        """
        encoded = self.encode(state)
        self.state_value[encoded] = value
        return

    def get_value(self, state):
        """
        Performs a lookup to the state-value table for the given state.
        Args:
            state (List[str]): The board state as a list of strings: the player markers
        Returns:
            (float): the value of the state, if already visited, the initial value otherwise

        """
        encoded = self.encode(state)
        if encoded in self.state_value:
            return self.state_value[encoded]
        else:
            return self.init_value

    def back_up(self, state, next_state, next_value):
        """
        Performs a backup update of the current state using temporal difference learning
        Args:
            state (List[str]): The current state of the board as a list of strings: the player markers
            next_state (List[str]): The state of the board if we had to make this move, as a list of strings
            next_value (float): the value of the next state, if we had to make this move
        """
        current_value = self.get_value(state)
        self.set_value(next_state, next_value)

        # TD update
        new_value = current_value + self.step_size * (next_value - current_value)
        self.set_value(state, new_value)
        return

    def get_action(self, state):
        """
        Computes the e-greedy action, according to the current state of the board.
        Args:
            state (List[str]): The current state of the board
        Returns:
            (Tuple[int, float]): A tuple containing the best move and the reward deriving from it
        """
        # perform exploratory move with probability self.e_greedy
        # do not update value table
        if random.random() < self.e_greedy and self.learn:
            return RandomPlayer.get_action(state)

        best_move = None
        best_state = None
        best_reward = -float("inf")
        for pos in range(len(state)):
            # check that the cell is free
            if state[pos] == " ":
                next_state = copy.deepcopy(state)
                next_state[pos] = self.marker
                reward = self.get_reward(next_state)

                # choose greedily
                if reward >= best_reward:
                    best_move = pos
                    best_state = next_state
                    best_reward = reward

        if self.learn:
            self.back_up(state, best_state, best_reward)

        self.n_plays += 1
        if self.n_plays % self.decrement_each == 0:
            self.step_size *= self.decrement

        return best_move, best_reward

    def get_reward(self, state):
        """
        Computes the reward of the player at a given state of the board
        Args:
            state (List[str]): The current state of the board
        Returns:
            (float): 1 if the move is a winning move, 0 is the state is making the player lose.
                     In all the other cases returns the probability of the current state as stored in the
                     state-value table.
        """
        # row
        for i in range(0, 9, 3):
            if (state[i] == state[i + 1] == state[i + 2]):
                # three in a row
                if state[i] == self.marker:
                    return 1  # won, win prop 1.
                elif state[i] == " ":
                    return self.get_value(state)  # row is empty
                else:
                    return 0  # lost, win prob 0.
        # col
        for i in range(3):
            if (state[i] == state[i + 3] == state[i + 6]):
                # three in a row
                if state[i] == self.marker:
                    return 1  # won, win prop 1.
                elif state[i] == " ":
                    return self.get_value(state)  # row is empty
                else:
                    return 0  # lost, win prob 0.
        # diag
        if (state[0] == state[4] == state[8]):
            if not self.learn:
                print("diag")
            # three in a row
            if state[0] == self.marker:
                return 1  # won, win prop 1.
            elif state[0] == " ":
                return self.get_value(state)  # row is empty
            else:
                return 0  # lost, win prob 0.
        # anti-diag
        if (state[2] == state[4] == state[6]):
            # three in a row
            if state[2] == self.marker:
                return 1  # won, win prop 1.
            elif state[2] == " ":
                return self.get_value(state)  # row is empty
            else:
                return 0  # lost, win prob 0.
        return self.get_value(state)