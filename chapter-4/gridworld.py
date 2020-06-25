import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap

W = LinearSegmentedColormap.from_list('w',["w", "w"], N=256)

ACTIONS = {
    0: [1, 0],   # north
    1: [-1, 0],  # south
    2: [0, -1],  # west
    3: [0, 1],   # east
}

class GridWorld:
    def __init__(self, size=4):
        """
        A gridworld environment with absorbing states at [0, 0] and [size - 1, size - 1].
        Args:
            size (int): the dimension of the grid in each direction
            cell_reward (float): the reward return after extiting any non absorbing state
        """
        self.state_value = np.zeros((size, size))
        return

    def reset(self):
        self.state_value = np.zeros((size, size))
        return

    def step(self, state, action):
        # is terminal state?
        size = len(self.state_value) - 1
        if (state == (0, 0)) or (state == (size, size)):
            return state, 0

        s_1 = (state[0] + action[0], state[1] + action[1])
        reward = -1
        # out of bounds north-south
        if s_1[0] < 0 or s_1[0] >= len(self.state_value):
            s_1 = state
        # out of bounds east-west
        elif s_1[1] < 0 or s_1[1] >= len(self.state_value):
            s_1 = state

        return s_1, reward

    def render(self, title=None):
        """
        Displays the current value table of mini gridworld environment
        """
        size = len(self.state_value) if len(self.state_value) < 20 else 20
        fig, ax = plt.subplots(figsize=(size, size))
        if title is not None:
            ax.set_title(title)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        return sn.heatmap(self.state_value, annot=True, fmt=".1f", cmap=W, linewidths=1, linecolor="black", cbar=False)

    def bellman_expectation(self, state, probs, discount):
        """
        Makes a one step lookahead and applies the bellman expectation equation to the state self.state_value[state]
        Args:
            state (Tuple[int, int]): the x, y indices that define the address on the value table
            probs (List[float]): transition probabilities for each action
            in_place (bool): if False, the value table is updated after all the new values have been calculated.
                             if True the state [i, j] will new already new values for the states [< i, < j]
        Returns:
            (numpy.ndarrray): the new value for the specified state
        """
        # absorbing state
        value = 0
        for c, action in ACTIONS.items():
            s_1, reward = self.step(state, action)
            value += probs[c] * (reward + discount * self.state_value[s_1])
        return value

    def bellman_optimality(self, state, action, probs):
        """
        Makes a one step lookahead and applies the bellman optimality equation.
        Args:
            state (Tuple[int, int]): the x, y indices that define the address on the value table
            probs (List[float]): transition probabilities for each action
            in_place (bool): if False, the value table is updated after all the new values have been calculated.
                             if True the state [i, j] will new already new values for the states [< i, < j]
        Returns:
            (numpy.ndarrray): the new value for the specified state
        """
        # absorbing state
        raise NotImplementedError