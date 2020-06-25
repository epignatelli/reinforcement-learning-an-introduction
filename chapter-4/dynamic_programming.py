import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap

RdGu = LinearSegmentedColormap.from_list('RdGu',["r", "w", "g"], N=256)
RdWu = LinearSegmentedColormap.from_list('RdWu',["r", "w"], N=256)
W = LinearSegmentedColormap.from_list('w',["w", "w"], N=256)

ACTIONS = {
    0: [1, 0],   # north
    1: [-1, 0],  # south
    2: [0, -1],  # west
    3: [0, 1],   # east
}

class GridWorld:
    def __init__(self, size=4, cell_reward=-1):
        """
        A small square gridworld environment, with absorbing states at [0, 0] and [size - 1, size - 1].
        Args:
            size (int): the dimension of the grid in each direction
            cell_reward (float): the reward return after extiting any non absorbing state
        """
        self.state_value = np.zeros((size, size))
        self.cell_reward = cell_reward

    def bellman_expectation(self, state, probs, in_place=False):
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
        if (state[0] == 0 and state[1] == 0) or (state[0] == len(self.state_value) - 1 and state[1] == len(self.state_value) - 1):
            return self.state_value[state]

        s = tuple(state)
        r = self.cell_reward

        values = []
        for c, a in ACTIONS.items():
            p = probs[c]
            s_1 = (s[0] + a[0], a[1] + s[1])

            # out of bounds north-south
            if s_1[0] < 0 or s_1[0] >= len(self.state_value):
                s_1 = s
            # out of bounds east-west
            elif s_1[1] < 0 or s_1[1] >= len(self.state_value):
                s_1 = s

            value = self.state_value[s_1] + self.cell_reward
            values.append(value)
        return np.mean(values)

    def bellman_optimality(self, state, action, probs):
        raise NotImplementedError

    def policy_evaluation_step(self, policy, discount, in_place):
        """
        Args:
            policy (numpy.array): a numpy 3-D numpy array, where the first two dimensions identify a state and the third dimension identifies the actions.
                                  The array stores the probability of taking each action.
            discount (float): discount factor for the bellman equations
            in_place (bool): if False, the value table is updated after all the new values have been calculated.
                 if True the state [i, j] will new already new values for the states [< i, < j]
        """
        # cache old values if not in place
        values = self.state_value if in_place else np.empty_like(self.state_value)
        for i in range(len(self.state_value)):
            for j in range(len(self.state_value[i])):
                # apply bellman expectation equation to each state
                state = (i, j)
                value = self.bellman_expectation(state, policy[i, j])
                values[i, j] = value * discount
        # set the new value table
        self.state_value = values
        return

    def policy_evaluation(self, policy=None, steps=100, discount=1, in_place=False):
        """
        Args:
            policy (numpy.array): a numpy 3-D numpy array, where the first two dimensions identify a state and the third dimension identifies the actions.
                                  The array stores the probability of taking each action.
            steps (int): the number of iterations of the algorithm
            discount (float): discount factor for the bellman equations
            in_place (bool): if False, the value table is updated after all the new values have been calculated.
                     if True the state [i, j] will new already new values for the states [< i, < j]
        Returns:
            The evaluated value table for the given policy
        """
        if policy is None:
            # uniform random policy
            policy = np.ones((*self.state_value.shape, len(ACTIONS))) * 0.25

        for k in range(steps):
            self.policy_evaluation_step(policy, discount, in_place)
        return env.state_value

    def render(self):
        """
        Displays the current value table of mini gridworld environment
        """
        size = len(self.state_value) if len(self.state_value) < 20 else 20
        fig, ax = plt.subplots(figsize=(size, size))
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        return sn.heatmap(self.state_value, annot=True, fmt=".1f", cmap=W, linewidths=1, linecolor="black", cbar=False)


if __name__ == "__main__":
    env = GridWorld(4)
    env.policy_evaluation(steps=1000, in_place=False)
    env.render()