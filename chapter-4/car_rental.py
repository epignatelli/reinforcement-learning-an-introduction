try:
    import jax.numpy as np
except ImportError as e:
    print(repr(e))
    import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sns
import math


ACTIONS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
CAR_RENTAL_COST = 10
CAR_MOVE_COST = 2
MAX_CARS = 20
MAX_MOVE = 5
REQUEST_1_LAMBDA = 3
DROPOFF_1_LAMBDA = 3
REQUEST_2_LAMBDA = 4
DROPOFF_B_LAMBDA = 2
DISCOUNT = 0.9


class CarRental:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self._poisson_prob = {}
        self.state_values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), np.int32)
        self._probs_1, self._rewards_1 = self.precompute_model(REQUEST_1_LAMBDA, DROPOFF_1_LAMBDA)
        self._probs_2, self._rewards_2 = self.precompute_model(REQUEST_2_LAMBDA, DROPOFF_B_LAMBDA)
        return

    def step(self, state, action):
        morning_n1 = int(state[0] - action)
        morning_n2 = int(state[1] + action)
        new_state = (morning_n1, morning_n2)
        reward = self.get_reward(new_state)
        return new_state, reward

    def render(self):
        # plot value table
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(self.state_values, cmap="gray", ax=ax[0])
        ax[0].set_ylim(0, MAX_CARS)
        ax[0].set_title("Value table V_π")
        # plot policy
        cmaplist = [plt.cm.RdBu(i) for i in range(plt.cm.RdBu.N)]
        dRbBu = matplotlib.colors.LinearSegmentedColormap.from_list('dRdBu', cmaplist, plt.cm.RdBu.N)
        sns.heatmap(self.policy, vmin=-5, vmax=5, cmap=dRbBu, ax=ax[1], cbar_kws={"ticks": ACTIONS, "boundaries": ACTIONS})
        ax[1].set_ylim(0, MAX_CARS)
        ax[1].set_title("Policy π")
        plt.show()
        return fig, ax

    def get_transition_probability(self, state, new_state):
        return self._probs_1[(state[0], new_state[0])] * self._probs_2[(state[1], new_state[1])]

    def get_reward(self, state):
        return self._rewards_1[state[0]] + self._rewards_2[state[1]]

    def get_valid_action(self, state, action):
        cars_at_1, cars_at_2 = state
        # Jack can't move more cars than he has available
        action = max(-cars_at_2, min(action, cars_at_1))
        # Jack can move at most 5 cars
        action = max(-MAX_MOVE, min(MAX_MOVE, action))
        return action

    def get_available_actions(self, state):
        return list(range(max(-MAX_CARS, - state[1]), min(MAX_CARS, state[0]) + 1))

    def poisson_probability(self, n, lam):
        key = (n, lam)
        if key not in self._poisson_prob:
            self._poisson_prob[key] = math.exp(-lam) * (math.pow(lam, n) / math.factorial(n))
        return self._poisson_prob[key]

    def precompute_model(self, lambda_requests, lambda_dropoffs):
        P, R = {}, {}
        requests = 0
        for requests in range(MAX_CARS + max(ACTIONS) + 1):
            request_prob = self.poisson_probability(requests, lambda_requests)
            for n in range(MAX_CARS + max(ACTIONS) + 1):
                if n not in R:
                    R[n] = 0.
                R[n] += CAR_RENTAL_COST * request_prob * min(requests, n)
            dropoffs = 0
            for dropoffs in range(MAX_CARS + max(ACTIONS) + 1):
                dropoffs_prob = self.poisson_probability(dropoffs, lambda_dropoffs)
                for n in range(MAX_CARS + max(ACTIONS) + 1):
                    satisfied_requests = min(requests, n)
                    new_n = max(0, min(MAX_CARS, n + dropoffs - satisfied_requests))
                    if (n, new_n) not in P:
                        P[(n, new_n)] = 0.
                    P[(n, new_n)] += request_prob * dropoffs_prob
        return P, R

    def bellman_expectation(self, state, action):
        """
        Solves the bellman expectation equation for given state
        V(s) = p(s, r | s' π(s)) * (R(s) + γ * V(s'))
        Args:
            state (Tuple[int, int]): a tuple storing the number of available locations, respectively at A and B
            action (int): The key to the action dict
        Returns:
            (float): the value V(s) of the current state pair
        """
        action = self.get_valid_action(state, action)
        (morning_n1, morning_n2), r = self.step(state, action)

        state_value = -CAR_MOVE_COST * abs(action)
        for new_n1 in range(MAX_CARS + 1):
            for new_n2 in range(MAX_CARS + 1):
                p = self.get_transition_probability((morning_n1, morning_n2), (new_n1, new_n2))
                state_value += p * (r + DISCOUNT * self.state_values[new_n1, new_n2])
        return state_value

    def policy_evaluation(self, theta=1e-3):
        """
        Computes the true value table for the current policy using iterative policy evaluation.
        At the end of the process it updates the state-value table with the newly computed value function.
        Args:
            env (CarRental): The CarRental environment, which contains the model dynamics, the active policy and the state-value table
        Returns:
            (numpy.ndarray): The value function of the current policy stored as a 2D array
        """
        new_values = np.empty_like(self.state_values)
        while True:
            # for each state s ∈ S
            for available_A in range(MAX_CARS + 1):
                for available_B in range(MAX_CARS + 1):
                    state = (available_A, available_B)
                    # a = π(s)
                    action = self.policy[available_A, available_B]
                    # V(s) = p(s, r | s' π(s)) * (R(s) + γ * V(s'))
                    new_state_value = self.bellman_expectation(state, action)
                    # according the the original lisp code, the evaluation is performed asynchronously:
                    # http://incompleteideas.net/book/code/jacks.lisp
                    new_values[state] = new_state_value
            delta = np.max(np.abs(self.state_values - new_values))
            print("\t\tValue delta {:.5f}\t\t ".format(delta), end="\r")
            self.state_values = new_values.copy()
            new_values = np.empty_like(self.state_values)
            if delta < theta:
                print()
                return

        raise ValueError("The value table did not converge. Check your inputs or look for bugs.")

    def policy_improvement(self):
        """
        Makes one step of policy improvement following a greedy policy.
        For each state of the model, it iterates through all the feasible actions and finds the greediest one.
        The current policy is updated synchronously for each state, i.e. only after all the states have been visited.
        Returns:
            (bool): True if the policy has not improved
        """
        new_policy = np.empty_like(self.policy)
        # for each state s ∈ S
        for available_A in range(MAX_CARS + 1):
            for available_B in range(MAX_CARS + 1):
                state = (available_A, available_B)
                best_action = self.policy[state]
                best_value = -float("inf")
                for action in self.get_available_actions(state):
                    value = self.bellman_expectation(state, action)
#                     print(state, action, value)
                    if value > best_value:
                        best_value = value
                        best_action = action
                new_policy[available_A, available_B] = best_action
        converged = (new_policy == self.policy).all()
        self.policy = new_policy.copy()
        return converged

    def policy_iteration(self, plot=False):
        """
        Computes the optimal policy π* using policy iteration.
        Convergence is guaranteed since the MDP has only a finite number of policies.
        Note that the optimal policy might not be unique.
        Args:
            plot (bool): If true, self.render() will be called at each evaluation/iteration step
        Returns:
            (numpy.ndarray): The optimal policy
        """
        # policy evaluation/improvement loop
        iteration = 1
        if plot:
            self.render()
        while True:
            # log
            print("Iterating through policy {}".format(iteration))

            # policy evaluation to update the value table
            print("\tEvaluating policy {}".format(iteration))
            self.policy_evaluation()

            # policy improvement to update the current policy, based on the new value table
            print("\tImproving policy {}".format(iteration))
            converged = self.policy_improvement()

            # has π converged to π*? I.e. is the current policy stable?
            if converged:
                print("Policy is stable. π converged to π*")
                return

            # plot
            if plot:
                self.render()

            iteration += 1
        raise ValueError("The policy did not converge. Check your inputs or look for bugs.")

if __name__ == "__main__":
    env = CarRental()
    env.policy_iteration(True)