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

ACTIONS = list(range(-5, 6))
CAR_RENTAL_COST = 10
CAR_MOVE_COST = 2
MAX_CARS = 20
MAX_MOVE = 5
REQUEST_A_LAMBDA = 3
RETURN_A_LAMBDA = 3
REQUEST_B_LAMBDA = 4
RETURN_B_LAMBDA = 2
DISCOUNT = 0.9


class CarRental:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self._poisson_prob = {}
        self._P1, self._R1 = self.precompute_model(REQUEST_A_LAMBDA, RETURN_A_LAMBDA)
        self._P2, self._R2 = self.precompute_model(REQUEST_B_LAMBDA, RETURN_B_LAMBDA)
        self.state_values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        return

    def step(self, state, action):
        morning_n1 = state[0] - action
        morning_n2 = state[1] + action
        new_state = (morning_n1, morning_n2)
        reward = self.get_reward(new_state)
        return new_state, reward

    def render(self):
        # plot value table
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(self.state_values, ax=ax[0])
        ax[0].set_ylim(0, 20)
        ax[0].set_title("Value table V_π")
        # plot policy
        sns.heatmap(self.policy, vmin=-5, vmax=5, cmap="RdBu", ax=ax[1])
        ax[1].set_ylim(0, 20)
        ax[1].set_title("Policy π")
        plt.show()
        return fig, ax

    def get_transition_probability(self, state, new_state):
        return self._P1[(state[0], new_state[0])] * self._P2[(state[1], new_state[1])]

    def get_reward(self, state):
        return self._R1[state[0]] + self._R2[state[1]]

    def get_valid_action(self, state, action):
        cars_at_1, cars_at_2 = state
        action = max(-cars_at_2, min(action, cars_at_1))
        action = max(-MAX_MOVE, min(MAX_MOVE, action))
        return action

    def poisson_probability(self, n, lam):
        key = (n, lam)
        if key not in self._poisson_prob:
            self._poisson_prob[key] = math.exp(-lam) * (math.pow(lam, n) / math.factorial(n))
        return self._poisson_prob[key]

    def precompute_model(self, lambda_requests, lambda_dropoffs):
        P, R = {}, {}
        requests = 0
        for requests in range(26):
            request_prob = self.poisson_probability(requests, lambda_requests)
            for n in range(27):
                if n not in R:
                    R[n] = 0.
                R[n] += CAR_RENTAL_COST * request_prob * min(requests, n)
            dropoffs = 0
            for dropoffs in range(26):
                dropoffs_prob = self.poisson_probability(dropoffs, lambda_dropoffs)
                for n in range(26):
                    satisfied_requests = min(requests, n)
                    new_n = max(0, min(MAX_CARS, n + dropoffs - satisfied_requests))
                    if (n, new_n) not in P:
                        P[(n, new_n)] = 0.
                    P[(n, new_n)] += request_prob * dropoffs_prob
        return P, R

    def bellman_expectation(self, state, action):
        """
        Args:
            state (Tuple[int, int]): a tuple storing the number of available locations, respectively at A and B
            action (int): The key to the action dict
            value (float): discount factor for future rewards
        Returns:
            (float): the value of the current state pair
        """
        action = self.get_valid_action(state, action)
        (morning_n1, morning_n2), r = self.step(state, action)

        state_value = -CAR_RENTAL_COST * abs(action)
        for new_n1 in range(21):
            for new_n2 in range(21):
                p = self.get_transition_probability((morning_n1, morning_n2), (new_n1, new_n2))
                state_value += p * (r + DISCOUNT * self.state_values[new_n1, new_n2])
        return state_value

    def policy_evaluation(self, theta=1e-3):
        """
        Computes the true value table for the current policy using iterative policy evaluation
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
                    action = self.policy[available_A, available_B]
                    # check for invalid actions:
                    if state[0] - action > 20 or state[1] + action > 20:
                        continue
                    if state[0] < action:  # not enough cars in A to move
                        continue
                    if state[1] < -action:  # not enough cars in B to move
                        continue
                    new_state_value = self.bellman_expectation(state, action)
                    # according the the original lisp code, the evaluation is performed asynchronously:
                    # http://incompleteideas.net/book/code/jacks.lisp
                    new_values[state] = new_state_value
            delta = np.max(np.abs(self.state_values - new_values))
            print("\t\tValue delta {:.5f}\t".format(delta), end="\r")
            if delta < theta:
                return new_values
            self.state_values = new_values.copy()
            new_values = np.empty_like(self.state_values)

        raise ValueError("The value table did not converge. Check your inputs or look for bugs.")

    def policy_improvement(self):
        """
        Makes one step of policy improvement using a greedy policy.
        Args:
            values (numpy.ndarray):
            policy (numpy.ndarray):
            discount (float):
        Returns:
            (numpy.ndarray): The improved policy
        """
        new_policy = np.empty_like(self.policy)
        for available_A in range(MAX_CARS + 1):
            for available_B in range(MAX_CARS + 1):
                state = (available_A, available_B)
                best_value = self.bellman_expectation(state, self.policy[available_A, available_B])
                best_action = self.policy[available_A, available_B]
                for action in ACTIONS:
                    value = self.bellman_expectation(state, action)
                    if value > best_value:
                        best_value = value
                        best_action = action
                new_policy[available_A, available_B] = best_action
        improved = (self.policy != new_policy).any()
        self.policy = new_policy
        return improved

    def policy_iteration(self, plot=False):
        """
        Computes the optimal policy π* using policy iteration.
        Convergence is guaranteed since the MDP has only a finite number of policies.
        Note that the optimal policy might not be unique.
        Args:
            env (CarRental): The CarRental environment
            policy (numpy.ndarray): an optional starting policy. If omitted the policy is initialised with zeros everywhere
        Returns:
            (numpy.ndarray): The optimal policy
        """
        # policy evaluation/improvement loop
        iteration = 1
        while True:
            # log
            print("Iterating through policy {}".format(iteration))

            # policy evaluation to get the new value table
            print("\tEvaluating policy {}".format(iteration))
            new_values = self.policy_evaluation()

            # policy improvement to adjust the policy to the new value table
            print("\tImproving policy {}".format(iteration))
            improved = self.policy_improvement()

            # plot
            if plot:
                self.render()

            # has π converged to π*? I.e. is the current policy stable?
            if not improved:
                return

            iteration += 1
        raise ValueError("The policy did not converge. Check your inputs or look for bugs.")

if __name__ == "__main__":
    env = CarRental()
#     env.policy_evaluation()
#     env.policy_improvement()
#     env.render()
    env.policy_iteration(True)