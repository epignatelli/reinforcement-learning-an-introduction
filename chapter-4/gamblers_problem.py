# The MIT License (MIT)
# Copyright (c) {{ 2020 }} {{ Eduardo Pignatelli }}
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math


P_H = 0.25
THETA = 1e-5
WIN = 100


class GamblersProblem:
    def __init__(self):
        self.values = np.zeros((101,), np.int32)
        self.values[100] = 1
        self.policy = np.zeros_like(self.values)
        return

    def render(self):
        print(self.values)
        print(self.policy)

    def state_space(self):
        return range(1, WIN + 1)

    def action_space(self, state=WIN + 1):
        return range(min(state, WIN - state) + 1)

    def bellman_expectation(self, state, action):
        # evaluate the expected return - model is stochastic
        if (state == 60 and action == 40):
            print("YES")
        win_reward = int(state + action >= WIN)
        value_win = (P_H * (win_reward + self.values[state + action]))
        value_lose = (1 - P_H) * self.values[state - action]
        return value_win + value_lose

    def policy_evaluation_step(self):
        """
        Updates the value table according to the current policy
        """
        new_values = np.empty_like(self.values)
        for state in self.state_space():
            new_values[state] = self.bellman_expectation(state, self.policy[state])
#             print(state, values[state])
        converged = abs((new_values - self.values)).max() < THETA
        self.values = new_values.copy()
        return converged

    def policy_improvement(self):
        """
        Updates the policy according to the current value table
        """
        new_policy = np.empty_like(self.policy)
        for state in self.state_space():
            best_value, best_action = -float("inf"), self.policy[state]
            for action in self.action_space(state):
                value = self.bellman_expectation(state, action)
#                 print(state, action, value)
                if value > best_value:
                    best_value, best_action = value, action
            new_policy[state] = best_action
        converged = (new_policy == self.policy).all()
        self.policy = new_policy.copy()
        return converged

    def value_iteration(self):
        """
        Searches for the optimal state value table using value iteration to THETA
        """
        iteration = 0
        while True:
            print("Iterating through value table {}".format(iteration))
            v_converged = self.policy_evaluation_step()
            p_converged = self.policy_improvement()

            if iteration == 100 or (v_converged and p_converged):
                print("Stopped at {}".format(iteration))
                return

            self.render()

            iteration += 1
        raise ValueError("The policy did not converge. Check your inputs or look for bugs.")


if __name__ == "__main__":
    env = GamblersProblem()
    env.value_iteration()