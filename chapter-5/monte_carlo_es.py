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


from blackjack import Blackjack, ACTIONS
import numpy as np


DISCOUNT = 1.


def monte_carlo_es(env, iterations=1e5, first_visit=True):
    # initialise visits counter
    counts = np.ones(env.observation_space) * 1e-9

    # initialise policy and returns
    policy = np.zeros(env.observation_space)
    policy[20:] = 1
    returns = np.ones(env.observation_space)

    for k in range(iterations):
        # select starting state
        obs, done = env.reset()
        G = 0.
        best_q = - float("inf")
        best_action = None
        while not done:
            for action in ACTIONS:

                # cache the current state
                starting_state = obs
                # interact, either hit or stick
                obs, reward, done = env.step()
                # skip if we use first visit and the state is unvisited
                if not (first_visit and counts[starting_state] == 0):
                    G += DISCOUNT * G + reward
                # register the new visit
                counts[obs] += 1


if __name__ == "__main__":
    env = Blackjack()
    optimal_policy = monte_carlo_es(env)
    env.render()

