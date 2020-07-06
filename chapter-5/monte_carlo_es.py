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

