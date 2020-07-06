from blackjack import Blackjack
import numpy as np


def mc_policy_evaluation(env, policy, iterations=10000, first_visit=True):
    counts = np.ones_like(env.values) * 1e-9
    for k in range(iterations):
        # run episode
        print("MC iteration {}/{}\t".format(k + 1, iterations), end="\r")
        # select starting state
        obs, done = env.reset()
        while not done:
            # cache starting state
            starting_state = obs
            # iterate on each action and pick the one tha gives the highest return
            obs, reward, done = env.step(policy[obs])
            # if we use first-visit MC, we skip the first time we see the state
            if not (first_visit and counts[starting_state] == 0):
                # update the action-value state
                # note that we sum the value and average only at the end of the iteration
                env.values[starting_state] += reward
                policy[obs] = np.argmax()
            counts[starting_state] += 1
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
