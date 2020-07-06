from blackjack import Blackjack
import numpy as np


def mc_policy_evaluation(env, policy, iterations=10000, first_visit=True):
    counts = np.ones_like(env.values) * 1e-9
    for k in range(iterations):
        # run episode
        print("MC iteration {}/{}\t".format(k + 1, iterations), end="\r")
        obs, done = env.reset()
        while not done:
            obs, reward, done = env.step(policy[obs])
            # print(env, obs, reward, done)
            if not (first_visit and counts[obs] == 0):
                env.values[obs] += reward
            counts[obs] += 1
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
