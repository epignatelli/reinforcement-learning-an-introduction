from gridworld import GridWorld, ACTIONS
import matplotlib
import numpy as np


def policy_evaluation(env, policy=None, steps=1, discount=1., in_place=False):
    """
    Args:
        policy (numpy.array): a numpy 3-D numpy array, where the first two dimensions identify a state and the third dimension identifies the actions.
                              The array stores the probability of taking each action.
        steps (int): the number of iterations of the algorithm
        discount (float): discount factor for the bellman equations
        in_place (bool): if False, the value table is updated after all the new values have been calculated.
             if True the state [i, j] will new already new values for the states [< i, < j]
    """
    if policy is None:
        # uniform random policy
        policy = np.ones((*env.state_value.shape, len(ACTIONS))) * 0.25

    for k in range(steps):
        # cache old values if not in place
        values = env.state_value if in_place else np.empty_like(env.state_value)
        for i in range(len(env.state_value)):
            for j in range(len(env.state_value[i])):
                # apply bellman expectation equation to each state
                state = (i, j)
                value = env.bellman_expectation(state, policy[i, j])
                values[i, j] = value * discount
        # set the new value table
        env.state_value = values
    return env.state_value


if __name__ == "__main__":
    for k in [1, 2, 3, 10, 1000]:
        env = GridWorld(4)
        value_table = policy_evaluation(env, steps=k, in_place=False)
        env.render()
