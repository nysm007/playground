### MDP Policy Iteration 
### You must not change the arguments and output types of the given function. 
### You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *

np.set_printoptions(precision=3)


def policy_iteration(env, gamma, max_iteration, tol):
    """Implement Policy iteration algorithm.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    env: OpenAI env.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor. 
    max_iteration: int
            The maximum number of iterations to run before stopping. 
    tol: float
            Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)
    ############################
    #         YOUR CODE        #
    ############################
    for n in range (max_iteration):
        V = policy_evaluation(env, policy, gamma, max_iteration, tol)
        policy, policy_stable = policy_improvement(env, V, policy, gamma)
        
        if policy_stable:
            print("[INFO] Policy iteration converges at {} iteration.".format(n))
            break

    if not policy_stable:
        print("[INFO] Policy iteration fails to converge after {} iterations.".format(max_iteration))

    return V, policy


def policy_evaluation(env, policy, gamma, max_iteration, tol):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    env: OpenAI env. 
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor. 
    policy: np.array
            The policy to evaluate. Maps states to actions.
    max_iteration: int
            The maximum number of iterations to run before stopping. 
    tol: float
            Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
            The value function from the given policy.
    """
    ############################
    #         YOUR CODE        #
    ############################
    delta = 0
    theta = tol
    V = np.zeros(env.nS)

    for n in range (max_iteration):
        delta = 0
        temp = V[:]
        V_next = np.zeros(env.nS)
        for i in range (env.nS):
            t = V[i]
            V_next[i] = env.P[i][policy[i]][0][2] + gamma * V[env.P[i][policy[i]][0][1]]
            if (delta < np.abs(t - V_next[i])):
                delta = np.abs(t - V_next[i])
        
        V = V_next
        if delta < theta:
            break

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """Given the value function from policy, improve the policy.

    Parameters
    ----------
    env: OpenAI env
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    
    value_from_policy: np.ndarray
            The value calculated from the policy
    policy: np.array
            The previous policy.
    gamma: float
            Discount factor. 

    Returns
    -------
    new policy: np.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    stable policy: bool
            True if the optimal policy is found, otherwise false 
    """
    ############################
    #         YOUR CODE        #
    ############################
    policy_stable = True
    temp = policy[:]

    for i in range (env.nS):
        t = policy[i]
        policy[i] = np.argmax([env.P[i][j][0][2] + gamma * value_from_policy[env.P[i][j][0][1]] for j in range (6)])
        
        if policy[i] != t:
            policy_stable = False
    
    return policy, policy_stable



def render_episode(env, policy):
    """Run one episode for given policy on environment. 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
        print(ob,rew)
    assert done
    env.render()
    print("Episode reward: %f" % episode_reward)


def avg_performance(env, policy):
    """ Evaluate the average rewards 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    sum_reward = 0.
    episode = 100
    max_iteration = 6000
    for i in range(episode):
        done = False
        ob = env.reset()

        for j in range(max_iteration):
            a = policy[ob]
            ob, reward, done, info = env.step(a)
            sum_reward += reward
            if done:
                break

    return sum_reward / i

def main():
    """
    You can test your game when you finish setting up your environment.
    Input range from 0 to 5:
        0 : South (Down)
        1 : North (Up)
        2 : East (Right)
        3 : West (Left)
        4: Pick up
        5: Drop off
    """

    GAME = "Assignment1-Taxi-v2"
    env = gym.make(GAME)
    n_state = env.observation_space.n
    n_action = env.action_space.n
    env = Monitor(env, "taxi_simple", force=True)

    s = env.reset()
    steps = 100
    for step in range(steps):
        env.render()
        action = int(input("Please type in the next action:"))
        s, r, done, info = env.step(action)
        print(s)
        print(r)
        print(done)
        print(info)

    # close environment and monitor
    env.close()


if __name__ == '__main__':
    # Uncomment the Main function and comments the rest to test your program
    # main()
    env = gym.make("Assignment1-Taxi-v2")
    print(env.__doc__)

    V_pi, policy_pi = policy_iteration(env, gamma=0.95, max_iteration=6000, tol=1e-5)
    scores = avg_performance(env, policy_pi)
    print("[1] Optimal value function: \n", V_pi)
    print("[2] Optimal policy: \n", policy_pi)
    print("[3] Average rewards after 100 episodes:", scores)
 
