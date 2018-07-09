### Model free learning using Q-learning and SARSA
### You must not change the arguments and output types of the given function. 
### You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *
import random
import matplotlib.pyplot as plt

def QLearning(env, num_episodes, gamma, lr, e):
    """Implement the Q-learning algorithm following the epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int 
      Number of episodes of training.
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    #         YOUR CODE        #
    ############################
    if __name__ == "__main__":
        global q_rewards
        global q_nums
    else:
        q_rewards = list()
        q_nums = list()

    Q = np.ones((env.nS, env.nA))
    for i in range (env.nS):
        for j in range(env.nA):
            if env.P[i][j][0][3]:
                Q[i][j] = 0

    total_reward = 0
    for n in range (num_episodes):
        #total_reward = 0
        cnt = 0
        s = np.random.randint(env.nS)
        if random.random() > e:
            a = np.argmax(Q[s])
        else:
            a = random.choice([i for i in range(env.nA) if i != np.argmax(Q[s])])

        while True:
            R = env.P[s][a][0][2]
            total_reward += R
            cnt += 1
            s_next = env.P[s][a][0][1]
            if random.random() > e:
                a_next = np.argmax(Q[s_next])
            else:
                a_next = random.choice([i for i in range (env.nA) if i != np.argmax(Q[s_next])])

            Q[s][a] += lr * (R + gamma * max(Q[s_next][j] for j in range (6)) - Q[s][a])
            s = s_next
            a = a_next

            if env.P[s][a][0][3]:
                break

        episode_reward = total_reward / (n + 1)
        q_rewards.append(episode_reward)
        q_nums.append(cnt)

    return Q


def SARSA(env, num_episodes, gamma, lr, e):
    """Implement the SARSA algorithm following epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function 
    num_episodes: int 
      Number of episodes of training
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    #         YOUR CODE        #
    ############################
    if __name__ == "__main__":
        global s_rewards
        global s_nums
    else:
        s_rewards = list()
        s_nums = list()

    Q = np.ones((env.nS, env.nA))
    for i in range (env.nS):
        for j in range (env.nA):
            if env.P[i][j][0][3]:
                Q[i][j] = 0

    total_reward = 0
    for n in range (num_episodes):
        #total_reward = 0
        cnt = 0
        s = np.random.randint(env.nS)
        if random.random() > e:
            a = np.argmax(Q[s])
        else:
            a = random.choice([i for i in range (env.nA) if i != np.argmax(Q[s])])

        while True:
            s_next = env.P[s][a][0][1]
            R = env.P[s][a][0][2]
            total_reward += R
            cnt += 1

            if random.random() > e:
                a_next = np.argmax(Q[s_next])
            else:
                a_next = random.choice([i for i in range (env.nA) if i != np.argmax(Q[s])])

            Q[s][a] += lr * (R + gamma * Q[s_next][a_next] - Q[s][a])
            s = s_next
            a = a_next
            
            if env.P[s][a][0][3]:
                break
   
        episode_reward = total_reward / (n + 1)
        s_rewards.append(episode_reward)
        s_nums.append(cnt)

    return Q


def render_episode_Q(env, Q):
    """Renders one episode for Q functionon environment.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. 
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print ("Episode reward: %f" %episode_reward)



def main():
    env = gym.make("Assignment1-Taxi-v2")
    Q_QL = QLearning(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    Q_Sarsa = SARSA(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    #render_episode_Q(env, Q_QL)
    plt.plot(q_rewards)
    plt.show()
    '''
    for i in range(len(Q_QL)):
        print([Q_QL[i][j] for j in range(len(Q_QL[0]))])
    '''
    plt.plot(s_rewards)
    plt.show()
    '''
    for i in range(len(Q_QL)):
        print([Q_Sarsa[i][j] for j in range(len(Q_Sarsa[0]))])
    '''
    plt.plot(q_nums)
    plt.show()
    plt.plot(s_nums)
    plt.show()

if __name__ == '__main__':
    q_rewards = list()
    s_rewards = list()
    q_nums = list()
    s_nums = list()
    main()
