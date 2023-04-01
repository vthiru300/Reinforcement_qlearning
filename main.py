#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
from gridworld import *
from analyze import *


# In[12]:


if __name__ == '__main__':
    #runtime = 100
    filename_state = 'state'
    filename_qtable = 'qtable'
    filename_results = 'results'
    number_of_robots = 2
    number_of_interestpoints = 2
    dimension = 4
    observation_shape =(3,84,84)
    env = GridWorld(number_of_robots, number_of_interestpoints, dimension, observation_shape)

    number_of_episodes = 100
    env.train(number_of_episodes)
    analyze_convergence()
    env.visualize_training(filename_state, number_of_episodes)
    #env.evaluate(filename_qtable)
    #env.visualize(filename_results)
    env.close()
    fig, axs = plt.subplots(2)
    fig.suptitle('Plot Steps and Reward Buffer')
    axs[0].plot(range(len(rew_buf)), rew_buf)
    axs[1].plot(range(len(steps)), steps)
    #plt.plot(range(len(rew_buf)),rew_buf)
    plt.show()


# # Final Comments
# We can conclude from the above plots that the number of rewards keeps rising and essentially stays the same after a few episodes. Inconsistencies exist in the rewards between related episodes. 2. After each episode, the steps required to reach the interest point get progressively shorter, however there may be some variation. 3. The algorithm converges at a faster pace when the learning rate and discount factor are high and low, respectively. This results in fewer steps being needed and higher rewards. While the method takes longer to converge when the learning rate and discount rate are low and high, necessitating more steps and fewer rewards.
