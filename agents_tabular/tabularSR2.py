"""
Implementation of successor representation.

Code adopted from: https://github.com/awjuliani/successor_examples

Need to fix:
- Not sure why the TD error goes up when we plot. Needs to go down.
"""

import numpy as np

def onehot(value, max_value):
    vec = np.zeros(max_value)
    vec[value] = 1
    return vec

def twohot(value, max_value):
    vec_1 = np.zeros(max_value)
    vec_2 = np.zeros(max_value)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])

def mask_grid(grid, blocks, mask_value=-100):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i,j] in blocks:
                grid[i,j] = mask_value
    grid = np.ma.masked_where(grid == mask_value, grid)
    return grid


class TabularSuccessorAgent(object):
    def __init__(self, env, learning_rate, gamma):
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.M = np.stack([np.identity(self.state_size) for i in range(self.action_size)])
        self.w = np.zeros([self.state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
        
    def Q_estimates(self, state, goal=None):
        # Generate Q values for all actions.
        if goal == None:
            goal = self.w
        else:
            goal = onehot(goal, self.state_size)
        return np.matmul(self.M[:,state,:],goal)
    
    def sample_action(self, state, goal=None, epsilon=0.0):
        # Samples action using epsilon-greedy approach
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
        return action
    
    def update_w(self, current_exp):
        # A simple update rule
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error    
        self.w[s_1] = r # added by me, deterministic rewards
        return error
    
    def update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        s_a_1 = next_exp[1]
        r = current_exp[3]
        d = current_exp[4]
        I = onehot(s, self.state_size)
        
        if d:            
            td_error = (I + self.gamma * onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

