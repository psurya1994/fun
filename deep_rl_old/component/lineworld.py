"""
o - o - o - x - o - o - t
x => start location, o => unoccupied state, t => terminal state
"""


import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class LineWorld(gym.Env):
    """
    """
    def __init__(self, n=7):

        # Initialzing required parameters
        self.update_count = 0
        self.n = n # Length of the chain
        self.state = (n-1) // 2  # Start at middle of the chain
        self.action_space = spaces.Discrete(2) # Number of actions: 2 - [0: left, 1: right]
        # self.observation_space = spaces.Discrete(self.n) # number of states is equal to chain length
        self.observation_space = spaces.Box(low=np.zeros(n), high=np.ones(n), dtype=np.uint8) # number of states is equal to chain length
        
        # Setting reward values
        self.step_reward = -1
        
        self.seed() # not sure what this does, so not changing it

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        
        # Making sure valid action is chosen
        assert self.action_space.contains(action)

        self.update_count += 1

        if(self.update_count > 40):
            temp = np.zeros(self.n)
            temp[self.state] = 1
            return temp, self.step_reward, True, {}

        # Stepping along on the chain
        if(action == 0):
            self.state = self.state - 1
        else:
            self.state = self.state + 1

        # Because this is a continuing case
        if((self.state == self.n-1) or (self.state == 0)):
            done = True
            reward = self.step_reward
        else:
            done = False
            reward = self.step_reward

        temp = np.zeros(self.n)
        temp[self.state] = 1

        return temp, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.update_count = 0 
        self.state = (self.n - 1) // 2
        temp = np.zeros(self.n)
        temp[self.state] = 1
        return temp