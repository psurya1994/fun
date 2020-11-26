
"""
Tabular grid world environment.
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class PixelGridWorld(gym.Env):
    """Grid world environment with variable size grid
    """
    def __init__(self, n=5, loc_r=[12], loc_t=[12], init_state=None, loc_w=None, p=0):
        '''Constructor object for the grid worlds
        Location is measured from the bottom left of the screen.
        
        Inputs:
            n -> (int) grid would be square of size n
            loc_r -> (list of int) reward locations
            loc_t -> (list of int) location of terminal states
            init_state -> (int) start state
            loc_w -> (list of int) location of walls
        '''

        # Initialzing required parameters
        self.update_count = 0
        self.loc_r = loc_r
        self.loc_w = loc_w
        self.loc_t = loc_t
        self.cs = 100 # cell size for rendering
        self.p = p
        self.horizon=100

        self.n = n # side of grid world
        self.observation_space = spaces.box.Box(low=0,high=255,shape=(80,80,3),dtype='uint8')
        # self.observation_space = spaces.Discrete(self.n * self.n) # number of states is equal to area of gridworld
        self.action_space = spaces.Discrete(4) # Number of actions: 4 - [0: up, 1: right, 2: left, 3: down]
        self.init = init_state
        if(init_state is None):
            self.state = 0  # Start at beginning of the chain (bottom left)
        else:
            self.state = init_state

        # Setting up rewards, terminal states and walls
        self.R = -np.ones(self.n * self.n) # storing the rewards in this vector
        if(loc_r is not None):
            self.R[loc_r] = np.zeros(len(loc_r))

        self.T = np.zeros(self.n * self.n) # storing boolean values of terminal states
        if(loc_t is not None):
            self.T[loc_t] = np.ones(len(loc_t))

        self.W = np.zeros(self.n * self.n)
        if(loc_w is not None):
            self.W[loc_w] = np.ones(len(loc_w))
        
        self.seed()

    def seed(self, seed=None):
        """
        Setting the seed of the agent for replication
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def next_state_pred(self, action):
        """
        Function to predict next state when you take an action
        """
        if(self.check_boundaries(action)):
            return self.state
        if(action == 0):
            nstate = self.state + self.n
        elif(action == 1):
            nstate = self.state + 1
        elif(action == 2):
            nstate = self.state - self.n
        else:
            nstate = self.state - 1

        return nstate

    def step(self, action):
        '''
        takes action as an argument and returns the next_state, reward, done, info.
        '''
        
        # Making sure valid action is chosen
        assert self.action_space.contains(action)
        
        # If the environment is stocastic, you move equivalent to taking a random action
        if(np.random.rand() <= self.p):
            action = np.random.randint(low=0, high=4)
        
        done = False
        self.update_count += 1
        if(self.update_count >= self.horizon): # NEEDS FIXING
            return self.render(), self.R[self.state], 1, {}
        
        if(self.check_boundaries(action) == True): # you hit a boundary
            return self.render(), self.R[self.state], done, {}

        if(self.W[self.next_state_pred(action)]==1): # you hit a wall
            return self.render(), self.R[self.state], done, {}
            
        # Step in the grid world
        self.state = self.next_state_pred(action)

        # Check for terminal state
        if(self.T[self.state] == 1):
            done = True

        return self.render(), self.R[self.state], done, {}
        
    
    def check_boundaries(self, action):
        """
        Function to check if you're crashing into walls.
        """
        if((self.state < self.n) and (action == 2)): # you hit bottom wall
            return True
        elif((self.state >= self.n*(self.n-1)) and (action == 0)): # you hit upper wall
            return True
        elif(((self.state % self.n) == 0) and (action == 3)): # hit left wall
            return True
        elif(((self.state % self.n) == (self.n-1)) and (action == 1)): # you hit right wall
            return True

        return False        

    def plot(self):
        """
        Function to generate a human readable plot of the world.
        """
        print(f'state = {self.state}')
        for j in range(self.n-1, -1, -1):
            for i in range(0, self.n):
                if(i+self.n*j == self.state):
                    print('H', end = ' ')
                elif(not self.T[i+self.n*j]==0): # terminal state
                    print('T', end=' ')
                elif(not self.R[i+self.n*j]==0): # reward state
                    print('R', end=' ')
                else:
                    print('o', end = ' ')
            print(' ')
        pass
    
    def reset(self, state=0):
        '''
        transitions back to first state
        '''
        self.update_count = 0 
        self.state = state
        return self.render()
    
    def ind2xy(self, ind):
        return self.n - 1 - ind // self.n, ind % self.n
    
    def show(self):
        """
        Function to print cell IDs.
        """
        for i in range(self.n):
            for j in range(self.n):
                print('{0:3d}'.format((self.n - i - 1) * self.n + j), end=' ')
            print('')

    def render(self, mode='rgb_array', printR=True, img_size=40):
        '''Rendering the state of environment
        
        passing for now, need to implement a nice visualization later.
        
        '''
        bias = 4
        cs = (img_size-2*bias) // (self.n)
        h=img_size
        w=img_size
        img = Image.new('RGBA', (h, w), "grey")
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([bias, bias, h-bias, w-bias], fill="black")
        

        for i in range(self.n * self.n):

            y, x = self.ind2xy(i)
            x = x * cs + cs//2 + bias
            y = y * cs + cs//2 + bias
            width = cs // 3

            if(self.T[i]==1): # Plotting terminal state
                draw.ellipse([x-width, y-width, x+width, y+width], fill="yellow")
            if(self.state==i):
                draw.ellipse([x-width, y-width, x+width, y+width], fill="red")
            if(self.W[i]==1):
                draw.rectangle([x-width, y-width, x+width, y+width], fill="black")

        for i in range(1, self.n):
            draw.line([bias+cs*i, 0, bias+cs*i, h], fill="gray", width=1)
            draw.line([0, bias+cs*i, h, bias+cs*i], fill="gray", width=1)

        if(mode=="human"):
            plt.imshow(img)

        elif(mode=="rgb_array"):
            return np.asarray(img)[:,:,:3]