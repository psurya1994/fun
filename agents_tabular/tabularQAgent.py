import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from PIL import Image, ImageDraw, ImageFont
from random import sample

class TabularQAgent:
    # Setting things up
    def __init__(self, env, eps=0.2, gamma = 1):
        self.noStates = env.observation_space.n
        self.noActions = env.action_space.n
        self.env = env
        self.Q = np.zeros((self.noStates, self.noActions))
        self.eps = eps
        self.gamma = gamma
        self.updates = 0
        
    def reset(self):
        self.Q = np.zeros((self.noStates, self.noActions))
    
    # Function to do eps-greedy exploration
    def select_action(self, greedy=False):
        if(greedy):
            return np.argmax(self.Q[self.env.state, :])
        
        thresh = np.random.rand()
        if(thresh < self.eps): # Explore
            return self.env.action_space.sample()
        else: # Exploit
            greedy_actions = np.where(self.Q[self.env.state, :]==np.amax(self.Q[self.env.state, :]))
            return np.random.choice(greedy_actions[0])
        
    # Function for SARSA policy updates
#     def updatePolicySARSA(self, S, A, R, S2, A2, alpha):
#         self.updates = self.updates + 1
#         self.Q[S,A] = self.Q[S,A] + alpha * (R + self.gamma * self.Q[S2, A2] - self.Q[S,A])
    
    # Function for Qlearning policy updates
    def update(self, S, A, R, S2, lr=0.1):
        self.updates = self.updates + 1
        self.Q[S,A] = self.Q[S,A] + lr * (R + self.gamma * np.max(self.Q[S2, :]) - self.Q[S,A])
        
    def train(self, no_episodes=200, horizon=1000, lr=0.1, track=True):
        self.reset()
        R_vec = []
        ts = 0
        
        for i in tqdm(range(no_episodes), leave=True):
            state = self.env.reset()
            for t in range(horizon):
                action = self.select_action()
                state2, r, done, _ = self.env.step(action)
                ts += 1
                self.update(state, action, r, state2)
                if(track):
                    R_vec.append(r)
                if(done):
                    break
                state = state2

        return R_vec, self.Q, ts
    
    def train_multiple_ts(self, runs=5, max_ts=20000, horizon=1000, lr=0.1, track=True):
        """
        This function trains the agent on the environment multiple times. The stopping
        criterion is number of timesteps.
        """
        R_mat = []; Q_vec=[]
        for j in range(runs):
            R_vec = []
            self.reset()
            ts = 0
            state = self.env.reset()
            for i in tqdm(range(max_ts), leave=True):
                action = self.select_action()
                state2, r, done, _ = self.env.step(action)
                self.update(state, action, r, state2)
                if(track):
                    R_vec.append(r)

                if(done):
                    state = self.env.reset()
                    continue

                state = state2

            R_mat.append(R_vec)
            Q_vec.append(self.Q)

        return R_mat, Q_vec