"""
Code implementing a tabular SR agent for control
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from PIL import Image, ImageDraw, ImageFont
from random import sample
import random
from agents.agent import Agent

class TabularSRAgent(Agent):
    """Class for tabular SR agent for control
    """
    def __init__(self, env, eps=None, gamma=0.95):

        # Initialization of params
        self.env = env
        self.observation_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.M = np.zeros((self.observation_space, self.action_space, self.observation_space))
        self.R = np.zeros(self.observation_space)
        self.updates = 0
        self.gamma = gamma
        
        # Setting the SR values of terminal states
        for i in range(self.observation_space):
            self.M[i, :, i] = np.ones(self.action_space) # TODO: do this for all states
        
        # Exploration parameters
        self.eps_decay = 5e4
        self.eps_start = 1
        self.eps_end = 0.05
        if(eps is not None):
            self.eps_start = eps
            self.eps_end = eps

        self.eps = self.eps_start
        pass
    
    def update_eps_linear(self):
        """
        Updating exploration rate
        """
        if(self.updates > self.eps_decay):
            return
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * self.updates / self.eps_decay

    def q_value(self):
        """Function that estimates qvalues at a given state.
        """
        return self.M @ self.R

    
    # def select_action(self, state, greedy=False):
    #     """
    #     Function to choose the action for agent
    #     """
    #     thresh = np.random.rand()
    #     if(greedy):
    #         self.Q = self.M[state, :, :] @ self.R
    #         return np.argmax(self.Q)
    #     if(not greedy):
    #         self.update_eps_linear()
            
    #     if(greedy or thresh > self.eps):
    #         self.Q = self.M[state, :, :] @ self.R # CHECK
    #         greedy_actions = np.where(self.Q==np.amax(self.Q))
    #         return np.random.choice(greedy_actions[0])
    #     else:
    #         return sample(range(0, self.action_space), 1)[0]
    
    def update(self, s, a, r, s2, a2, lr=0.1):
        """
        Function to update the SR and reward for the agent.
        Assumes deterministic state dependant reward.
        Updates the SR with a TD style update
        """
#         import pdb; pdb.set_trace();
        self.M[s, a, :] = (1 - lr) * self.M[s, a, :] + lr * \
                        ((np.arange(self.env.observation_space.n)==s).astype(dtype='int') + self.gamma * self.M[s2, a2, :])
#         Qtmp = self.M[self.env.state, :, :] @ self.R
#         amax = np.where(Qtmp==np.amax(Qtmp))
#         self.M[s, a, :] = (1 - lr) * self.M[s, a, :] + lr * \
#                         ((np.arange(self.env.observation_space.n)==s).astype(dtype='int') \
#                          + self.gamma * np.mean(self.M[s2, amax[0], :], axis=0))
        self.R[s2] = r # reward when you enter the state
        self.updates = self.updates + 1

#     def predict(self, env, no_episodes=200, horizon=1000, lr=0.1, track=True, max_ts=None):
#         """
#         Function for prediction of M for random policy in the given environment.
#         """
#         state = self.env.reset()
#         R_vec = []
#         ts = 0
# #         for i in tqdm(range(no_episodes), leave=True):
#         for i in range(no_episodes):
#             state = env.reset()
#             action = env.action_space.sample()
#             for t in range(horizon):
#                 state2, r, done, _ = env.step(action)
#                 action2 = env.action_space.sample()
#                 ts += 1
# #                 import pdb; pdb.set_trace()
#                 self.update(state, action, r, state2, action2)
#                 if(track):
#                     R_vec.append(r)
#                 if(done):
#                     break
#                 state = state2
#                 action = action2
#                 if((max_ts is not None) and (ts >= max_ts)):
#                     return R_vec, self.M

#         return R_vec, self.M

#     def train(self, env, no_episodes=200, horizon=1000, lr=0.1, track=True, M=None, R=None):

#         self.reset(M=M, R=R)
#         R_vec = []
#         ts = 0
# #         for i in tqdm(range(no_episodes), leave=True):
#         for i in range(no_episodes):
#             state = env.reset()
#             action = self.select_action(state)
#             for t in range(horizon):
#                 state2, r, done, _ = env.step(action)
#                 action2 = self.select_action(state2)
#                 ts += 1
#                 if(t >= 1):
#                     self.update(state, action, r, state2, action2, lr=lr)
#                 action = action2
#                 if(track):
#                     R_vec.append(r)
#                 if(done):
#                     break
#                 state = state2

#         return R_vec, self.M, ts

    # def train_multiple_ts(self, env, runs=5, M=None, R=None, max_ts=20000, horizon=1000, lr=0.1, track=True, changeG=True, goals=None, config=1):
    #     """
    #     This function trains the agent on the environment multiple times. The stopping
    #     criterion is number of timesteps.
    #     goals -> list of goal states
    #     changeG -> boolean that indicates if golas should changeo ver time
    #     config -> 1 (everything else), 2 (don't reset the agent), 3 (reset only r)
    #     """

    #     R_mat = []; M_vec=[]; ts_vec=[]; gl=None
    #     if(goals == None):
    #         goals = env.listGoalStates()

    #     for j in range (runs):
    #         R_vec = []
    #         self.reset(M=M, R=R)
    #         ts = 0
    #         state = env.reset()
    #         action = self.select_action(state)
    #         for i in tqdm(range(max_ts), leave=True):
                
    #             state2, r, done, _ = env.step(action)
    #             action2 = self.select_action(state2)
    #             ts += 1
    #             self.update(state, action, r, state2, action2)
    #             if(track):
    #                 R_vec.append(r)
    #             if(changeG==True and (ts%500)==0):
    #                 random.seed(i)
    #                 gl = sample(goals,1)
    #                 if(config==1):
    #                     self.reset(M=M, R=R)
    #                 elif(config==3):
    #                     self.R = np.zeros(self.observation_space)
    #             elif(changeG==False):
    #                 gl=None
                    
    #             if(done):
    #                 state = env.reset(loc_r=gl, loc_t=gl)
    #                 continue

    #             state = state2
    #             action = action2

    #         R_mat.append(R_vec)
    #         M_vec.append(self.M)
    #         ts_vec.append(ts)

    #     return R_mat, M_vec, ts_vec

    def reset(self, M=None, R=None):
        
        if(M is None):
            self.M = np.zeros((self.observation_space, self.action_space, self.observation_space))
            # Setting the SR values of terminal states
            for i in range(self.observation_space):
                self.M[i, :, i] = np.ones(self.action_space)
        else:
            self.M = M.copy()
        if(R is None):
            self.R = np.zeros(self.observation_space)
        else:
            self.R = R.copy()
        self.updates = 0
        return

            
    def visualize_Q(self, env):
        self.Q = self.M @ self.R
        plt.subplot(2,2,1)
        plt.imshow(np.flip(self.Q[:,0].reshape((env.n, env.n)), axis=0))
        plt.subplot(2,2,2)
        plt.imshow(np.flip(self.Q[:,1].reshape((env.n, env.n)), axis=0))
        plt.subplot(2,2,3)
        plt.imshow(np.flip(self.Q[:,2].reshape((env.n, env.n)), axis=0))
        plt.subplot(2,2,4)
        plt.imshow(np.flip(self.Q[:,3].reshape((env.n, env.n)), axis=0))
        plt.show()
        
    def ind2xy(self, ind):
        return self.env.n - 1 - ind // self.env.n, ind % self.env.n
    
    def dir2coords(self, val, x, y, cs):
        """Converts policy direction to coordinates
        for plotting a triangle.
        
        """
        if(val == 0):
            return [x, y, x, y-cs*0.45]
        elif(val == 1):
            return [x, y, x+cs*0.45, y]
        elif(val == 2):
            return [x, y, x, y+cs*0.45]
        elif(val == 3):
            return [x, y, x-cs*0.45, y]
        
    def visualize_policy(self, mode='visual'):
        h=self.env.n*self.env.cs
        w=self.env.n*self.env.cs
        cs = self.env.cs
        
        self.Q = self.M @ self.R
        if(mode=='value'):
            best_action_value = np.max(self.M @ self.R, axis=1)
        if(mode=='text'):
            best_action_value = np.argmax(self.M @ self.R, axis=1)
        if(mode=='visual'):
            best_action_value = np.argmax(self.M @ self.R, axis=1)
        img = self.env.render(mode='rgb_array', printR=False)
        fnt = ImageFont.truetype("arial.ttf", h//20)
        draw = ImageDraw.Draw(img)
        
        
        for i in range(self.env.n * self.env.n):
            y, x = self.ind2xy(i)
            x = x * cs + cs//2
            y = y * cs + cs//2
            if(mode=='value' or mode=='text'):
                draw.text([x-cs/3, y-cs/3], "{:.2f}".format(best_action_value[i]), font=fnt, fill="blue")
            if(mode=='visual'):
                Qtmp = self.M[i,:,:] @ self.R
                actions = np.where(Qtmp==np.amax(Qtmp))
                for action in actions[0]:
                    draw.line(self.dir2coords(action, x, y, cs), fill="darkblue", width=3)
                
        plt.imshow(img)
        plt.grid(color='k', linestyle='-', linewidth=2)
        plt.show()
        
    def get_Mpi(self):
        """Returns the SR matrix for current policy
        """
        self.Mpi = np.zeros((self.env.n**2, self.env.n**2))
        for i in range(self.env.n**2):
            # calculating best actions
            Qtmp = self.M[i,:,:] @ self.env.R
            actions = np.where(Qtmp==np.amax(Qtmp))
            for j in range(self.env.n**2):
                tmp = 0
                for action in actions[0]:
                    tmp = tmp + self.M[i, action, j]/actions[0].shape[0]
                self.Mpi[i, j] = tmp
                
        return self.Mpi
    
    def visualize_M(self, s):
        h = self.env.n*self.env.cs
        w = self.env.n*self.env.cs
        cs = self.env.cs
        
        self.get_Mpi()
        
        img = self.env.render(mode='rgb_array', printR=False)
        fnt = ImageFont.truetype("arial.ttf", h//20)
        draw = ImageDraw.Draw(img)
        
        for i in range(self.env.n * self.env.n):
            y, x = self.ind2xy(i)
            x = x * cs + cs//2
            y = y * cs + cs//2
            draw.text([x-cs/3, y-cs/3], "{:.2f}".format(self.Mpi[s, i]), font=fnt, fill="blue")
                
        plt.imshow(img)
        plt.title('M(s,:) for s = ' + str(s))
        plt.show()
    
    def visualize_R(self):
        plt.imshow(np.flip(self.R.reshape((5,5)), axis=0))
        plt.colorbar()
        plt.show()
        
    def evaluate(self, env, no_seeds=10, horizon=100):
        for i in range(no_seeds):
            s = env.reset()
            env.seed(i)
            t_vec = []
            for t in range(horizon):
                s, r, done, _ = env.step(self.select_action(s, greedy=True))
                if(done):
                    t_vec.append(env.update_count)
                    
        return t_vec

    
class avSRAgent(TabularSRAgent):

    def __init__(self, env, eps=None, gamma=0.95):
        super().__init__(env, eps=eps, gamma=0.95)
        self.Mav = self.M
        # self.task_index = 0

    def reset(self):
        """Fully reset the agent
        """
        super().reset()
        self.Mav = self.M

    def reset_task(self, lr=0.5):
        """
        This resets the agent when a new task comes up.
        """
        # self.task_index += 1
        self.Mav += lr * (self.M - self.Mav)
        self.M = self.Mav
        # self.updates = 0
        return

class prevSRAgent(TabularSRAgent):
    """
    Agent that uses previous M and R for new tasks too.
    """
    def reset(self):
        """Fully reset the agent
        """
        super().reset()

    def reset_task(self):
        """
        This resets the agent when a new task comes up.
        """
        self.updates = 0
        pass