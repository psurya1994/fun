import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Owl(gym.Env):
    """
    Description: 
        The agent is at the center of the circle. It hears sounds
        through both it's ears and it sees objects in front. The agent
        needs to turn it's head towards moving preys.
    
    """
    
    def __init__(self, sight_range=np.pi*2, focus_range=np.pi/3, n=60):
        self.sight_range = sight_range
        self.focus_range = focus_range
        self.n = n
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.zeros(self.n), high=np.ones(self.n), dtype=np.uint8)
        
        self.view_angle = 0
        self.rot_speed = np.pi / 10
        self.prey = True
        self.action_space = spaces.Discrete(3)
        self.horizon = 100
        self.steps = 0
        # Random number generator
        self.rng = np.random.RandomState(1234)

    def find_state_reward(self):

        # Finding the state
        prey_angle = np.arctan2(self.prey_location[1], self.prey_location[0])
        error = prey_angle - self.view_angle
        if((error > np.pi) or (error < -np.pi)):
            error = 2*np.pi - error
        eyes = np.zeros(self.n)
        if(np.abs(error) < self.sight_range):
            mask_loc = self.n - self.n / 2 * \
                        (1 + (error) * 2 / self.sight_range)
            eyes[int(mask_loc)] = 1
#         if(error < np.pi/6):
#             mask_loc = 60 - (30 + (prey_angle - self.view_angle) * 30 * 6 / np.pi)
#             eyes[int(mask_loc)] = 1

        if(error < self.focus_range / 2):
            R = 0
        else:
            R = -1



        return eyes, R

        
    def reset(self):
        self.preys()
        self.steps = 0
        self.view_angle = 0
        
        state, R = self.find_state_reward()

        return state

    def seed(self, seed=None):
        """
        Setting the seed of the agent for replication
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        done = False
        
        # Update prey location
        self.prey_location += self.prey_speed * self.prey_dirn_vec
            
        # Update animal view
        self.view_angle += (action-1) * self.rot_speed
        if(self.view_angle < -np.pi):
            self.view_angle += 2 * np.pi
        elif(self.view_angle > np.pi):
            self.view_angle -= 2 * np.pi

        # Calculate observation and reward
        state, R = self.find_state_reward()

        # Calculate done
        self.steps += 1
        if(self.steps>=self.horizon or np.linalg.norm(self.prey_location) > 1):
            done = True
            
        return state, R, done, {}
            
        
    def preys(self):
        
        self.prey_speed = 0.1
        # self.prey_location = np.random.rand(2)-0.5
        self.prey_location = np.array([1, 0])
        self.prey_location = self.prey_location / np.linalg.norm(self.prey_location)
        prey_angle = np.arctan2(self.prey_location[1],self.prey_location[0])
        self.prey_direction = prey_angle + np.pi + np.pi/2 * (np.random.rand() - 0.5)
        self.prey_dirn_vec = np.array([np.cos(self.prey_direction), np.sin(self.prey_direction)])
        self.prey_size = 0.1
    
    def render(self, mode='human'):
        fig, ax = plt.subplots(dpi=100)
        
        # plot boundary
        rads = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(rads), np.sin(rads))
        ax.set_aspect('equal', 'box')
        
        # plot view
        ax.add_artist(mpatches.Arc((0,0),0.7,0.7,theta1=(self.view_angle-self.sight_range/2)*180/np.pi, \
                                   theta2=self.view_angle*180/np.pi, ec='y'))
        ax.add_artist(mpatches.Arc((0,0),0.7,0.7,theta1=self.view_angle*180/np.pi, \
                                   theta2=(self.view_angle+self.sight_range/2)*180/np.pi, ec='g'))
        
        # plot prey
        if(self.prey):
            ax.scatter(self.prey_location[0], self.prey_location[1])
            ax.plot([self.prey_location[0], self.prey_location[0] + 3 * self.prey_speed * self.prey_dirn_vec[0]], \
                    [self.prey_location[1], self.prey_location[1] + 3 * self.prey_speed * self.prey_dirn_vec[1]], \
                   'k--')
            
        # plot animal
        ax.add_artist(plt.Circle((0,0),0.1))
        left_angle = self.view_angle - np.pi/6
        right_angle = self.view_angle + np.pi/6
        ax.plot([0,np.cos(left_angle)], [0, np.sin(left_angle)], 'k--')
        ax.plot([0,np.cos(right_angle)], [0, np.sin(right_angle)], 'k--')
        
        plt.show()
        
class OwlTabular(Owl):
    def __init__(self):
        Owl.__init__(self)
        self.observation_space = spaces.Discrete(61)

    def reset(self):
        obs = Owl.reset(self)
        if(np.sum(obs) == 0):
            return 0
        else:
            return 1 + np.where(obs==1)[0][0]

    def step(self, action):
        obs, R, done, info = Owl.step(self, action)
        if(np.sum(obs) == 0):
            return 0, R, done, info
        else:
            return 1 + np.where(obs==1)[0][0], R, done, info 
