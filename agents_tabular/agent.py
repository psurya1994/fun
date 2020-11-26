"""
Base class for all agents.
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from random import sample
import random
from tqdm import tqdm

class Agent:
	def __init__(self):
		raise NotImplementedError

	def select_action(self, state, greedy=False):
		"""TODO: implement this standard function for all classes
		"""
		thresh = np.random.rand()
		if(greedy or thresh > self.eps):
			greedy_actions = np.where(self.q_value()[state]==np.amax(self.q_value()[state]))
			return np.random.choice(greedy_actions[0])
		else:
			return sample(range(0, self.action_space), 1)[0]

	def visualize_q_gridworld(self, env):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	def update(self, s, a, r, s2, a2, lr=0.1):
		raise NotImplementedError

	def q_value(self):
		raise NotImplementedError

	def train_one_eps(self, env, horizon=1000, lr=0.1):
		"""
		Function to train the agent on one episode.

		v1: updated to return total reward in the episode instead
		of vector of rewards at each time step. check github before
		Feb 19th to revert to older version.

		Inputs: 
			env -> openAI environment object of the environment
			horizon -> max timesteps before episode ends
			lr -> learning rate for updates (TODO: include this in self.lr)

		Outputs: 
			r_ep -> returns the reward in the episode
			(obselete) r_vec -> vector of rewards obtained the environments
			ep_len -> length of episode
			self -> state of the object updated internally while learning
		"""
		r_ep = 0 # variable to track episode 
		ep_len = 0

		state = env.reset()
		action = self.select_action(state)

		for t in range(horizon):
			
			state2, r, done, _ = env.step(action)
			ep_len = ep_len + 1
			r_ep += r

			action2 = self.select_action(state2)
			self.update(state, action, r, state2, action2, lr=lr)
			action = action2
			
			if(done):
				break
			state = state2

		return r_ep, ep_len


	def train_multiple_eps(self, env, no_episodes=200, horizon=1000, lr=0.1):
		"""
		Trains the agent for multiple episodes.

		Returns:
			r_vec -> vector of rewards obtained at each timestep
			ep_len_vec -> vector of episode lengths
		"""

		r_vec = []
		ep_len_vec = []
		for i in range(no_episodes):

			# Run the agent for one episode and get a vector of rewards and a scalar for episode length
			r_ep, ep_len = self.train_one_eps(env, horizon=horizon, lr=lr)

			# Storing the information
			r_vec.append(r_ep)
			ep_len_vec.append(ep_len)

		return r_vec, ep_len_vec

	def train_multiple_eps_dynamic(self, env, no_episodes=200, ng_int=50, horizon=1000, lr=0.1):
		"""
		Trains the agent for multiple episodes.
		ng_int -> new goal interval

		Returns:
			r_vec -> vector of rewards obtained at each timestep
			ep_len_vec -> vector of episode lengths
		"""

		r_vec = []
		ep_len_vec = []
		tasks = 0
		for i in range(no_episodes):
			if(i % ng_int == 0):
				goal_index = [0, 12, 156, 168]
				new_goal = [goal_index[tasks]]
				tasks += 1
				env.reset(loc_r=new_goal, loc_t=new_goal)
				# new_goal = sample(env.listGoalStates(), 1)
				# env.reset(loc_r=new_goal, loc_t=new_goal)

			# Run the agent for one episode and get a vector of rewards and a scalar for episode length
			r_ep, ep_len = self.train_one_eps(env, horizon=horizon, lr=lr)

			# Storing the information
			r_vec.append(r_ep)
			ep_len_vec.append(ep_len)

		return r_vec, ep_len_vec

	def train_multiple_runs_eps_dynamic(self, env, runs=3, no_episodes=200, ng_int=50, horizon=1000, lr=0.1):
		"""
		Trains the agent for multiple runs, with each run having multiple episodes.
		The number of episodes are decided by maximum timesteps specified in `ts_max`.
		
		Input:
			ng_int -> new goal interval

		Returns:
			r_mat -> matrix of rewards collected with row representing run number
				and column representing the rewards obtained at a specific timestep
				in that run.
			info -> additional details about the matrix such as mean and standard 
				deviation.
		"""
		r_mat = []
		info = {}
		
		for i in tqdm(range(runs)):

			# Resetting agent to default before each run
			if(hasattr(self, 'reset_task')):
				self.reset_task()
			else:
				self.reset()

			# Training the agent for ts_max
			r_vec, _ = self.train_multiple_eps_dynamic(env, no_episodes=no_episodes, ng_int=ng_int, horizon=horizon, lr=lr)

			# Storing the results in a matrix
			r_mat.append(r_vec)

		# Finding the mean and standard deviation 
		info['mean'] = np.mean(np.array(r_mat), axis=0)
		info['std'] = np.std(np.array(r_mat), axis=0)

		return r_mat, info

	def train_multiple_runs_eps(self, env, runs=3, no_episodes=200, horizon=1000, lr=0.1):
		"""
		Trains the agent for multiple runs, with each run having multiple episodes.
		The number of episodes are decided by maximum timesteps specified in `ts_max`.

		Returns:
			r_mat -> matrix of rewards collected with row representing run number
				and column representing the rewards obtained at a specific timestep
				in that run.
			info -> additional details about the matrix such as mean and standard 
				deviation.
		"""
		r_mat = []
		info = {}
		
		for i in range(runs):

			# Resetting agent to default before each run
			self.reset()

			# Training the agent for ts_max
			r_vec, _ = self.train_multiple_eps(env, no_episodes=no_episodes, horizon=horizon, lr=lr)

			# Storing the results in a matrix
			r_mat.append(r_vec)

		# Finding the mean and standard deviation 
		info['mean'] = np.mean(np.array(r_mat), axis=0)
		info['std'] = np.std(np.array(r_mat), axis=0)

		return r_mat, info

	def train_multiple_ts(self, env, ts_max=10000, horizon=1000, lr=0.1):
		"""
		Same function as above, running for a fixed number of timesteps. The ending is sepcified
		by timesteps instead of episodes.

		Returns:
			r_vec -> vector of rewards obtained at each timestep
			:rtype: list of int
		"""

		r_vec = []
		state = env.reset()
		action = self.select_action(state)

		# Loop to run for each timestep
		for i in range(ts_max):

			state2, r, done, _ = env.step(action)
			r_vec.append(r)

			action2 = self.select_action(state2)
			self.update(state, action, r, state2, action2, lr=lr)
			action = action2
			state = state2
			
			if(done):
				state = env.reset()

		return r_vec

	def train_multiple_ts_dynamic(self, env, no_tasks=20, ts_task=1000, lr=0.1):
		"""
		Function to train an agent for a give number of timesteps in a dynamically
		changing environment.

		Inputs:
			env -> needs to be dynamic. needs to implement .reset(newgoal)
			no_tasks -> number of tasks to be solved
			ts_task -> number of timesteps for each task
		
		Returns:
			r_vec -> vector of rewards obtained at each timestep
			:rtype: list of int
		"""
		r_vec = []

		for i in tqdm(range(no_tasks),leave=True):

			state = env.reset(newgoal=True)
			action = self.select_action(state)

			# Determines the config of the agent at the start of new task
			self.reset()

			r_ep = 0 # variable to track total reward in episode

			for j in range(ts_task):

				state2, r, done, _ = env.step(action)
				r_ep += r
				action2 = self.select_action(state2)
				self.update(state, action, r, state2, action2, lr=lr)

				action = action2
				state = state2
				
				if(done):
					r_vec.append(r_ep)
					r_ep = 0
					state = env.reset()

		return r_vec

	def train_multiple_runs_dynamic(self, env, runs=3, no_tasks=20, ts_task=1000, lr=0.1):
		r_mat = []
		info = {}
		
		for i in range(runs):

			# Resetting agent to default before each run
			self.reset()

			# Training the agent for ts_max
			r_vec = self.train_multiple_ts_dynamic(env, no_tasks=20, ts_task=1000, lr=0.1)

			# Storing the results in a matrix
			r_mat.append(r_vec)

		# Finding the mean and standard deviation 
		info['mean'] = np.mean(np.array(r_mat), axis=0)
		info['std'] = np.std(np.array(r_mat), axis=0)

		return r_mat, info

		return

	def train_multiple_runs(self, env, runs=3, ts_max=10000, horizon=1000, lr=0.1):
		"""
		Trains the agent for multiple runs, with each run having multiple episodes.
		The number of episodes are decided by maximum timesteps specified in `ts_max`.

		Returns:
			r_mat -> matrix of rewards collected with row representing run number
				and column representing the rewards obtained at a specific timestep
				in that run.
			info -> additional details about the matrix such as mean and standard 
				deviation.
		"""
		r_mat = []
		info = {}
		
		for i in range(runs):

			# Resetting agent to default before each run
			self.reset()

			# Training the agent for ts_max
			r_vec = self.train_multiple_ts(env, ts_max=ts_max, horizon=horizon, lr=lr)

			# Storing the results in a matrix
			r_mat.append(r_vec)

		# Finding the mean and standard deviation 
		info['mean'] = np.mean(np.array(r_mat), axis=0)
		info['std'] = np.std(np.array(r_mat), axis=0)

		return r_mat, info

	def visualize_M_gridworld(self, state=0):
		"""
		Function to visualize the SR in gridworld.

		state -> specify the index of state you want to visualize here
		"""

		plt.subplot(221); plt.imshow(self.M[12,0,:].reshape(5,5)), plt.colorbar()
		plt.subplot(222); plt.imshow(self.M[12,1,:].reshape(5,5)), plt.colorbar()
		plt.subplot(223); plt.imshow(self.M[12,2,:].reshape(5,5)), plt.colorbar()
		plt.subplot(224); plt.imshow(self.M[12,3,:].reshape(5,5)), plt.colorbar()
		plt.show()

class RandomSRAgent(Agent):
	"""Class for tabular SR agent for control
	"""
	def __init__(self, env, gamma=0.95):
		self.no_updates = 0
		self.observation_space = env.observation_space.n
		self.action_space = env.action_space.n
		self.M = np.zeros((self.observation_space, self.action_space, self.observation_space))
		self.gamma = gamma
		# Setting the SR values of terminal states
		for i in range(self.observation_space):
			self.M[i, :, i] = np.ones(self.action_space) # TODO: do this for all states

	def select_action(self, state):
		"""Agent always takes random actions
		"""
		return sample(range(0, self.action_space), 1)[0]

	def update(self, s, a, r, s2, a2, lr=0.1):
		self.M[s, a, :] = (1 - lr) * self.M[s, a, :] + lr * \
						((np.arange(self.observation_space)==s).astype(dtype='int') + \
							self.gamma * self.M[s2, a2, :])