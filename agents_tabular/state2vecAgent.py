"""
Code that implements the testing phase of the paper:
https://arxiv.org/abs/1910.10277

Need to set terminal states to 0 before we start learning.

"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from random import sample
import random
from agents.agent import Agent

class State2vecAgent(Agent):
	"""
	Q(s,a) = M . theta
	M is learnt in the training phase and theta is learnt during the
	testing phase. 

	All the implementations here are done using fitted q iteration 
	style.
	"""
	
	def __init__(self, env, Mrand, eps=0.1, gamma=0.95, w=None):
		"""
		Constructor function.

		Inputs: 
			env -> Environment on which the agent learns. Used to initialize
			the size of w
			Mrand -> Expected SR fosr random policy, encode the structure 
			of space. This is learnt during the training phase which has
			not been implemented here. Needs to be of size s,a,s
			eps -> exploration parameter
			gamma -> discount factor
			w -> weights that need to be learnt
		"""
		self.observation_space = env.observation_space.n
		self.action_space = env.action_space.n
		self.M = Mrand
		self.eps = eps
		self.gamma = gamma
		self.errors = []

		# Initializing w in [V = psi * w] with zeros
		if(w is None):
			self.w = np.zeros(self.observation_space)
		else:
			self.w = w

	def q_value(self):
		"""Function to estimate Q value for agent
		"""
		return self.M @ self.w

	def reset(self, w=None):
		if(w is None):
			self.w = np.zeros(self.observation_space)
		else:
			self.w = w
		return

	def update(self, s, a, r, s2, a2, lr=0.1):
		"""
		SARSA style update.
		"""
		error = (r + self.gamma * (self.M[s2, a2, :] @ self.w) - (self.M[s, a, :] @ self.w))
		self.w += lr * error * self.M[s, a, :]
		self.errors.append(error)

	def plot_errors(self):
		plt.plot(self.errors)
		plt.xlabel('timesteps'), plt.ylabel('errors')
		plt.title('Error plot for state2vec')
		plt.show()