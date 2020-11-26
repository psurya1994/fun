from deep_rl_old import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
import random
import numpy as np

from deep_rl_old.network import *
from deep_rl_old.component import *
from deep_rl_old.utils import *

import sys
sys.path.append('../')

from deep_rl_old.component.fourrooms import FourRooms
env = FourRooms(goal=9)

### FUNCTIONS FOR PLOTTING LOSS FUNCTIONS

def convolve(points, kernelSize=5):
    array = np.convolve(points, np.ones(kernelSize)/kernelSize, 'valid')
    return array

def plot_dqn(obj, title = 'DQN'):
    """plot loss and returns"""
    plt.figure(figsize=(12,12),dpi=100)
    plt.subplot(311)
    plt.plot(convolve(obj.loss_vec, kernelSize=11)/11)
    plt.title(title+' loss ')
    plt.ylabel('loss value'), plt.xlabel('batch no')
    plt.subplot(312)
    plt.plot(convolve(obj.loss_vec, kernelSize=111)/111)
    plt.title(title+' smoothed loss ')
    plt.ylabel('loss value'), plt.xlabel('batch no')
    plt.subplot(313)
    plt.plot(np.array(obj.returns)[:,0],np.array(obj.returns)[:,1])
    plt.xlabel('timesteps'), plt.ylabel('return')
    plt.title(title+' training performance')
    plt.show()

def plot_dsr(obj, title='DSR'):
    """Visualize q and psi loss"""
    plt.figure(figsize=(12,12),dpi=100)
    plt.subplot(211)
    plt.plot(convolve(obj.loss_q_vec, kernelSize=111))
    plt.title(title+' loss')
    plt.ylabel('loss q value'), plt.xlabel('batch no')
    plt.subplot(212)
    plt.plot(convolve(obj.loss_psi_vec, kernelSize=111))
    plt.title(title+' loss')
    plt.ylabel('loss psi value'), plt.xlabel('batch no')
    
def plot_dsr2(obj, title='DSR'):
    """Visualize just the psi loss"""
    plt.figure(figsize=(12,4),dpi=100)
    plt.plot(convolve(obj.loss_psi_vec, kernelSize=11)/11)
    plt.title(title+' loss')
    plt.ylabel('loss psi value'), plt.xlabel('batch no')
    
def params_count(model):
    """Returns number of parameters in a model.
    Example: print(params_count(dsr.network))
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

### SANITY CHECKS for 4rooms



def visualize_vector(vector, title="unnamed", show_goal=False, show=True, label=None):
    """Function to visualize vector of size 104"""
    
    current_grid = np.array(env.occupancy, dtype=np.float)
    for i in range(104):
        p_cell = env.tocell[i]
        current_grid[p_cell[0], p_cell[1]] = vector[i]

        if show_goal:
            goal_cell = env.tocell[env.goal]
            current_grid[goal_cell[0], goal_cell[1]] = -1
        
    plt.imshow(current_grid), plt.colorbar()
    plt.title(title)
    if show:
        plt.show()
        
def sanity(agent, is_dsr=True, title="unnamed"):
    
    print(agent.network)
    plt.figure(figsize=(10,10))
    
    # Setting things up
    env = FourRooms(goal=9)
    
    # Visualize w
    if(is_dsr):
        plt.subplot(221)
        w_vector = agent.network.state_dict()['psi2q.w'].numpy()
        visualize_vector(w_vector, title=title+" w learnt values", show=False)
    
    # Visualize psi
    if(is_dsr):
        plt.subplot(222)
        states = [100]
#         plt.figure(figsize=(len(states)*12, 5))

        for i, s in enumerate(states):
            temp = np.zeros(104)
            temp[s] = 1
            phi, psi, q = agent.network(tensor(temp).unsqueeze(0))
            psi = psi.detach().numpy()
            optimal_actions = q.max(1)[1].detach().numpy()
            psi = psi[:, optimal_actions, :]
            q = q.detach().numpy().flatten()
#             visualize_vector(psi[:,0,:].flatten(), title=title+" q({},{})={}".format(s, 0, q[0]), show=False, label="0")
            visualize_vector(psi.flatten(), title=title+" psi*, s = {}".format(s), show=False, label="1")
            
    # Visualize policy
    state_space = np.identity(104)
    if(is_dsr):
        
        phi, psi, q = agent.network(tensor(state_space))
    else:
        q = agent.network(tensor(state_space))
    
    plt.subplot(223)
    optimal_actions = q.max(1)[1].detach().numpy()
    visualize_vector(optimal_actions, title=title+'policy (0:up,1:down,2:left,3:right)', show=False)
    plt.subplot(224)
    visualize_vector(q.max(1)[0].detach().numpy(), title=title+' Q* estimates')
    
    return
  
def sanity_psi(dsr, action_net=None, states=[0, 100, 10, 97], title="unnamed"):
    """
    action_net needs to be a DQN or None for this funciton to work.
    """
    plt.figure(figsize=(10,10))
    state_space = np.identity(104)
    phi, psi, q = dsr.network(tensor(state_space))
    psi = psi.detach().numpy()
    if(action_net is not None):
        q = action_net.network(tensor(state_space))
    optimal_actions = q.max(1)[1].detach().numpy()
    
    psi = psi[:, optimal_actions, :]

    for i, s in enumerate(states):
        plt.subplot(2,2,i+1)
        visualize_vector(psi[s,:,:,].flatten(), title=title+" psi*, s = {}".format(s), show=False, label="1")
    return