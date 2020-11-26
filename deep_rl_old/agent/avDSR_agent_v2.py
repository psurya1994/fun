import random
"""
Code for learning the averageSR agent across good policies. This version has the state reconstruction part too.
"""
from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class torch_reshape(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 64, 9, 9)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class Pick():
    def __init__(self):
        self.network = 0
        
class avDSRActor_v2(BaseActor):
    def __init__(self, config, agents, style='DQN', choice=1):
        """
        style -> depicts the network config of the agent used for exploration.
        choice -> tells how we choose which agent to use for exploration
            0 - at every timestep, we randomly pick and agent and take an eps greedy action
            1 - we choose a new DQN every switch_period
        """
        BaseActor.__init__(self, config)
        self.config = config
        self.agents = agents
        self.style = style
        self.choice = choice

        # Parameters to decide which agents should learn
        self.batch_steps = 0
        self.switch_period = 10
        self.agent_id = 0

        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        
        # Choosing which agent for taking actions

        if(len(self.agents) == 0):
            pick = Pick()
            pick.network = lambda a: torch.zeros(self.config.action_dim,1)
        elif(self.choice == 0):                
            pick = random.choice(self.agents)
        elif(self.choice == 1):
            self.batch_steps += 1
            if(self.batch_steps % self.switch_period == 0): 
                # CHECK: multiprocessing might be screwing something up
                self.agent_id = np.random.randint(len(self.agents))
            pick = self.agents[self.agent_id]
        else:
            raise NameError('Invalid choice config')

        # Find qvalues of the picked agent for the present state
        with config.lock:
            if(self.style == 'DSR'):
                _, _, q_values = pick.network(config.state_normalizer(self._state))
            elif(self.style == 'DQN'):
                q_values = pick.network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()

        # Take action based on this estimated q value
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
            
        next_state, reward, done, info = self._task.step([action])
        
        # Also estimate next action
        #############
        if(len(self.agents) == 0):
            pick2 = Pick()
            pick2.network = lambda a : torch.zeros(self.config.action_dim,1)
        elif(self.choice == 0):
            pick2 = random.choice(self.agents)
        elif(self.choice == 1):
            pick2 = pick

        with config.lock:
            if(self.style == 'DSR'):
                _, _, q_values = pick2.network(config.state_normalizer(next_state))
            elif(self.style=='DQN'):
                q_values = pick2.network(config.state_normalizer(next_state))
        q_values = to_np(q_values).flatten()

        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            next_action = np.random.randint(0, len(q_values))
        else:
            next_action = np.argmax(q_values)
        
        entry = [self._state[0], action, reward[0], next_state[0], next_action, int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry

class avDSRAgent_v2(BaseAgent):
    def __init__(self, config, agents, style='DQN'):
        """
        agents -> list of agents whose actions we need to consider.
        """
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()
        
        self.loss_rec_vec = []
        self.loss_psi_vec = []
        self.loss_vec = []

        self.replay = config.replay_fn()
        self.choice = config.choice
        self.actor = avDSRActor_v2(config, agents, style, self.choice)

        self.network = config.network_fn()
        self.network.share_memory()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size) # Need to make this size bigger

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        _, _, q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        # Store transitions in the buffer
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, next_action, done, info in transitions:
#             self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, next_action, done])
        self.replay.feed_batch(experiences)

        # Start updating network parameters after exploration_steps
        if self.total_steps > self.config.exploration_steps:

            # Getting samples from buffer
            experiences = self.replay.sample()
            states, actions, rewards, next_states, next_actions, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)

            # Estimate targets
            with torch.no_grad():
                _, psi_next, _, _ = self.network(next_states)

            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                next_actions = tensor(next_actions).long()
                psi_next = psi_next[self.batch_indices, next_actions, :] # TODO: double check dims here

            terminals = tensor(terminals)
            psi_next = self.config.discount * psi_next * (1 - terminals.unsqueeze(1).repeat(1, psi_next.shape[1]))
            phi, psi, state_rec, _ = self.network(states)
            psi_next.add_(phi) # TODO: double chec this

            # Computing estimates
            actions = tensor(actions).long()
            psi = psi[self.batch_indices, actions, :]
            loss_psi = (psi_next - psi).pow(2).mul(0.5).mean()

            # Estimating reconstuction loss
            loss_rec = (state_rec - tensor(states)).pow(2).mul(0.5).mean()
            loss = loss_psi + config.c * loss_rec
            
            total_loss = loss.mean()
            self.loss_vec.append(total_loss.item())
            self.loss_psi_vec.append(loss_psi.item())
            self.loss_rec_vec.append(loss_rec.item())
            
            
            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

            with config.lock:
                self.optimizer.step()