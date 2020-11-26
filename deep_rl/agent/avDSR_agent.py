#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import wandb

class avDSRActorRandom(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()

        config = self.config # DELETE this line

        # Predict action
        action = np.random.randint(self.config.action_dim)

        # Take action
        next_state, reward, done, info = self._task.step(action)

        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry


class avDSRAgent(BaseAgent):
    def __init__(self, config, agents=None):
        """
            agents -> list of agents whose actions we need to sample.
        """
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        if(agents is None):
            self.actor = avDSRActorRandom(config)
            self.config.style = 'random'
        else:
            raise NotImplementedError

        self.network = config.network_fn()
        self.network.share_memory()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0

        try:
            self.is_wb = config.is_wb
        except:
            print('is_wb config not found, using deafult.')
            self.is_wb = True

        if(self.is_wb):
            wandb.init(entity="psurya", project="sample-project")
            wandb.watch_called = False

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        actions = tensor(transitions.action).long()
        with torch.no_grad():
            psi_next = self.target_network(next_states)['psi'].detach()
            # q_next = q_next.max(1)[0]
            if(self.config.style == 'random'):
                next_actions = tensor(np.random.randint(self.config.action_dim)).long()
            else:
                raise NotImplementedError
            psi_next = psi_next[self.batch_indices, next_actions, :] # FIX BATCH INDICES
            psi_next = self.config.discount * psi_next * (1 - masks.unsqueeze(1).repeat(1, psi_next.shape[1]))

        

        out = self.network(states)
        phi, psi, state_rec = out['phi'], out['psi'], out['state_rec']
        psi = psi[self.batch_indices, actions, :]
        psi_next.add_(phi)

        loss_rec = (state_rec - tensor(states)).pow(2).mul(0.5).mean()
        loss_psi = (psi_next - psi).pow(2).mul(0.5).mean()


        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q

        return loss

    def step(self):
        config = self.config
        # Step and get new transisions
        transitions = self.actor.step()

        for states, actions, rewards, next_states, dones, info in transitions:

            # Recording results
            self.record_online_return(info) # to log, screen
            for i, info_ in enumerate(info): # to wandb
                ret = info_['episodic_return']
                if ret is not None:
                    if(self.is_wb):
                        wandb.log({"steps_ret": self.total_steps, "returns": ret})

            self.total_steps += 1

            # Feed the transisions into replay
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            loss = self.compute_loss(transitions)

            loss = self.reduce_loss(loss)
            if(self.is_wb):
                wandb.log({"steps_loss": self.total_steps, "loss": loss.item()})

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()
