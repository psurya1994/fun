"""
Standard protocol with reward error.
"""

from deep_rl import *
import pickle
import uuid
import torch

# Class for avDSR actor
class avDSRActorRandom(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()

        config = self.config # DELETE this line

        # Predict action
        # action = np.random.randint(self.config.action_dim)
        state_norm = config.state_normalizer(self._state)
        action = epsilon_greedy(1, torch.zeros(state_norm.shape[0], self.config.action_dim))
        # import pdb; pdb.set_trace()

        # Take action
        next_state, reward, done, info = self._task.step(action)

        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry

# Class for avDSR agent
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
        self.optimizer_phi = config.optimizer_fn(list(self.network.encoder.parameters()) + \
                                                 list(self.network.decoder.parameters()) + \
                                                 list(self.network.decode_r.parameters()))
        self.optimizer_psi = config.optimizer_fn(self.network.layers_sr.parameters())


        self.actor.set_network(self.network)
        self.total_steps = 0
        self.c = 1

        try:
            self.is_wb = config.is_wb
        except:
            print('is_wb config not found, using deafult.')
            self.is_wb = True

        self.track_loss = True
        if(self.track_loss):
            self.loss_rec_vec = []
            self.loss_psi_vec = []
            self.loss_rew_vec = []
            self.loss_vec = []

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
            psi_next = self.network(next_states)['psi'].detach()

        if(self.config.style == 'random'):
            next_actions = tensor(epsilon_greedy(1, torch.zeros(psi_next.shape[0], self.config.action_dim))).long()
        else:
            raise NotImplementedError
            
        psi_next = psi_next[:, next_actions, :]
        out = self.network(states)
        psi_next = self.config.discount * psi_next * (masks.unsqueeze(1).unsqueeze(2).expand(-1, 32,512))
        phi, psi, state_rec, rs = out['phi'], out['psi'], out['state_rec'], out['rewards']
        psi_next.add_(phi.clone())
        psi = psi[:, actions, :]

        loss_rec = (state_rec - tensor(states))
        loss_psi = (psi_next - psi)
        loss_rew = (rs-rewards)

        return loss_rec, loss_psi, loss_rew

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
            # import pdb; pdb.set_trace()
            loss_rec, loss_psi, loss_rew = self.compute_loss(transitions)
            loss_rec, loss_psi, loss_rew = self.reduce_loss(loss_rec), self.reduce_loss(loss_psi), self.reduce_loss(loss_rew)
            loss = (loss_psi + self.c * loss_rec).mean()

            if(self.track_loss):
                self.loss_vec.append(loss.item())
                self.loss_psi_vec.append(loss_psi.item())
                self.loss_rec_vec.append(loss_rec.item())
                self.loss_rew_vec.append(loss_rew.item())

            if(self.is_wb):
                wandb.log({"steps_loss": self.total_steps, "loss": loss.item()})

            self.optimizer.zero_grad()
            loss_rec.backward(retain_graph=True)
            loss_rew.backward(retain_graph=True)
            with config.lock:
                self.optimizer_phi.step()

            self.optimizer_psi.zero_grad()
            loss_psi.backward()
            with config.lock:
                self.optimizer_psi.step()

# Class for network 
class SRNetNatureUnsup(nn.Module):
    def __init__(self, output_dim, hidden_units_sr=(512*4,), hidden_units_psi2q=(), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetNatureUnsup, self).__init__()
        self.feature_dim = 512
        self.output_dim = output_dim
        self.gate = gate
        in_channels = 4
        
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),  # b, 16, 10, 10
            nn.ReLU(True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), 
            nn.ReLU(True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), 
            nn.ReLU(True),
            Flatten(),
            nn.Linear(7 * 7 * 64, self.feature_dim)
        )
        
        self.decoder = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 7 * 7 * 64)),
            torch_reshape(into=[64, 7, 7]),
            layer_init(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)),  # b, 16, 5, 5
            nn.ReLU(True),
            layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),  # b, 16, 5, 5
            nn.ReLU(True),
            layer_init(nn.ConvTranspose2d(32, in_channels, kernel_size=8, stride=4, output_padding=0)),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.Tanh()
        )

        self.decode_r = nn.Linear(self.feature_dim, 1)

        # layers for SR
        dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
        self.layers_sr = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.encoder(tensor(x)) # shape: b x state_dim

        # Reconstruction
        state_rec = self.decoder(phi)

        rewards = self.decode_r(phi)

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.feature_dim) # shape: b x action_dim x state_dim


        return dict(phi=phi, psi=psi, state_rec=state_rec, rewards=rewards)

# Function for unsupervised representation learning
def dsr_unsup_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    kwargs.setdefault('is_wb', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: SRNetNatureUnsup(output_dim=config.action_dim)
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 1.0, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(1e5)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = None
    config.exploration_steps = 5000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.async_actor = False
    return run_steps(avDSRAgent(config))

if __name__ == "__main__":
    select_device(0)
    game='BoxingNoFrameskip-v0'
    avdsr = dsr_unsup_pixel(game=game)

    uid = str(uuid.uuid4())
    print('Run ID is ' + uid)
    dicts = {'l_r':avdsr.loss_rec_vec, 'l_p': avdsr.loss_psi_vec, 'l': avdsr.loss_vec}
    pickle.dump(dicts, open("storage/"+uid+".lc", "wb")) # learning curves
    torch.save(avdsr.network.state_dict(), "storage/"+uid+".weights") # trained weights