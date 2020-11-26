#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
retrain let's you retrain the network with specific features loaded and frozen weights.
retrain_v2 let's you do the same with unfrozen weights.
"""
from deep_rl import *
import uuid
import sys

# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.history_length = 1
    config.batch_size = 10
    config.discount = 0.99
    config.max_steps = 1e5

    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length)

    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.async_actor = False
    run_steps(DQNAgent(config))

def dqn_pixel(**kwargs):
    # generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    kwargs.setdefault('is_wb', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: SRNetNatureSup(config.action_dim, in_channels=config.history_length)
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(2e7)
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
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    # config.exploration_steps = 100
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.async_actor = True
    run_steps(DQNAgent_v2(config))

class SRNetNatureSup(nn.Module):
    def __init__(self, output_dim, in_channels=4, hidden_units_sr=(512*4,), hidden_units_psi2q=(1000,), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetNatureSup, self).__init__()
        self.feature_dim = 512
        self.output_dim = output_dim
        self.gate = gate
        
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

        # layers for SR
        dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
        self.layers_sr = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

        dims_q = (self.feature_dim * output_dim,) + hidden_units_psi2q + (output_dim,)
        self.psi2q = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_q[:-1], dims_q[1:])])

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.encoder(tensor(x)) # shape: b x state_dim

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)
        # psi = psi.view(psi.size(0), self.output_dim, self.feature_dim) # shape: b x action_dim x state_dim

        q = psi
        for layer in self.psi2q[:-1]:
            q = self.gate(layer(q))
        q = self.psi2q[-1](q)

        return dict(phi=phi, psi=psi, q=q)

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(0)

    # game = 'BreakoutNoFrameskip-v4'
    game = 'BoxingNoFrameskip-v0'
    version = 'phi'
    # game = 'PixelGridWorld'


    # READFILE='storage/41-avdsr-trained-boxing-5e-4-1e5.weights'
    READFILE = sys.argv[1]
    
    uid = str(uuid.uuid4())[-5:]
    print('Run ID is ' + uid)
    # dqn_pixel(game=game, n_step=1, replay_cls=UniformReplay, async_replay=False)

    dqn_pixel(weights_file=READFILE, version=version, game=game, n_step=1, replay_cls=UniformReplay, async_replay=False, tag='retrain_dqn_feature_'+game+'_'+'version-shuffle_'+uid)