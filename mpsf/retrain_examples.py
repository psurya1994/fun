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

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
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
    run_steps(DQNAgent(config))

def retrain_dqn_feature(**kwargs):
    """
        version -> specify phi or psi
        weights -> specify weights init
    """

    generate_tag(kwargs)
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


    if(config.version == 'phi'):
        # For psi version
        config.network_fn = lambda: SRNetNature_v2_phi_40(output_dim=config.action_dim, feature_dim=512, hidden_units_psi2q=(4096,2048,1024,))
    else:
        # For psi version
        config.network_fn = lambda: SRNetNature_v2_psi(output_dim=config.action_dim, feature_dim=512, hidden_units_sr=(512*4,), hidden_units_psi2q=(2048,512))

    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    if('Boxing' in config.game):
        config.history_length = 4
    else:
        config.history_length = 1
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

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # -1 is CPU, a positive integer is the index of GPU
    # select_device(-1)
    select_device(0)

    # game = 'BoxingNoFrameskip-v4'
    # READFILE = '/home/mila/p/penmetss/trash/DeepRLv2/storage/41-avdsr-trained-boxing-512.weights'

    # game = 'PixelGridWorld'
    # READFILE = '/home/mila/p/penmetss/trash/DeepRLv3/storage/41-avdsr-trained-pixelgrid.weights'

    game = 'MiniGrid-Empty-5x5-v0'
    READFILE = '/home/mila/p/penmetss/trash/DeepRLv2/storage/40-avdsr-trained-minigrid.weights'

    version = 'phi'
    # torch.nn.Module.dump_patches = True

    # weights = torch.load(READFILE).state_dict()

    # if(version == 'phi'):
    #     # For phi version
    #     to_remove = ['decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias', 'decoder.6.weight', 'decoder.6.bias', 'layers_sr.0.weight', 'layers_sr.0.bias', 'layers_sr.1.weight', 'layers_sr.1.bias', 'psi2q.layers.0.weight', 'psi2q.layers.0.bias']
    # else:
    #     # For psi version
    #     to_remove = ['decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias', 'decoder.6.weight', 'decoder.6.bias', 'psi2q.layers.0.weight', 'psi2q.layers.0.bias']

    # for key in to_remove:
    #     weights.pop(key)

    retrain_dqn_feature(weights_file=READFILE, version=version, game=game, n_step=1, replay_cls=UniformReplay, async_replay=False, tag='retrain_dqn_feature_'+game+'_'+'version')

    # dqn_pixel(game=game, n_step=1, replay_cls=UniformReplay, async_replay=False)
