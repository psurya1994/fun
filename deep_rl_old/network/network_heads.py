#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from torch.nn.parameter import Parameter

class torch_reshape(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 64, 9, 9)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

# class SRNetImage(nn.Module):
#     def __init__(self, output_dim, hidden_units_sr=(512*4,), hidden_units_psi2q=(), gate=F.relu, config=1):
#         """
#         This network has two heads: SR head (SR) and reconstruction head (rec).
#         config -> type of learning on top of state abstraction
#             0 - typical SR with weights sharing
#             1 - learning SR without weights sharing
#         """
#         super(SRNetImage, self).__init__()
#         self.feature_dim = 512
#         self.output_dim = output_dim
#         self.gate = gate

#         self.encoder = nn.Sequential(
#             layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),  # b, 16, 10, 10
#             nn.ReLU(True),
#             layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)), 
#             nn.ReLU(True),
#             Flatten(),
#             nn.Linear(9 * 9 * 64, self.feature_dim)
#         )

#         self.decoder = nn.Sequential(
#             layer_init(nn.Linear(self.feature_dim, 9 * 9 * 64)),
#             torch_reshape(),
#             layer_init(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)),  # b, 16, 5, 5
#             nn.ReLU(True),
#             layer_init(nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1)),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.Tanh()
#         )

#         # layers for SR
#         dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
#         self.layers_sr = nn.ModuleList(
#             [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

#         # SR final head layer
#         if(config == 0):
#             self.psi2q = Psi2QNet(output_dim, self.feature_dim)
#         if(config == 1):
#             self.psi2q = Psi2QNetFC(output_dim, self.feature_dim, hidden_units=hidden_units_psi2q)

#         self.to(Config.DEVICE)

#     def forward(self, x):

#         # Finding the latent layer
#         phi = self.encoder(tensor(x)) # shape: b x state_dim

#         # Reconstruction
#         state_est = self.decoder(phi)

#         # Estimating the SR from the latent layer
#         psi = phi
#         for layer in self.layers_sr[:-1]:
#             psi = self.gate(layer(psi))
#         psi = self.layers_sr[-1](psi)
#         psi = psi.view(psi.size(0), self.output_dim, self.feature_dim) # shape: b x action_dim x state_dim
#         q_est = self.psi2q(psi)

#         return phi, psi, state_est, q_est

class Psi2QNet(nn.Module):
    def __init__(self, output_dim, feature_dim):
        super(Psi2QNet, self).__init__()
        self.w = Parameter(torch.Tensor(feature_dim))
        nn.init.constant_(self.w, 0) # CHECK for better initialization
        self.to(Config.DEVICE)
    
    def forward(self, psi):
        return torch.matmul(psi, self.w)

class Psi2QNetFC(nn.Module):
    def __init__(self, output_dim, feature_dim, hidden_units=(), gate=F.relu):
        super(Psi2QNetFC, self).__init__()

        dims = (feature_dim*output_dim,) + hidden_units + (output_dim,)
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.to(Config.DEVICE)
    
    def forward(self, psi):
        out = psi.view(psi.size(0), -1)
        for layer in self.layers[:-1]:
            out = self.gate(layer(out))
        out = self.layers[-1](out)
        return out

class SRNetCNN(nn.Module):
    """
    Added by Surya.
    SR fully connected body network.
    """
    def __init__(self, output_dim, body, hidden_units=(1000,), gate=F.relu, config=1):
        """
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetCNN, self).__init__()
        self.body = body
        self.output_dim = output_dim
        self.width = int(np.sqrt(body.feature_dim))

        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.fc_size = 128 * ((self.width - (3-1)*2) // 2) ** 2

        # FC layers
        dims = (self.fc_size,) + hidden_units + (body.feature_dim * output_dim,)
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        
        self.gate = gate
        self.feature_dim = body.feature_dim * output_dim
        if(config == 0):
            self.psi2q = Psi2QNet(output_dim, body.feature_dim)
        if(config == 1):
            self.psi2q = Psi2QNetFC(output_dim, body.feature_dim)
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x)) # shape: b x state_dim
        psi = phi

        # Convert to image
        psi = psi.view(phi.size(0), 1, self.width, self.width)

        # Conv layers
        psi = self.conv1(psi)
        psi = F.relu(psi)
        psi = self.conv2(psi)
        psi = F.relu(psi)
        psi = F.max_pool2d(psi, 2)
        psi = torch.flatten(psi, 1)

        # FC layers
        for layer in self.layers[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.body.feature_dim) # shape: b x action_dim x state_dim
        out = self.psi2q(psi)

        return phi, psi, out

class SRNetWithReconstruction(nn.Module):
    def __init__(self, output_dim, body, hidden_units_sr=(64,64), hidden_units_rec=(64,64), hidden_units_psi2q=(), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetWithReconstruction, self).__init__()
        self.body = body
        self.output_dim = output_dim
        self.gate = gate

        dims_sr = (body.feature_dim,) + hidden_units_sr + (body.feature_dim * output_dim,)
        dims_rec = (body.feature_dim,) + hidden_units_rec + (body.state_dim,) # this doesn't work for conv layers

        # layers for SR
        self.layers_sr = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])
        
        # layers for reconstruction
        self.layers_rec = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_rec[:-1], dims_rec[1:])])

        # SR final head layer
        if(config == 0):
            self.psi2q = Psi2QNet(output_dim, body.feature_dim)
        if(config == 1):
            self.psi2q = Psi2QNetFC(output_dim, body.feature_dim, hidden_units=hidden_units_psi2q)

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.body(tensor(x)) # shape: b x state_dim

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.body.feature_dim) # shape: b x action_dim x state_dim
        q_est = self.psi2q(psi)

        # Reconstructing the state from the latent layer    
        k = phi
        for layer in self.layers_rec[:-1]:
            k = self.gate(layer(k))
        state_est = self.layers_rec[-1](k)


        return phi, psi, state_est, q_est

class SRNet(nn.Module):
    """
    Added by Surya.
    SR fully connected body network.
    """
    def __init__(self, output_dim, body, hidden_units=(), gate=F.relu, config=1):
        """
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNet, self).__init__()
        self.body = body
        self.output_dim = output_dim# TODO: check if this is the right way to do it
        dims = (body.feature_dim,) + hidden_units + (body.feature_dim * output_dim,)
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        
        self.gate = gate
        self.feature_dim = body.feature_dim * output_dim
        if(config == 0):
            self.psi2q = Psi2QNet(output_dim, body.feature_dim)
        if(config == 1):
            self.psi2q = Psi2QNetFC(output_dim, body.feature_dim)

        self.to(Config.DEVICE)



    def forward(self, x):
        phi = self.body(tensor(x)) # shape: b x state_dim
        psi = phi
        for layer in self.layers[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.body.feature_dim) # shape: b x action_dim x state_dim
        out = self.psi2q(psi)

        return phi, psi, out
    

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.phi_params = list(self.phi_body.parameters())

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters()) + self.phi_params
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters()) + self.phi_params

        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}


class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2
