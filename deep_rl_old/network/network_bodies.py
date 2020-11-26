#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

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

class SRIdentityBody(nn.Module):
    def __init__(self, state_dim):
        super(SRIdentityBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class MinigridConvBody(nn.Module):
    def __init__(self, in_channels=3):
        super(MinigridConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.fc4 = layer_init(nn.Linear(9 * 9 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class torch_reshape(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 64, 9, 9)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class SRNetImage(nn.Module):
    def __init__(self, output_dim, hidden_units_sr=(512*4,), hidden_units_psi2q=(), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetImage, self).__init__()
        self.feature_dim = 512
        self.output_dim = output_dim
        self.gate = gate

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),  # b, 16, 10, 10
            nn.ReLU(True),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)), 
            nn.ReLU(True),
            Flatten(),
            nn.Linear(9 * 9 * 64, self.feature_dim)
        )

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 9 * 9 * 64)),
            torch_reshape(),
            layer_init(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)),  # b, 16, 5, 5
            nn.ReLU(True),
            layer_init(nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1)),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.Tanh()
        )

        # layers for SR
        dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
        self.layers_sr = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

        # SR final head layer
        if(config == 0):
            self.psi2q = Psi2QNet(output_dim, self.feature_dim)
        if(config == 1):
            self.psi2q = Psi2QNetFC(output_dim, self.feature_dim, hidden_units=hidden_units_psi2q)

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.encoder(tensor(x)) # shape: b x state_dim

        # Reconstruction
        state_est = self.decoder(phi)

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.feature_dim) # shape: b x action_dim x state_dim
        q_est = self.psi2q(psi)

        return phi, psi, state_est, q_est

class SRNetNature_v2_psi(nn.Module):
    def __init__(self, output_dim, feature_dim=512, hidden_units_sr=(512*4,), hidden_units_psi2q=(2048,512), gate=F.relu):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetNature_v2_psi, self).__init__()
        self.feature_dim = feature_dim
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

        # layers for SR
        dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
        self.layers_sr = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

        # layers for FC
        dims_fc = (self.feature_dim * output_dim,) + hidden_units_psi2q + (output_dim,)
        self.psi2q = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_fc[:-1], dims_fc[1:])])
        self.to(Config.DEVICE)

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.encoder(tensor(x)) # shape: b x state_dim

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)

        q_est = psi
        for layer in self.psi2q[:-1]:
            q_est = self.gate(layer(q_est))
        q_est = self.psi2q[-1](q_est)

        return q_est

class SRNetNature_v2_phi(nn.Module):
    def __init__(self, output_dim, feature_dim=512, hidden_units_psi2q=(2048,512), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetNature_v2_phi, self).__init__()
        self.feature_dim = feature_dim
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

        # layers for FC
        dims_fc = (self.feature_dim,) + hidden_units_psi2q + (output_dim,)
        self.psi2q = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_fc[:-1], dims_fc[1:])])
        self.to(Config.DEVICE)

    def forward(self, x):

        q_est = self.encoder(tensor(x))
        for layer in self.psi2q[:-1]:
            q_est = self.gate(layer(q_est))
        q_est = self.psi2q[-1](q_est)

        return q_est

class SRNetImage_v2(nn.Module):
    def __init__(self, output_dim, hidden_units_psi2q=(1024,512), gate=F.relu):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetImage_v2, self).__init__()
        self.feature_dim = 512
        self.output_dim = output_dim
        self.gate = gate

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),  # b, 16, 10, 10
            nn.ReLU(True),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)), 
            nn.ReLU(True),
            Flatten(),
            nn.Linear(9 * 9 * 64, self.feature_dim)
        )

        # layers for SR
        dims_fc = (self.feature_dim,) + hidden_units_psi2q + (output_dim,)

        self.psi2q = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_fc[:-1], dims_fc[1:])])

        self.to(Config.DEVICE)

    def forward(self, x):

        q_est = self.encoder(tensor(x))
        for layer in self.psi2q[:-1]:
            q_est = self.gate(layer(q_est))
        q_est = self.psi2q[-1](q_est)

        return q_est

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.state_dim = state_dim
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi


class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
