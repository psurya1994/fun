#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class torch_reshape(torch.nn.Module):
    def __init__(self, into=[64, 9, 9]):
        super(torch_reshape, self).__init__()
        self.into = into
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.into[0], self.into[1], self.into[2])

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

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

        return dict(q=q_est)

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

        return dict(q=q_est)

class SRNetNature_v2_phi_40(nn.Module):
    def __init__(self, output_dim, feature_dim=512, hidden_units_psi2q=(2048,512), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetNature_v2_phi_40, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.gate = gate
        in_channels = 4
        
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),  # b, 16, 10, 10
            nn.ReLU(True),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)), 
            nn.ReLU(True),
            Flatten(),
            nn.Linear(9 * 9 * 64, self.feature_dim)
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

        return dict(q=q_est)

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

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


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
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
