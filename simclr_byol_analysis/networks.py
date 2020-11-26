import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.decomposition import PCA

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class NNClassifier(nn.Module):
    def __init__(self, in_size=2, hidden=(5,), out_size=2):
        super(NNClassifier, self).__init__()

        dims = (in_size,) + hidden + (out_size,)
        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = F.relu

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        y = F.softmax(self.layers[-1](x))

        return y

    def forward_visible(self, x):
        out = []
        out.append(x)
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
                out.append(x)
                x = self.gate(x)
                out.append(x)
            y = self.layers[-1](x)
            out.append(y)
            y = F.softmax(y)
            out.append(y)
            return out

    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


class NN(nn.Module):
    def __init__(self, in_size=2, hidden=(5,3)):
        super(NN, self).__init__()

        dims = (in_size,) + hidden
        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = F.relu

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        y = self.layers[-1](x)

        return y

    def forward_visible(self, x):
        out = []
        out.append(x)
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
                out.append(x)
                x = self.gate(x)
                out.append(x)
            y = self.layers[-1](x)
            out.append(y)
            return out

class NNSimCLR(nn.Module):
    def __init__(self, in_size=2, hidden=(5,3)):
        super(NN, self).__init__()

        dims = (in_size,) + hidden
        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = F.relu

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        y = self.layers[-1](x)

        return y

    def forward_visible(self, x):
        out = []
        out.append(x)
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
                out.append(x)
                x = self.gate(x)
                out.append(x)
            y = self.layers[-1](x)
            out.append(y)
            return out