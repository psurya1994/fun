import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.decomposition import PCA


def dataset_pattern_010():
    """
    The x axis can be seen as data agumentation for a given point.
    """
    X = np.zeros((18,2))
    X[:,0] = np.hstack((np.arange(-1,1.01,.4),np.arange(-1,1.01,.4),np.arange(-1,1.01,.4)))
    X[6:,1] = np.hstack((np.ones(6), -np.ones(6)))
    y = np.hstack((np.zeros(6), np.ones(12)))
    return X, y

class DataSupervised(Dataset):
    """Supervised data."""

    def __init__(self, id='moon'):
        size = 200
        if(id == 'moon'):
            X,y = sklearn.datasets.make_moons(size,noise=0.2)
        elif(id=='circles'):
            X, y = sklearn.datasets.make_circles(n_samples=size)
        elif(id == '010'):
            X, y = dataset_pattern_010()
        else:
            raise
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)
        self.X, self.y = X, y
        self.length = size 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx]

    def full_numpy(self):
        return self.X.numpy(), self.y.numpy()
    
    def full(self):
        return self.X, self.y


class DataUnsupervised(Dataset):
    """Positive and negative samples."""

    def __init__(self, id='moon'):
        size = 200
        if(id == 'moon'):
            X,y = sklearn.datasets.make_moons(size,noise=0.2) 
        elif(id=='circles'):
            X, y = sklearn.datasets.make_circles(n_samples=size)
        else:
            raise
        
        self.X, self.y = X, y
        self.X_pos = X[y==1,:]
        self.X_neg = X[y==0,:]
        self.length = size // 2

    def full(self):
        return torch.from_numpy(self.X).type(torch.FloatTensor), torch.from_numpy(self.y).type(torch.LongTensor)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.X_pos[idx, :]).type(torch.FloatTensor), \
                torch.from_numpy(self.X_neg[idx, :]).type(torch.FloatTensor)