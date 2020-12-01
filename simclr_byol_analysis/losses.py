import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

def cosine_pairwise(x):
    x = x.unsqueeze(0)
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    cos_sim_pairwise = cos_sim_pairwise.squeeze(0)
    return cos_sim_pairwise

def simclr_loss(z_pos, z_neg):
    z = torch.cat((z_pos,z_neg),0)
    z = F.normalize(z, dim=1)
    s = cosine_pairwise(z)
    loss = torch.zeros(1)
    for i in range(s.size(0)):
        for j in range(s.size(1)):
            if((i >= s.size(0) // 2) and (j >= s.size(0) // 2)):
                loss += s[i, j]
            else:
                loss -= s[i, j]
    return loss

def simclr_loss_v2(z_pos, z_neg):
    z = torch.cat((z_pos,z_neg),0)
    z = F.normalize(z, dim=1)
    s = cosine_pairwise(z)
    loss = torch.zeros(1)
    for i in range(s.size(0)//2):
        loss -= s[i+s.size(0)//2, i]
    return loss

def simclr_loss_v3(z_pos, z_neg, tau=1):
    z = torch.cat((z_pos,z_neg),0)
    z = F.normalize(z, dim=1)
    s = cosine_pairwise(z)
    s = s/tau
    loss = torch.zeros(1)
    for i in range(s.size(0)//2):
        num = torch.exp(s[i+s.size(0)//2, i])
        den = torch.exp(torch.sum(s[i,:])-s[i,i])
        loss = loss-torch.log(num/den)
    loss = loss/s.size(0)
    return loss

def simclr_loss_mse(z_pos, z_neg, tau=1):
    z = torch.cat((z_pos,z_neg),0)
    z = F.normalize(z, dim=1)
    s = cosine_pairwise(z)
    s = s/tau
    loss = torch.zeros(1)
    for i in range(s.size(0)//2):
        num = torch.exp(s[i+s.size(0)//2, i])
        den = torch.exp(torch.sum(s[i,:])-s[i,i])
        loss = loss-torch.log(num/den)
    loss = loss/s.size(0)
    return loss 



class contrastive_loss(nn.Module):
    def __init__(self, tau=2, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss

class NTXentLoss(torch.nn.Module):
    """
    Adpoted from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
