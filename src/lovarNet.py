"""
@author:chenyankai
@file:lovarNet.py
@time:2021/06/16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.aggregator as aggregator

class LovarNet(nn.Module):
    def __init__(self, n_user, n_item, dim):
        super(LovarNet).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.dim = dim

        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)

        # Glorot initialization with gain relu
        nn.init.xavier_uniform_(self.user_emb.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.item_emb.weight, gain=nn.init.calculate_gain('relu'))


    def _get_loss(self):
        return


    def forward(self):
        return self._get_loss()