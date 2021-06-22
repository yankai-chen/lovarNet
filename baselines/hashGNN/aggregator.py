"""
@author:chenyankai
@file:aggregator.py
@time:2021/06/16
"""
import torch
import torch.nn as nn


class Aggregator(object):
    def __init__(self, batch_size, in_dim, out_dim, dropout, act, bias=False):
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.bias = bias

    def forward(self, root_vec, ngh_vecs):
        """
        :param root_vec: [B, dim]
        :param ngh_vecs: [B*ngh_num, dim]
        :return:        [B, dim]
        """
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, in_dim, out_dim, dropout, bias=False, act=nn.ReLU):
        super(MeanAggregator, self).__init__(
            batch_size=batch_size,
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            act=act,
            bias=bias
        )
        print('init mean aggregator')

        self.linear = nn.Sequential(
            nn.Linear(in_features=2*in_dim, out_features=out_dim, bias=self.bias),
            nn.Dropout(self.dropout)
        )

        nn.init.xavier_uniform_(self.linear[0].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, root_vec, ngh_vecs):
        """
        :param root_vec: [B, dim]
        :param ngh_vecs: [B*num_ngh, dim]
        :return:    [B, dim]
        """
        root_vec = root_vec.view(-1, self.in_dim)
        ngh_vecs = ngh_vecs.view(self.batch_size, -1, self.in_dim)
        ngh_mean = torch.mean(ngh_vecs, dim=1)
        cat = torch.cat((root_vec, ngh_mean), dim=-1)
        output = self.act(self.linear(cat))

        return output


class GCNAggregator(Aggregator):
    def __init__(self, batch_size, in_dim, out_dim, dropout, bias=False, act=nn.ReLU):
        super(GCNAggregator, self).__init__(
            batch_size=batch_size,
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            act=act,
            bias=bias
        )
        print('init GCN aggregator')

        self.linear = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.bias),
            nn.Dropout(self.dropout)
        )

        nn.init.xavier_uniform_(self.linear[0].weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, root_vec, ngh_vecs):
        """
        :param root_vec: [B, dim]
        :param ngh_vecs: [B*ngh_num, dim]
        :return:         [B, dim]
        """
        ngh_vecs = ngh_vecs.view(self.batch_size, -1, self.in_dim)
        _, ngh_num, _ = ngh_vecs.shape
        root_vec = root_vec.expand(self.batch_size, ngh_num, self.in_dim)
        means = torch.mean(torch.cat((ngh_vecs, root_vec), dim=1), dim=1)
        output = self.act(self.linear(means))

        return output


class MaxpoolingAggregator(Aggregator):
    """Simple max-pooling aggregator reproduced from the HashGNN source codes
    """
    def __init__(self, batch_size, in_dim, out_dim, dropout, bias=False, act=nn.ReLU):
        super(MaxpoolingAggregator, self).__init__(
            batch_size=batch_size,
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            act=act,
            bias=bias
        )
        print('max pooling concat aggregator')

        self.linear = nn.Sequential(
            nn.Linear(in_features=2*in_dim, out_features=out_dim, bias=self.bias),
            nn.Dropout(self.dropout)
        )

        nn.init.xavier_uniform_(self.linear[0].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, root_vec, ngh_vecs):
        """
        :param root_vec: [B, dim]
        :param ngh_vecs: [B*ngh_num, dim]
        :return:        [B, dim]
        """
        ngh_vecs = ngh_vecs.view(self.batch_size, -1, self.in_dim)
        ngh_vecs, _ = torch.max(ngh_vecs, dim=1)
        cat = torch.cat((root_vec, ngh_vecs), dim=-1)
        output = self.act(self.linear(cat))

        return output