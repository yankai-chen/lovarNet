"""
@author:chenyankai
@file:aggregator.py
@time:2021/06/16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(object):
    def __init__(self, batch_size, in_dim, out_dim, dropout, act, bias=False):
        self.batch_size = batch_size,
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.bias = bias

    def aggregator(self, root_vec, ngh_vecs):
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


# class Aggregator(nn.Module):
#     def __init__(self, dim, agg_type, u_ngh_num, dropout):
#         super(Aggregator, self).__init__()
#         self.dim = dim
#         self.agg_type = agg_type
#         self.u_ngh_num = u_ngh_num
#         self.dropout = dropout
#
#         if self.agg_type == 'concat':
#             self.linear = nn.Sequential(
#                 nn.Linear(in_features=2*self.dim, out_features=self.dim, bias=True),
#                 nn.Dropout(self.dropout)
#             )
#         elif self.agg_type in ['sum', 'ngh']:
#             self.linear = nn.Sequential(
#                 nn.Linear(in_features=self.dim, out_features=self.dim, bias=True),
#                 nn.Dropout(self.dropout)
#             )
#         else:
#             raise NotImplementedError
#
#     def _compute_att(self):
#         pass
#
#     def _aggregate(self):
#         pass
#
#     def forward(self):
#         pass


