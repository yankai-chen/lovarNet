"""
@author:chenyankai
@file:node_encoder.py
@time:2021/06/17
"""
import torch
import torch.nn as nn

class NodeEncoder(object):
    """
    Individual node embedding encoder
    encoder each node (with id & attributes) into a vector
    """

    def __init__(self, args, dense_dims=(512, ), act='leaky_relu', dropout=0., encode_id_num=0, encode_id_dim=32):
        self.args = args
        self.dense_dims = dense_dims
        self.act = None
        self.dropout = dropout
        self.encode_id_num = encode_id_num
        self.encode_id_dim = encode_id_dim

        if act == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()

        if self.encode_id_num > 0 and self.encode_id_dim > 0:
            self.embeddings = nn.Embedding(self.encode_id_num, self.encode_id_dim)

        # Glorot initialization with gain relu
        nn.init.xavier_uniform_(self.embeddings.weight, gain=nn.init.calculate_gain('relu'))

        tmp_dense_dim = [self.encode_id_dim]
        for _, dense_dim in enumerate(self.dense_dims):
            tmp_dense_dim.append(dense_dim)

        self.mlp = nn.Sequential()

        for i in range(len(tmp_dense_dim) - 1):
            self.mlp.add_module(name='linear-{}'.format(i), module=nn.Linear(in_features=tmp_dense_dim[i], out_features=tmp_dense_dim[i+1], bias=True))
            self.mlp.add_module(name='act-{}'.format(i), module=self.act)

        if self.dropout:
            self.mlp.add_module(name='drop', module=nn.Dropout(self.dropout))

    def encode(self):
        """
        :return:
        """
        x_batch = self.embeddings.weight
        output = self.mlp(x_batch)
        return output
