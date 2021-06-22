"""
@author:chenyankai
@file:model.py
@time:2021/06/16
"""
from baselines.hashGNN.aggregator import *
import torch
import torch.nn as nn
import numpy as np
from baselines.hashGNN.node_encoder import *

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient
    More details can be found in https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    """
    return torch.sign(x)


class BipartiteGraphNodeEncoder(object):
    def __init__(self, u_encoder, i_encoder, encoding_schema='u-i-u', ids=None):
        self.u_encoder = u_encoder
        self.i_encoder = i_encoder
        self.ids = ids
        assert encoding_schema == 'u-i-u' or encoding_schema == 'u-i-i'
        self.encoding_schema = encoding_schema

    def encode_user_layer(self, layer_features):
        """
        :param layer_features: a list of features for each layer. layer-0 is the root node to be finally encoded.
            shape of each element = [-1, i/u_feature_dim]
        :return: a list of the same length, all elements are corresponding encoded features
        """
        # [B, L...(max_layer-layer_from_bottom)...L, i_dim]
        ret = []
        for i, layer_feat in enumerate(layer_features):



class HashGNN(object):
    def __init__(self, args, global_step, graph_input,
                 u_id_encode=(0, 0), i_id_encode=(0, 0),
                mode='train', ps_num=None):

        self.u_emb_dim = []
        self.i_emb_dim = []
        self.dim = args.dim

        self.u_id_encode = u_id_encode
        self.i_id_encode = i_id_encode
        self.args = args
        self.dropout = args.dropout
        if self.u_id_encode[0] > 0:
            self.ids = 1
        else:
            self.ids = None
        self.act = None
        if self.args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU
        elif self.args.activation == 'sigmoid':
            self.act = nn.Sigmoid
        elif self.args.activation == 'tanh':
            self.act = nn.Tanh

        self.is_training = True if mode == 'train' else False
        self.u_ngh_num = [] if args.u_ngh_num == '' else [int(x) for x in args.u_ngh_num.split(',')]
        self.i_ngh_num = [] if args.i_ngh_num == '' else [int(x) for x in args.i_ngh_num.split(',')]
        self.u_emb_dim = [] if args.u_embed == '' else [int(x) for x in args.u_embed.split(',')]
        self.i_emb_dim = [] if args.u_embed == '' else [int(x) for x in args.u_embed.split(',')]
        self.u_depth = len(self.u_ngh_num)
        self.i_depth = len(self.i_ngh_num)

        if args.agg_type == 'mean':
            self.aggregator = MeanAggregator
        elif args.agg_type == 'max_pooling':
            self.aggregator = MaxpoolingAggregator
        elif args.agg_type == 'gcn':
            self.aggregator = GCNAggregator
        else:
            print('agg_type error.')
            raise NotImplementedError

    def build_graph(self, mode):
        self.u_encoder = NodeEncoder(self.args, (self.u_emb_dim[0],),
                                     act=self.act, dropout=self.dropout,
                                     encode_id_num=self.u_id_encode[0],
                                     encode_id_dim=self.u_id_encode[1])

        self.i_encoder = NodeEncoder(self.args, (self.i_emb_dim[0], ),
                                     act=self.act, dropout=self.dropout,
                                     encode_id_num=self.i_id_encode[0],
                                     encode_id_dim=self.i_id_encode[1])

        encoding_schema = self.args.encoding_schema


