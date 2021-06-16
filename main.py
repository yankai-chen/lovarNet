"""
@author:chenyankai
@file:main.py
@time:2021/06/16
"""
import argparse
from exp.expriment import Exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for lovarNet")
    parser.add_argument('--data_dir', type=str, default='dataset/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='ml-1m', help='select a dataset, e.g., ml-1m')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    parser.add_argument('--dim', type=int, default=32, help='dimension of nodes and edges')
    parser.add_argument('--agg_type', type=str, default='mean', help='aggregator from [mean,  sum, ngh]')
    # parser.add_argument('--n_layer', type=int, default=2, help='number of layers')
    parser.add_argument('--u_ngh_num', type=str, default='10,5', help='the neighbor numbers of users on all layers')
    parser.add_argument('--i_ngh_num', type=str, default='10', help='the neighbor numbers of items on all layers')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop count threshold')
    parser.add_argument('--hash_size', type=int, default=25, help='hash size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lambda_', type=float, default=1e-1, help='lambda when calculating L2 loss')

    parser.add_argument('--seed', type=int, default=1111, help='seed')

    args = parser.parse_args()
    args.saved_dir = 'trained_model/{}/'.format(args.data_name)

    Exp(args)