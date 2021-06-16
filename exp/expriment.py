"""
@author:chenyankai
@file:expriment.py
@time:2021/06/16
"""
from util.loger import *
from util.data_loader import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.lovarNet import LovarNet
from util.data_loader import *


def ctr_evaluate():
    pass


def topk_evaluate():
    pass


def Exp(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    train_data = load_data()
    train_data.to(device)

    # create model object
    model = LovarNet(n_user=args.n_user, n_item=args.n_item, dim=args.dim).to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_)

    max_eval_auc = 0.0
    best_test_auc = 0.0
    best_test_f1 = 0.0
    best_epoch1 = 0
    eval_precision_list = []
    eval_recall_list = []
    eval_ndcg_list = []

    test_precision_list = []
    test_recall_list = []
    test_ndcg_list = []

    best_eval_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch2 = [0, 0, 0, 0, 0, 0]

    for i in range(args.epoch):
        start = 0
        iter = 0
        total_loss = 0
        np.random.shuffle(train_data)

        model.train()



