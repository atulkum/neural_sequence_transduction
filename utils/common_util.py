import torch
import os
import json
import random
import numpy as np

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                #drange = np.sqrt(6. / (np.sum(wt.size())))
                drange = 0.1
                wt.data.uniform_(-drange, drange)

            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    #drange = np.sqrt(6. / (np.sum(linear.weight.size())))
    drange = 0.1
    linear.weight.data.uniform_(-drange, drange)

    if linear.bias is not None:
        linear.bias.data.fill_(0.)

def set_seed(seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda > 0:
        torch.cuda.manual_seed_all(seed)

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(filename):
    with open(filename, 'r') as fout:
        config_args = json.load(fout)

    model_config = Config(**config_args)

    return model_config