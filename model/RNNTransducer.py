import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Conv1d, ReLU


class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()


class TranscriptionNetwork(nn.Module):
    def __init__(self):
        super(TranscriptionNetwork, self).__init__()


class RNNTranducer(nn.Module):
    def __init__(self):
        super(RNNTranducer, self).__init__()