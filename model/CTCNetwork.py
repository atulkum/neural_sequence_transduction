import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Conv1d, ReLU
from utils import common_util
import numpy as np
from model.ctc_beam_decoder import prefix_serach_decoding

class ConnectionistTemporalClassification(nn.Module):
    def __init__(self, model_config):
        super(ConnectionistTemporalClassification, self).__init__()

        self.lstm = nn.LSTM(model_config.input_size,
                            model_config.lstm_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.model_config = model_config

        self.clf = nn.Linear(2 * model_config.lstm_dim, model_config.num_tags + 1)

        common_util.init_lstm_wt(self.lstm)
        common_util.init_linear_wt(self.clf)

    def forward(self, mfcc, length, **kwargs):
        lengths = length.view(-1).tolist()
        packed = pack_padded_sequence(mfcc, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)
        lstm_feats, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        lstm_feats = lstm_feats.contiguous()
        logits = self.clf(lstm_feats)
        logprobs = logits.log_softmax(dim=2)
        return logprobs

    def get_loss(self, logprobs, phone, length):
        b, t_k, d = list(logprobs.size())
        loss = 0
        for i in range(b):
            curr_len = length[i]
            curr_logit_logprob = logprobs[i]
            curr_phone = phone[i]
            neg_log_likelyhood_single = -self.ctc_likelihood_single(curr_logit_logprob,
                                                                       curr_len, curr_phone)
            loss += neg_log_likelyhood_single
        return loss

    '''
    For odd index return blank label 
    otherwise map the index to ground truth phone 
    index
    '''
    def get_phone_id(self, s, phone):
        if s % 2 == 1:
            return 0
        else:
            return phone[s // 2 - 1] + 1
    def has_same_label(self, s, phone):
        idx = s // 2
        return idx > 1 and phone[s // 2 - 1] == phone[s // 2 - 2]

    def ctc_likelihood_single(self, log_y, T, phone):
        num_phone = len(phone)
        S = 2*num_phone + 1
        log_alpha = torch.zeros((T + 1, S + 1))
        t, s = 1, 1
        log_alpha[t, s] = log_y[t-1, self.get_phone_id(s, phone)]
        t, s = 1, 2
        log_alpha[t, s] = log_y[t - 1, self.get_phone_id(s, phone)]
        for t in range(2, T + 1):
            for s in range(1, S + 1):
                y_ = log_y[t - 1, self.get_phone_id(s, phone)]
                # blank or same labels
                if s % 2 == 1 or self.has_same_label(s, phone):
                    sum_prev = torch.stack([log_alpha[t-1, s], log_alpha[t - 1, s-1]])
                    log_alpha[t, s] = sum_prev.logsumexp(dim=0) + y_
                else:
                    sum_prev = torch.stack([log_alpha[t - 1, s],
                                            log_alpha[t - 1, s - 1], log_alpha[t - 1, s-2]])
                    log_alpha[t, s] = sum_prev.logsumexp(dim=0) + y_
        log_likelihood = torch.stack([log_alpha[T, S], log_alpha[T, S - 1]]).logsumexp(dim=0)
        return log_likelihood

    def path_to_str(self, path):
        new_path = [path[0]] + [path[i] for i in range(1, len(path)) if path[i-1] != path[i]]
        return [p -1 for p in new_path if p > 0]

    def best_path_decode(self, logprobs, length):
        logprobs = logprobs.cpu().data.numpy()
        length = length.cpu().data.numpy()
        out_paths = []
        for y, curr_len in zip(logprobs, length):
            path = np.argmax(y[:curr_len, :], axis=1)
            out_paths.append(self.path_to_str(path))
        return out_paths

    def prefix_search_decode(self, logprobs, length):
        logprobs = logprobs.cpu().data.numpy()
        length = length.cpu().data.numpy()
        out_paths = []
        for logprob, T in zip(logprobs, length):
            path = prefix_serach_decoding(logprob, T, self.model_config.blank_threshold)
            out_paths.append(self.path_to_str(path))
        return out_paths
