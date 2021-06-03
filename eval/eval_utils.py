import numpy as np
import torch
from data_processing.timit import get_dataloader

def label_error_rate(path_true, path_hat):
    n = len(path_true)
    m = len(path_hat)
    dp = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dp[i][0] = i
    for j in range(1, m+1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if path_true[i-1] != path_hat[j-1]:
                dp[i][j] = min(dp[i-1][j-1] + 1, min(dp[i-1][j] + 1, dp[i][j-1] + 1))
            else:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[n][m] / n

def evaluate(test_generator, model, batch_size, is_cuda):
    dataloader = get_dataloader(test_generator, batch_size, False)
    model.eval()
    lers = []
    with torch.set_grad_enabled(False):
        for i_batch, inputs in enumerate(dataloader):
            if is_cuda:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
            logprobs = model(**inputs)
            path_hat = model.prefix_search_decode(logprobs, inputs['length'])
            path_true = inputs['phone'].cpu().data.numpy()
            for b in len(path_hat):
                lers.append(label_error_rate(path_true[b], path_hat[b]))
    ler = np.mean(lers)
    print(f'Label Error Rate {ler:.5f}', flush=True)
    return ler