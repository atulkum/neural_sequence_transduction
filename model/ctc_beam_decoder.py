import numpy as np

def logsumexp(a, b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    mx = max(a, b)
    mn = min(a, b)
    return np.log(1.0 + np.exp(mn-mx)) + mx

def logsubstractexp(a, b):
    #assert a >= b
    mx = max(a, b)
    mn = min(a, b)
    return np.log(1. - np.exp(mn-mx)) + mx

class GammaEntry(object):
    def __init__(self):
        self.blank = None
        self.other = None

class BeamEntry(object):
    def __init__(self):
        self.path = None
        self.gamma = None
        self.P_path_full = None
        self.P_path_partial = None

def prefix_search_decoding(logprob, T, threshold):
    ranges = prefix_beam_search_split(logprob, T, threshold)
    path = []
    for s, e in ranges:
        curr_logprob = logprob[s:e+1]
        curr_path = prefix_beam_search_prob(curr_logprob)
        path.extend(curr_path)
    return path

def prefix_beam_search_split(logprob, T, threshold):
    start = 0
    ranges = []
    for t in range(1, T):
        if logprob[t, 0] >= threshold:
            ranges.append((start, t))
            start = t+1
    if start < T:
        ranges.append((start, T-1))
    return ranges

#page 64 Prefix Search Decoding Algorithm
#Supervised Sequence Labelling with Recurrent Neural Networks (https://www.cs.toronto.edu/~graves/preprint.pdf
def prefix_beam_search_logprob(logprob, T):
    ZERO_LOG_PROB=None
    num_tags = logprob.shape[1] - 1
    beam = []
    gamma = []
    gamma_entry = GammaEntry()
    gamma_entry.blank = logprob[0, 0]
    gamma_entry.other = ZERO_LOG_PROB
    gamma.append(gamma_entry)

    for t in range(1, T):
        gamma_entry = GammaEntry()
        gamma_entry.blank = logprob[t, 0] + gamma[-1].blank
        gamma_entry.other = ZERO_LOG_PROB
        gamma.append(gamma_entry)

    beam_entry = BeamEntry()
    beam_entry.gamma = gamma
    beam_entry.path = []
    beam_entry.P_path_full = gamma[T - 1].blank
    beam_entry.P_path_partial = logsubstractexp(1.0, beam_entry.P_path_full)
    beam.append(beam_entry)

    l_star, P_l_star = beam_entry.path, beam_entry.P_path_full
    while len(beam) > 0:
        beam.sort(key=lambda x: x.P_path_partial)
        p_star = beam[-1]
        beam.pop()

        prob_remaining = p_star.P_path_partial
        if prob_remaining <= P_l_star:  # done
            break

        for k in range(1, num_tags + 1):
            path = p_star.path + [k]

            gamma = []
            gamma_entry = GammaEntry()
            gamma_entry.other = logprob[0, k] if p_star.path == [] else ZERO_LOG_PROB
            gamma_entry.blank = ZERO_LOG_PROB
            gamma.append(gamma_entry)

            prefix_prob = gamma_entry.other
            for t in range(1, T):
                new_label_prob = p_star.gamma[t - 1].blank
                if len(p_star.path) == 0 or p_star.path[-1] != k:
                    new_label_prob = logsumexp(new_label_prob, p_star.gamma[t - 1].other)
                gamma_entry = GammaEntry()
                gamma_entry.other = logprob[t, k]
                exp_sum = logsumexp(new_label_prob, gamma[t - 1].other)
                if exp_sum is not None:
                    gamma_entry.other = gamma_entry.other + exp_sum
                gamma_entry.blank = logprob[t, 0]
                exp_sum = logsumexp(gamma[t - 1].blank, gamma[t - 1].other)
                if exp_sum is not None:
                    gamma_entry.blank = gamma_entry.blank + exp_sum
                gamma.append(gamma_entry)
                exp_mult = logprob[t, k]
                if new_label_prob is not None:
                    exp_mult = exp_mult + exp_mult
                prefix_prob = logsumexp(prefix_prob, exp_mult)

            P_path_full = logsumexp(gamma[T - 1].blank, gamma[T - 1].other)
            P_path_partial = logsubstractexp(prefix_prob, P_path_full)

            if P_path_full > P_l_star:
                l_star = path
                P_l_star = P_path_full

            if P_path_partial > P_l_star:
                beam_entry = BeamEntry()
                beam_entry.path = path
                beam_entry.gamma = gamma
                beam_entry.P_path_full = P_path_full
                beam_entry.P_path_partial = P_path_partial
                beam.append(beam_entry)
            prob_remaining = logsubstractexp(prob_remaining, P_path_partial)
            if prob_remaining <= P_l_star:
                break
    return l_star

def prefix_beam_search_prob(prob):
    num_tags = prob.shape[1] - 1
    T = prob.shape[0]
    beam = []
    gamma = []
    gamma_entry = GammaEntry()
    gamma_entry.blank = prob[0, 0]
    gamma_entry.other = 0.
    gamma.append(gamma_entry)

    for t in range(1, T):
        gamma_entry = GammaEntry()
        gamma_entry.blank = prob[t, 0]*gamma[-1].blank
        gamma_entry.other = 0.
        gamma.append(gamma_entry)

    beam_entry = BeamEntry()
    beam_entry.gamma = gamma
    beam_entry.path = []
    beam_entry.P_path_full = gamma[T - 1].blank
    beam_entry.P_path_partial = 1.0 - beam_entry.P_path_full
    beam.append(beam_entry)

    l_star, P_l_star = beam_entry.path, beam_entry.P_path_full
    while len(beam) > 0:
        beam.sort(key=lambda x: x.P_path_partial)
        p_star = beam[-1]
        beam.pop()

        prob_remaining = p_star.P_path_partial
        if prob_remaining <= P_l_star:  # done
            break

        for k in range(1, num_tags + 1):
            path = p_star.path + [k]

            gamma = []
            gamma_entry = GammaEntry()
            gamma_entry.other = prob[0, k] if p_star.path == [] else 0.
            gamma_entry.blank = 0.
            gamma.append(gamma_entry)

            prefix_prob = gamma_entry.other
            for t in range(1, T):
                new_label_prob = p_star.gamma[t - 1].blank
                if len(p_star.path) == 0 or p_star.path[-1] != k:
                    new_label_prob = new_label_prob + p_star.gamma[t - 1].other
                gamma_entry = GammaEntry()
                gamma_entry.other = prob[t, k]
                exp_sum = new_label_prob + gamma[t - 1].other
                gamma_entry.other = gamma_entry.other * exp_sum
                gamma_entry.blank = prob[t, 0]
                exp_sum = gamma[t - 1].blank + gamma[t - 1].other
                gamma_entry.blank = gamma_entry.blank * exp_sum
                gamma.append(gamma_entry)
                exp_mult = prob[t, k]
                exp_mult = exp_mult + new_label_prob
                prefix_prob = prefix_prob * exp_mult

            P_path_full = gamma[T - 1].blank + gamma[T - 1].other
            P_path_partial = prefix_prob - P_path_full

            if P_path_full > P_l_star:
                l_star = path
                P_l_star = P_path_full

            if P_path_partial > P_l_star:
                beam_entry = BeamEntry()
                beam_entry.path = path
                beam_entry.gamma = gamma
                beam_entry.P_path_full = P_path_full
                beam_entry.P_path_partial = P_path_partial
                beam.append(beam_entry)
            prob_remaining = prob_remaining - P_path_partial
            if prob_remaining <= P_l_star:
                break
        print(prob_remaining, P_l_star, len(beam))
    return l_star

if __name__ == '__main__':
    from scipy.special import softmax
    alphabet = ['B', 'a', 'b', 'c', 'd']

    activation = np.array([[2, 0, 0, 0, 0], [1, 3, 0, 0, 0 ], [ 1, 4, 1, 0, 0 ], [ 1, 1, 5, 6, 0 ],
    [1, 1, 1, 1, 1 ], [ 1, 1, 7, 1, 1 ], [ 9, 1, 1, 1, 1 ]])
    prob = softmax(activation, axis = 1)
    logprob = np.log(prob)
    print([alphabet[p] for p in prefix_beam_search_prob(prob)])
    #dacba
    activation = np.array([[0, 0, 0, 0, 0]])
    prob = softmax(activation, axis=1)
    logprob = np.log(prob)
    alphabet = ['B', 'a', 'b', 'c', 'd']
    print([alphabet[p] for p in prefix_beam_search_prob(prob)])
    #a
    activation = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
    prob = softmax(activation, axis=1)
    logprob = np.log(prob)
    print([alphabet[p] for p in prefix_beam_search_prob(prob)])
    #cc => cBc
    activation = np.array([[-5,-4,-3,-2,-1], [-10,-9,-8,-7,-6], [-15,-14,-13,-12,-11]])
    prob = softmax(activation, axis=1)
    logprob = np.log(prob)
    print([alphabet[p] for p in prefix_beam_search_prob(prob)])
    #bc => Bbc, bBc, bcB
    #max path
    #print([alphabet[k] for k in np.argmax(logprob, axis=1)])
