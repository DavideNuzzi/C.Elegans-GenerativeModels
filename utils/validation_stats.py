import torch, torch.nn.functional as F, numpy as np, scipy.stats as st

# x_real, x_gen: (T, C) tensors on CPU
def ks_marginals(x_real, x_gen):
    return [st.ks_2samp(x_real[:, i], x_gen[:, i]).statistic for i in range(x_real.shape[1])]

def acf_gap(xr, xg, max_lag=100):
    def acf(x, L):                         # quick, unbiased ACF
        x = x - x.mean(0, keepdim=True)
        return torch.stack([ (x[:-l] * x[l:]).mean(0) / x.var(0) for l in range(L+1) ])
    return (acf(xr, max_lag) - acf(xg, max_lag)).abs().mean(0)   # per-channel L1
