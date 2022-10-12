import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import torch.nn.functional as F
import igraph as ig
from tqdm import tqdm


class MyOneHotCategorical:
    def __init__(self, mean):
        self.dist = torch.distributions.OneHotCategorical(probs=mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)

class EBM(nn.Module):
    def __init__(self, net, mean=None, is_binary=True):
        super().__init__()
        self.net = net
        self.is_binary = is_binary
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            if self.is_binary:
                base_dist = torch.distributions.Bernoulli(probs=self.mean)
            else:
                base_dist = MyOneHotCategorical(self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd

