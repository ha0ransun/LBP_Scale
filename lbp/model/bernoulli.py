import torch
import random
import numpy as np
import torch.distributions as dists
from torch.nn import Module


class Bernoulli(Module):
    def __init__(self, p=100, alpha=0.1, beta=0.8, seed=0, device=torch.device("cpu")):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.device = device
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # self.prob = torch.rand(p, device=device) * self.beta + self.alpha
        self.prob = torch.randint(0, 2, (p,), device=device) * self.beta + self.alpha
        # self.prob = torch.rand(p, device=device) * 0.3 + 0.1
        self.n_w = torch.log(self.prob) - torch.log(1 - self.prob)

        self.init_dist = dists.Bernoulli(probs=torch.ones((p,)) * .5)
        self.x0 = self.init_dist.sample((1, )).to(device)

    def forward(self, x):
        return torch.sum(x * self.n_w, dim=-1)

    def trace(self, x):
        return (x - self.x0).abs().sum(dim=-1)

    def change(self, x):
        return (1 - 2 * x) * self.n_w