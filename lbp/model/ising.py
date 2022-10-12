import torch
import numpy as np
import networkx as nx
import torch.distributions as dists
import random
from torch import Tensor, DoubleTensor, LongTensor
from torch.nn import Module, Conv2d, Linear, BatchNorm2d, ReLU
from itertools import combinations, product
from torch.types import Tuple
from torch_scatter import scatter_sum
from PAS.common.config import cmd_args
from numba import jit
import math


class Ising(Module):
    def __init__(self, p=100, mu=0.1, sigma=0.2, lamda=0.1, seed=0, device=torch.device("cpu")):
        super().__init__()
        self.p = p
        self.device = device
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_w = (2 * torch.rand(p, p, device=device) - 1) * sigma
        for i in range(p):
            for j in range(p):
                self.n_w[i, j] += self._weight((i, j), p, mu)
        self.e_w_h = - lamda * torch.ones(p, p - 1, device=device)
        self.e_w_v = - lamda * torch.ones(p - 1, p, device=device)
        self.init_dist = dists.Bernoulli(probs=torch.ones((p ** 2,)) * .5)
        self.x0 = self.init_dist.sample((1,)).to(self.device)

    def _weight(self, n, p, mu):
        if (n[0] / p - 0.5) ** 2 + (n[1] / p - 0.5) ** 2 < 0.5 / np.pi:
            return mu
        else:
            return - mu

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.p, self.p)
        message = self.aggr(x)
        message = message / 2 + self.n_w
        return - ((2 * x - 1) * message).sum(dim=[1, 2]).view(shape[:-1])

    def trace(self, x):
        return (x - self.x0).abs().sum(dim=-1)

    def change(self, x):
        shape = x.shape
        x = x.view(-1, self.p, self.p)
        message = self.aggr(x)
        message += self.n_w
        return - ((2 - 4 * x) * message).view(shape)

    def aggr(self, x):
        message = torch.zeros_like(x)
        message[:, :-1, :] += (2 * x[:, 1:, :] - 1) * self.e_w_v
        message[:, 1:, :] += (2 * x[:, :-1, :] - 1) * self.e_w_v
        message[:, :, :-1] += (2 * x[:, :, 1:] - 1) * self.e_w_h
        message[:, :, 1:] += (2 * x[:, :, :-1] - 1) * self.e_w_h
        return message