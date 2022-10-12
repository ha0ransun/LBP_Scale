import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import torch.nn.functional as F
import igraph as ig
from tqdm import tqdm



class BernoulliRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, data_mean=None):
        super().__init__()
        linear = nn.Linear(n_visible, n_hidden)
        self.W = nn.Parameter(linear.weight.data)
        self.b_h = nn.Parameter(torch.zeros(n_hidden,))
        self.b_v = nn.Parameter(torch.zeros(n_visible,))
        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((n_visible,)) * .5)
        self.data_dim = n_visible

    def p_v_given_h(self, h):
        logits = h @ self.W + self.b_v[None]
        return dists.Bernoulli(logits=logits)

    def p_h_given_v(self, v):
        logits = v @ self.W.t() + self.b_h[None]
        return dists.Bernoulli(logits=logits)

    def logp_v_unnorm(self, v):
        sp = torch.nn.Softplus()(v @ self.W.t() + self.b_h[None]).sum(-1)
        vt = (v * self.b_v[None]).sum(-1)
        return sp + vt

    def logp_v_unnorm_beta(self, v, beta):
        if len(beta.size()) > 0:
            beta = beta[:, None]
        vW = v @ self.W.t() * beta
        sp = torch.nn.Softplus()(vW + self.b_h[None]).sum(-1) - torch.nn.Softplus()(self.b_h[None]).sum(-1)
        #vt = (v * self.b_v[None]).sum(-1)
        ref_dist = torch.distributions.Bernoulli(logits=self.b_v)
        vt = ref_dist.log_prob(v).sum(-1)
        return sp + vt

    def forward(self, x):
        return self.logp_v_unnorm(x)

    def _gibbs_step(self, v):
        h = self.p_h_given_v(v).sample()
        v = self.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, n_steps=2000, n_samples=None, plot=False):
        if v is None:
            assert n_samples is not None
            v = self.init_dist.sample((n_samples,)).to(self.W.device)
        if plot:
           for i in tqdm(range(n_steps)):
               v = self._gibbs_step(v)
        else:
            for i in range(n_steps):
                v = self._gibbs_step(v)
        return v

    def trace(self, x):
        return x.sum(-1)

    def change(self, x):
        with torch.no_grad():
            weight = x @ self.W.T + self.b_h
            weight = torch.exp(weight - torch.logaddexp(weight, torch.zeros_like(weight)))
            return (weight @ self.W + self.b_v) * (1 - 2 * x)

