import torch
import torch.nn as nn
import torch.distributions as dists

class FHMM(nn.Module):
    def __init__(self, L=100, K=10, sigma=0.5, alpha=0.1, beta=0.8, seed=0, device=torch.device("cpu")) -> None:
        super().__init__()
        self.L = L
        self.K = K
        self.sigma = sigma
        self.device = device
        self.alpha = torch.FloatTensor([alpha]).to(device)
        self.beta = torch.FloatTensor([beta]).to(device)
        self.W = torch.randn((K, 1)).to(device)
        self.b = torch.randn((1, 1)).to(device)
        self.X = self.sample_X(seed)
        self.Y = self.sample_Y(self.X, seed)
        self.info = f"fhmm_L-{L}_K-{K}"
        self.init_dist = dists.Bernoulli(probs=torch.ones((L * K,)) * .5)
        self.P_X0 = torch.distributions.Bernoulli(probs=self.alpha)
        self.P_XC = torch.distributions.Bernoulli(logits=1 - self.beta)
        self.x0 = self.init_dist.sample((1,)).to(device)

    def sample_X(self, seed):
        torch.manual_seed(seed)
        X = torch.ones((self.L, self.K)).to(self.device)
        X[0] = torch.bernoulli(X[0] * self.alpha)
        for l in range(1, self.L):
            p = self.beta * X[l - 1] + (1 - self.beta) * (1 - X[l - 1])
            X[l] = torch.bernoulli(p)
        return X

    def sample_Y(self, X, seed):
        torch.manual_seed(seed)
        return torch.randn((self.L, 1)).to(self.device) * self.sigma + X @ self.W + self.b

    def forward(self, x):
        batch = x.shape[:-1]
        x = x.view(-1, self.L, self.K)
        x_0 = x[:, 0, :]
        x_cur = x[:, :-1, :]
        x_next = x[:, 1:, :]
        x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next
        logp_x = - self.P_X0.log_prob(x_0).sum(-1) - self.P_XC.log_prob(x_c).sum(dim=[1, 2])
        logp_y = - (self.Y - x @ self.W - self.b).square().sum(dim=[1,2]) / (2 * self.sigma ** 2)
        return (logp_x + logp_y).view(*batch)

    def error(self, x):
        x = x.view(-1, self.L, self.K)
        return (self.Y - x @ self.W - self.b).square().sum(dim=[1,2]) / (2 * self.sigma ** 2)

    def trace(self, x):
        return (x - self.x0).abs().sum(dim=1)

    def change(self, x):
        batch = x.shape[:-1]
        x = x.view(-1, self.L, self.K)
        x_0 = x[:, 0, :]
        x_cur = x[:, :-1, :]
        x_next = x[:, 1:, :]
        x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next

        change_x = torch.zeros_like(x)
        change_y = torch.zeros_like(x)
        P0 = self.P_X0.log_prob(x_0)
        PC = self.P_XC.log_prob(x_c)
        change_x[:, 0, :] += P0 - torch.log1p(-P0.exp())
        change_x[:, :-1, :] += PC - torch.log1p(-PC.exp())
        change_x[:, 1:, :] += PC - torch.log1p(-PC.exp())
        
        Y = self.Y - x @ self.W - self.b
        Y_change = - (1 - 2 * x) * self.W.T
        change_y = - (Y + Y_change).square() + Y.square()

        change = (change_x + change_y / (2 * self.sigma ** 2) ).view(*batch, self.L * self.K)
        return change

