import torch
import numpy as np
import torch_scatter


class BaseSampler():
    def __init__(self, args, U=1, adaptive=0, log_g=None, device=torch.device("cpu")):
        self.U = U
        self.ess_ratio = args.ess_ratio
        self.adaptive = adaptive
        self.log_g = log_g
        if hasattr(args, "target_rate"):
            self.target_rate = args.target_rate
        else:
            self.target_rate = 0.574
        self.device = device
        self._steps = 0
        self._lens = []
        self._accs = []
        self._hops = []

    def step(self, x, model):
        raise NotImplementedError

    @property
    def accs(self):
        return self._accs[-1]

    @property
    def hops(self):
        return self._hops[-1]

    @property
    def lens(self):
        return self._lens[-1]

    @property
    def avg_lens(self):
        ratio = self.ess_ratio
        return sum(self._lens[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_accs(self):
        ratio = self.ess_ratio
        return sum(self._accs[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_hops(self):
        ratio = self.ess_ratio
        return sum(self._hops[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)


class RandomWalkSampler(BaseSampler):
    def __init__(self, args, U=1, adaptive=0, log_g=None, device=torch.device("cpu")):
        super().__init__(args, U, adaptive, log_g, device)

    def step(self, x, model):
        if np.random.rand(1) < self.U - np.floor(self.U):
            R = int(np.floor(self.U))
        else:
            R = int(np.ceil(self.U))

        x_rank = len(x.shape) - 1
        bsize = x.shape[0]
        b_idx = torch.arange(bsize).unsqueeze(-1).to(x.device)
        index = torch.multinomial(torch.ones_like(x), R, replacement=False)
        with torch.no_grad():
            score_x = model(x)
            y = x.clone()
            y[b_idx, index] = 1 - y[b_idx, index]
            score_y = model(y)
            log_acc = score_y - score_x
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()
        if self.adaptive:
            self.U = min([max([1, self.U + (accs - self.target_rate)]), x.shape[-1]])

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x


class LocallyBalancedProposalSampler(BaseSampler):
    def __init__(self, args, U=1, adaptive=0, log_g=None, device=torch.device("cpu")):
        super().__init__(args, U, adaptive, log_g, device)
        self.dist = None

    def step(self, x, model):
        bsize = x.shape[0]
        b_idx = torch.arange(bsize).unsqueeze(-1)
        x_rank = len(x.shape) - 1

        if np.random.rand(1) < self.U - np.floor(self.U):
            R = int(np.floor(self.U))
        else:
            R = int(np.ceil(self.U))

        with torch.no_grad():
            score_x = model(x)
            score_change_x = self.log_g(model.change(x))
            log_prob_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
            index = torch.multinomial(log_prob_x.exp(), R)
            y = x.clone()
            y[b_idx, index] = 1 - y[b_idx, index]
            score_y = model(y)
            score_change_y = self.log_g(model.change(y))
            log_prob_y = score_change_y - torch.logsumexp(score_change_y, dim=-1, keepdim=True)

            tri_u = torch.triu(torch.ones(R, R, device=x.device), 1)
            log_x_selected = log_prob_x[b_idx, index]
            log_x_max = torch.max(log_x_selected, dim=-1, keepdim=True).values
            log_x_u = log_x_max + torch.log(torch.exp(log_x_selected - log_x_max) @ tri_u)
            log_x = (log_x_selected - torch.log1p(-torch.exp(log_x_u))).sum(dim=-1)

            tri_l = torch.tril(torch.ones(R, R, device=x.device), -1)
            log_y_selected = log_prob_y[b_idx, index]
            log_y_max = torch.max(log_y_selected, dim=-1, keepdim=True).values
            log_y_l = log_y_max + torch.log(torch.exp(log_y_selected - log_y_max) @ tri_l)
            log_y = (log_y_selected - torch.log1p(-torch.exp(log_y_l))).sum(dim=-1)

            log_acc = score_y + log_y - score_x - log_x
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()
        if self.adaptive:
            self.U = min([max([1, self.U + (accs - self.target_rate)]), x.shape[-1]])

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x



class GibbsWithGradientSampler(BaseSampler):
    def __init__(self, args, U=1, adaptive=0, log_g=None, device=torch.device("cpu")):
        super().__init__(args, U, adaptive, log_g, device)

    def step(self, x, model):
        if np.random.rand(1) < self.U - np.floor(self.U):
            R = int(np.floor(self.U))
        else:
            R = int(np.ceil(self.U))

        bsize = x.shape[0]
        b_idx = torch.arange(bsize).unsqueeze(-1)
        x_rank = len(x.shape) - 1

        with torch.no_grad():
            score_x = model(x)
            score_change_x = self.log_g(model.change(x))
            log_prob_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
            index = torch.multinomial(log_prob_x.exp(), R, replacement=True)
            y = x.clone()
            y[b_idx, index] = 1 - y[b_idx, index]
            score_y = model(y)
            score_change_y = self.log_g(model.change(y))
            log_prob_y = score_change_y - torch.logsumexp(score_change_y, dim=-1, keepdim=True)

            log_x = log_prob_x[b_idx, index].sum(dim=-1)
            log_y = log_prob_y[b_idx, index].sum(dim=-1)

            log_acc = score_y + log_y - score_x - log_x
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()
        if self.adaptive:
            self.U = min([max([1, self.U + (accs - self.target_rate)]), x.shape[-1]])

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x



class PAFSSampler(BaseSampler):
    def __init__(self, args, U=1, adaptive=0, device=torch.device("cpu")):
        super().__init__(args, U, adaptive, device)

    def step(self, x, model):
        if np.random.rand(1) < self.U - np.floor(self.U):
            R = int(np.floor(self.U))
        else:
            R = int(np.ceil(self.U))

        bsize = x.shape[0]
        x_rank = len(x.shape) - 1

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        delta_list = []
        with torch.no_grad():
            cur_x = x.clone()
            idx_list = []
            for step in range(R):
                delta_x = -(2.0 * cur_x - 1.0)
                delta_list.append(delta_x)
                score_change_x = delta_x * grad_x
                score_change_x = score_change_x - torch.logaddexp(score_change_x, torch.zeros_like(score_change_x))
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list.append(index.view(-1, 1))
                cur_x[b_idx, index] = 1.0 - cur_x[b_idx, index]
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(R).to(x.device).view(1, -1)
            b_idx = b_idx.view(-1, 1)
            idx_list = torch.cat(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            traj = torch.stack(delta_list, dim=1)  # bsize x max_r x dim
            score_fwd = traj * grad_x.unsqueeze(1)
            score_fwd = score_fwd - torch.logaddexp(score_fwd, torch.zeros_like(score_fwd))
            log_fwd = torch.log_softmax(score_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx, r_idx, idx_list], dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            delta_list.append(delta_y)
            traj = torch.stack(delta_list[1:], dim=1)  # bsize x max_r x dim
            score_backwd = traj * grad_y.unsqueeze(1)
            score_backwd = score_backwd - torch.logaddexp(score_backwd, torch.zeros_like(score_backwd))
            log_backwd = torch.log_softmax(score_backwd, dim=-1)
            log_backwd = torch.sum(log_backwd[b_idx, r_idx, idx_list], dim=-1) + score_y.view(-1)

            log_acc = log_backwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()
        if self.adaptive:
            self.U = min([max([1, self.U + (accs - self.target_rate)]), x.shape[-1]])

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x



class PASSampler(BaseSampler):
    def __init__(self, args, U=1, adaptive=0, device=torch.device("cpu")):
        super().__init__(args, U, adaptive, device)

    def step(self, x, model):
        if np.random.rand(1) < self.U - np.floor(self.U):
            R = int(np.floor(self.U))
        else:
            R = int(np.ceil(self.U))

        bsize = x.shape[0]
        x_rank = len(x.shape) - 1

        x = x.requires_grad_()

        Zx, Zy = 1., 1.
        b_idx = torch.arange(bsize).to(x.device)
        cur_x = x.clone()
        with torch.no_grad():
            for step in range(R):
                score_change_x = model.change(cur_x)
                score_change_x = score_change_x - torch.logaddexp(score_change_x, torch.zeros_like(score_change_x))
                if step == 0:
                    Zx = torch.logsumexp(score_change_x, dim=1)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                cur_x[b_idx, index] = 1 - cur_x[b_idx, index]
            y = cur_x

        score_change_y = model.change(y)
        score_change_y = score_change_y - torch.logaddexp(score_change_y, torch.zeros_like(score_change_y))
        Zy = torch.logsumexp(score_change_y, dim=1)

        log_acc = Zx - Zy
        accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
        new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()
        if self.adaptive:
            self.U = max([1, self.U + (accs - self.target_rate)])

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x


# class LocallBalancedBernoulliSampler(BaseSampler):
#     def __init__(self, args, U=1, adaptive=0, device=torch.device("cpu")):
#         super().__init__(args, U, adaptive, device)
#         self.dist = None
#
#     def step(self, x, model):
#         bsize = x.shape[0]
#         x_rank = len(x.shape) - 1
#
#         with torch.no_grad():
#             score_x = model(x)
#             score_change_x = model.change(x)
#             score_change_x = score_change_x - torch.logaddexp(score_change_x, torch.zeros_like(score_change_x))
#             log_prob_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
#             log_prob_x = torch.clamp(torch.log1p(-torch.exp(self.U * torch.log1p(-log_prob_x.exp()))), min=-20)
#             m_x = dists.Bernoulli(logits=log_prob_x)
#             index = m_x.sample()
#             y = x.clone()
#             y = index * (1 - y) + (1 - index) * y
#             score_y = model(y)
#             score_change_y = model.change(y)
#             score_change_y = score_change_y - torch.logaddexp(score_change_y, torch.zeros_like(score_change_y))
#             log_prob_y = score_change_y - torch.logsumexp(score_change_y, dim=-1, keepdim=True)
#             log_prob_y = torch.clamp(torch.log1p(-torch.exp(self.U * torch.log1p(-log_prob_y.exp()))), min=-20)
#             m_y = dists.Bernoulli(logits=log_prob_y)
#
#             log_acc = score_y + m_y.log_prob(index).sum(dim=-1) - score_x - m_x.log_prob(index).sum(dim=-1)
#             accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
#             new_x = y * accepted + (1.0 - accepted) * x
#
#         accs = torch.clamp(log_acc.exp(), max=1).mean().item()
#         if self.adaptive:
#             self.U += (accs - 0.574)
#
#         self._steps += 1
#         self._lens.append(index.sum().item() / bsize)
#         self._accs.append(accs)
#         self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
#         return new_x
#
#
# class BinomialWalkSampler(BaseSampler):
#     def __init__(self, args, U=1, adaptive=0, device=torch.device("cpu")):
#         super().__init__(args, U, adaptive, device)
#         self.dist = None
#
#     def step(self, x, model):
#         bsize = x.shape[0]
#         x_rank = len(x.shape) - 1
#
#         with torch.no_grad():
#             score_x = model(x)
#             score_change_x = model.change(x) / 2.0
#             log_prob_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
#             log_prob_x = torch.clamp(torch.log1p(-torch.exp(self.U * torch.log1p(-log_prob_x.exp()))), min=-20)
#             m_x = dists.Bernoulli(logits=log_prob_x)
#             index = m_x.sample()
#             y = x.clone()
#             y = index * (1 - y) + (1 - index) * y
#             score_y = model(y)
#             score_change_y = model.change(y) / 2.0
#             log_prob_y = score_change_y - torch.logsumexp(score_change_y, dim=-1, keepdim=True)
#             log_prob_y = torch.clamp(torch.log1p(-torch.exp(self.U * torch.log1p(-log_prob_y.exp()))), min=-20)
#             m_y = dists.Bernoulli(logits=log_prob_y)
#
#             log_acc = score_y + m_y.log_prob(index).sum(dim=-1) - score_x - m_x.log_prob(index).sum(dim=-1)
#             accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
#             new_x = y * accepted + (1.0 - accepted) * x
#
#         accs = accepted.sum().item() / bsize
#         if self.adaptive:
#             if accs > 0.574:
#                 if torch.rand(1) < accs - 0.574:
#                     self.U += 1
#             else:
#                 if torch.rand(1) < 0.574 - accs:
#                     self.U = max([1, self.U - 1])
#
#         self._steps += 1
#         self._lens.append(index.sum().item() / bsize)
#         self._accs.append(accs)
#         self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
#         return new_x