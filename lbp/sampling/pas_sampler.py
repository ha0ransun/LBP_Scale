import torch
from torch._C import _set_backcompat_keepdim_warn
from torch.nn import Module
import torch.distributions as dists
import numpy as np
from numba import njit

class PathAuxiliarySampler(Module):
    def __init__(self, R, args):
        super().__init__()
        self.R = R
        self.ess_ratio = args.ess_ratio
        self._steps = 0
        self._evals = []
        self._hops = []

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        Zx, Zy = 1., 1.
        b_idx = torch.arange(bsize).to(x.device)
        cur_x = x.clone()
        with torch.no_grad():
            for step in range(max_r):
                score_change_x = model.change(cur_x) / 2.0
                if step == 0:
                    Zx = torch.logsumexp(score_change_x, dim=1)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
            y = cur_x

        score_change_y = model.change(y) / 2.0
        Zy = torch.logsumexp(score_change_y, dim=1)
        
        log_acc = Zx - Zy
        accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
        new_x = y * accepted + (1.0 - accepted) * x
        self._steps += 1
        self._evals.append((0 + radius.sum()).item() / bsize)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x

    @property
    def evals(self):
        return self._evals[-1]

    @property
    def hops(self):
        return self._hops[-1]

    @property
    def avg_evals(self):
        ratio = self.ess_ratio
        return sum(self._evals[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_hops(self):
        ratio = self.ess_ratio
        return sum(self._hops[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def weight(self):
        return torch.ones(2 * self.R - 1) / (2 * self.R - 1)



class PathCorrectSampler(Module):
    def __init__(self, R, args):
        print('our binary sampler')
        super(PathCorrectSampler, self).__init__()
        self._hops = R
        self.R = R
        self.ess_ratio = args.ess_ratio
        self._steps = 0
        self._lens = []
        self._accs = []
        self._hops = []
        
    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        delta_list = []
        with torch.no_grad():
            log_fwd = score_x.view(-1)
            cur_x = x.clone()
            idx_list = []
            for step in range(max_r):
                delta_x = -(2.0 * cur_x - 1.0)
                delta_list.append(delta_x)
                score_change_x = delta_x * grad_x / 2.0
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list.append(index.view(-1, 1))
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)
            b_idx = b_idx.view(-1, 1)
            idx_list = torch.cat(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            traj = torch.stack(delta_list, dim=1) # bsize x max_r x dim
            log_fwd = torch.log_softmax(traj * grad_x.unsqueeze(1) / 2.0, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx, r_idx, idx_list] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            delta_list.append(delta_y)
            traj = torch.stack(delta_list[1:], dim=1) # bsize x max_r x dim
            log_backwd = torch.log_softmax(traj * grad_y.unsqueeze(1) / 2.0, dim=-1)
            log_backwd = torch.sum(log_backwd[b_idx, r_idx, idx_list] * r_mask, dim=-1) + score_y.view(-1)

            log_acc = log_backwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x

        self._steps += 1
        accs = accepted.sum().item() / bsize
        self._lens.append(radius.sum().item() / bsize)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x

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


class NBSampler(Module):
    def __init__(self, R):
        print('our binary sampler')
        super().__init__()
        self.R_list = []
        self.R = R


    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        self.R_list.append(radius)
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device).unsqueeze(-1).repeat(1, max_r)
        idx_list = []
        with torch.no_grad():
            cur_x = x.clone()
            delta_x = -(2.0 * cur_x - 1.0)
            score_change_x = delta_x * grad_x / 2.0
            prob_x = torch.softmax(score_change_x, dim=-1)
            for t in range(max_r):
                index = torch.multinomial(prob_x, 1).view(-1)
                cur_bits = cur_x[b_idx[:, t], index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, t]
                cur_x[b_idx[:, t], index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
                prob_x[b_idx[:, t], index] = 0
                idx_list.append(index)
            index = torch.stack(idx_list, dim=1)
            y = cur_x

        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)

            # fwd from x -> y
            change_fwd = score_change_x.unsqueeze(1).repeat(1, max_r, 1)
            for i in range(max_r):
                change_fwd[b_idx[:, i], i+1:, index[:, i]] = -float('inf')
            log_fwd = torch.log_softmax(change_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx, r_idx, index] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            score_change_y = delta_y * grad_y / 2.0
            change_bwd = score_change_y.unsqueeze(1).repeat(1, max_r, 1)
            for i in range(max_r):
                change_bwd[b_idx[:, i], :i, index[:, i]] = -float('inf')
            log_bwd = torch.log_softmax(change_bwd, dim=-1)
            log_bwd = torch.sum(log_bwd[b_idx, r_idx, index] * r_mask, dim=-1) + score_y.view(-1)

            log_acc = log_bwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            return new_x