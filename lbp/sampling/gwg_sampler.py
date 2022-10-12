import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import torch.nn.functional as F


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, log_g, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, step_size=1.0):
        print('gwg sampler')
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.step_acc = 0.0
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.step_size = step_size
        self.log_g = log_g
        if approx:
            self.diff_fn = lambda x, m: approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: difference_function(x, m)

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=self.log_g(delta))
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=self.log_g(forward_delta))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=self.log_g(reverse_delta))

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        self._ar = np.mean(a_s)
        self.step_acc = self._ar
        return x_cur


class GWGNorepSampler(nn.Module):
    def __init__(self, dim, log_g, n_steps=10, n_samples=1, target_acc_rate=0.65):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self.step_acc = 0.0
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.n_samples = n_samples
        self.succ = 0
        self.count = 0
        self.effective_movements = 0
        self.target_acc_rate = target_acc_rate
        self.diff_fn = lambda x, m: approx_difference_function(x, m)
        self.log_g = log_g

#    def adaptive_adjust(self, acc_rate):
#        self.n_samples = max(1, self.n_samples + (acc_rate - self.target_acc_rate))
#        return 'r: %.2f n: %d' % (acc_rate, self.n_samples)
    def adaptive_adjust(self, acc_rate):
        if acc_rate < self.target_acc_rate:
            self.n_samples -= 1
        elif acc_rate > self.target_acc_rate:
            self.n_samples += 1
        self.n_samples = max(1, self.n_samples)
        self.n_samples = min(self.dim, self.n_samples)
        return 'r: %.2f n: %d' % (acc_rate, self.n_samples)

    def step(self, x, model):
        #if np.random.rand(1) < self.n_samples - np.floor(self.n_samples):
        #    R = int(np.floor(self.n_samples))
        #else:
        #    R = int(np.ceil(self.n_samples))
        R = self.n_samples
        bsize = x.shape[0]
        b_idx = torch.arange(bsize).unsqueeze(-1)
        x_rank = len(x.shape) - 1
        forward_delta = self.diff_fn(x, model)

        with torch.no_grad():
            score_x = model(x)
            log_prob_x = F.log_softmax(self.log_g(forward_delta), dim=-1)
            index = torch.multinomial(log_prob_x.exp(), R)
            y = x.clone()
            y[b_idx, index] = 1 - y[b_idx, index]
            score_y = model(y)

        reverse_delta = self.diff_fn(y, model)

        with torch.no_grad():
            log_prob_y = F.log_softmax(self.log_g(reverse_delta), dim=-1)

            tri_l = torch.triu(torch.ones(R, R, device=x.device))
            log_x_selected = log_prob_x[b_idx, index]
            log_x_max = torch.max(log_x_selected, dim=-1, keepdim=True).values
            log_x_u = log_x_max + torch.log(torch.exp(log_x_selected - log_x_max) @ tri_l)
            log_x = (log_x_selected - torch.log1p(-torch.exp(log_x_u))).sum(dim=-1)

            tri_u = torch.triu(torch.ones(R, R, device=x.device))
            log_y_selected = log_prob_y[b_idx, index]
            log_y_max = torch.max(log_y_selected, dim=-1, keepdim=True).values
            log_y_u = log_y_max + torch.log(torch.exp(log_y_selected - log_y_max) @ tri_u)
            log_y = (log_y_selected - torch.log1p(-torch.exp(log_y_u))).sum(dim=-1)
            log_acc = score_y + log_y - score_x - log_x
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
        self.step_acc = accepted.mean().item()
        return new_x


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, log_g, n_steps=10, approx=False, n_samples=1, target_acc_rate=0.65):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self.step_acc = 0.0
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.n_samples = n_samples
        self.succ = 0
        self.count = 0
        self.effective_movements = 0
        self.target_acc_rate = target_acc_rate
        if approx:
            self.diff_fn = lambda x, m: approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: difference_function(x, m)
        self.log_g = log_g

    def adaptive_adjust(self, acc_rate):
        if acc_rate < self.target_acc_rate:
            self.n_samples -= 1
        elif acc_rate > self.target_acc_rate:
            self.n_samples += 1
        self.n_samples = max(1, self.n_samples)
        self.n_samples = min(self.dim, self.n_samples)
        return 'r: %.2f n: %d' % (acc_rate, self.n_samples)

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=self.log_g(forward_delta))
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0).long() % 2).float() # with backtrack

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=self.log_g(reverse_delta))

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            self.succ += a.sum().item()
            self.count += a.shape[0]
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self.step_acc = self._ar
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)
        self.effective_movements += (a * changes.sum(-1)).sum()

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d

# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        print("using categorical gwg sampler")
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur
