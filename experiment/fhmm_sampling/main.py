import numpy as np
import torch
import random
from lbp.common.config import cmd_args
from lbp.sampling.sampler import get_sampler
from lbp.model.fhmm import FHMM
from matplotlib import cm
import matplotlib.pyplot as plt
import os, pickle
import time
import tensorflow_probability as tfp
from tqdm import tqdm
from copy import deepcopy

def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def main(args):
    def run(temp):
        sampler = get_sampler(args, sampler=temp, data_dim=args.n_visible, device=device)
        x = model.init_dist.sample((args.n_test_samples,)).to(device)
        times, energys, traces = [], [], []
        cur_time = 0.
        progress_bar = tqdm(range(args.n_steps))
        for i in progress_bar:
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            x = xhat

            with torch.no_grad():
                energy = - model(x)
                trace = model.trace(x)
            # energys.append(energy.cpu().numpy())
            traces.append(trace.cpu().numpy())

            if i % args.print_every == 0:
                progress_bar.set_description(
                    "temp {}, itr = {}, energy = {:.4f}, lens = {:.4f}, accs = {:.4f}, "
                    "hops = {:.4f}".format(temp, i, energy.mean().item(), sampler.lens, sampler.accs, sampler.hops))

        # energy = np.stack(energys, 0)
        trace = np.stack(traces, 0)
        ess = get_ess(trace, 1 - args.ess_ratio)
        trace = []
        energy = []
        overall_time = cur_time
        stats[temp] = {'energy': energy, 'trace': trace, 'ess': ess, 'time': overall_time, 'len': sampler.avg_lens,
                       'acc': sampler.avg_accs, 'hop': sampler.avg_hops}
        print(f"{temp}: \t ess = {ess.mean():.4f} +/- {ess.std():.4f}, avg_lens = {sampler.avg_lens:.4f}, "
              f"avg_accs = {sampler.avg_accs:.4f}, avg_hops = {sampler.avg_hops:.4f}, time = {overall_time:.4f}")
        return sampler.avg_accs

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_visible = 10000
    args.input_type = 'binary'

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")



    model = FHMM(L=args.L, K=args.K, alpha=args.alpha, beta=args.beta, sigma=args.sigma, device=device)
    args.g_func = "sqrt2"
    methods1 = ['rw', 'GWG', 'lbp']
    stats = {}
    for method in methods1:
        acc = run(method + "-1")
        while acc > 0.03:
            acc -= 0.02
            run("a" + method + f"-{acc}")
        if method == 'rw':
            run("a" + method + f"-0.234")
        else:
            run("a" + method + f"-0.574")

    res = {}
    res["sqrt2"] = deepcopy(stats)

    args.g_func = "tdtp1"
    methods1 = ['GWG', 'lbp']
    stats = {}

    for method in methods1:
        acc = run(method + "-1")
        while acc > 0.03:
            acc -= 0.02
            run("a" + method + f"-{acc}")
        if method == 'rw':
            run("a" + method + f"-0.234")
        else:
            run("a" + method + f"-0.574")

    res["tdtp1"] = stats

    if not os.path.isdir('results'):
        os.mkdir('results')
    with open(f'results/L-{args.L}_sigma-{args.sigma}.pkl', 'wb') as handle:
        pickle.dump(res, handle)

if __name__ == '__main__':
    main(cmd_args)



        