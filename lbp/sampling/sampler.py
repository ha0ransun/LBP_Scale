from random import sample
import numpy as np
import torch
from PAS.sampling import gwg_sampler, pas_sampler, block_sampler, gibbs_sampler, adaptive_sampler



def get_sampler(args, sampler=None, data_dim=None, device=torch.device('cpu'), log_g=None):
    sampler = sampler or args.sampler
    data_dim = data_dim or np.prod(args.input_size)
    print(100 * '#')
    print(sampler)
    if log_g is None:
        if args.g_func == 'sqrt2':
            print('g(t) = sqrt(t)')
            log_g = lambda x: x / 2.0
        elif args.g_func == 'tdtp1':
            print('g(t) = t / (t + 1)')
            log_g = lambda x: x - torch.logaddexp(x, torch.zeros_like(x))
        else:
            raise ValueError("unknown g func %s" % args.g_func)
    if args.input_type == "binary":
        if sampler == "gwg":
            sampler = gwg_sampler.DiffSampler(data_dim, log_g, n_steps=1,
                                              fixed_proposal=False, approx=True, multi_hop=False)
        elif sampler.startswith("rw-"):
            radius = int(sampler.split('-')[1])
            sampler = adaptive_sampler.RandomWalkSampler(args=args, U=radius)
        elif sampler.startswith("arw-"):
            args.target_rate = float(sampler.split('-')[1])
            sampler = adaptive_sampler.RandomWalkSampler(args=args, adaptive=1)
        elif sampler.startswith("lbp-"):
            radius = int(sampler.split('-')[1])
            sampler = adaptive_sampler.LocallyBalancedProposalSampler(args=args, U=radius, log_g=log_g)
        elif sampler.startswith("albp-"):
            args.target_rate = float(sampler.split('-')[1])
            sampler = adaptive_sampler.LocallyBalancedProposalSampler(args=args, adaptive=1, log_g=log_g)
        elif sampler.startswith("GWG-"):
            radius = int(sampler.split('-')[1])
            sampler = adaptive_sampler.GibbsWithGradientSampler(args=args, U=radius, log_g=log_g)
        elif sampler.startswith("aGWG-"):
            args.target_rate = float(sampler.split('-')[1])
            sampler = adaptive_sampler.GibbsWithGradientSampler(args=args, adaptive=1, log_g=log_g)
        elif sampler.startswith("gwg-"):
            n_hops = int(sampler.split('-')[1])
            if args.adaptive_sampling:
                target_acc = float(sampler.split('-')[2])
            else:
                target_acc = None
            sampler = gwg_sampler.MultiDiffSampler(data_dim, log_g, n_steps=1, approx=True, n_samples=n_hops, target_acc_rate=target_acc)
        elif sampler.startswith('nrgwg-'):
            n_hops = int(sampler.split('-')[1])
            if args.adaptive_sampling:
                target_acc = float(sampler.split('-')[2])
            else:
                target_acc = None
            sampler = gwg_sampler.GWGNorepSampler(data_dim, log_g, n_steps=1, n_samples=n_hops, target_acc_rate=target_acc)
        elif sampler.startswith("lbp-"):
            radius = int(sampler.split('-')[1])
            sampler = pas_sampler.PathAuxiliarySampler(radius, args=args)
        elif sampler.startswith("pafs-"):
            radius = int(sampler.split('-')[1])
            sampler = pas_sampler.PathCorrectSampler(radius, args=args)
        elif sampler == 'dim-gibbs':
            sampler = gibbs_sampler.PerDimGibbsSampler(data_dim)
        elif sampler == "rand-gibbs":
            sampler = gibbs_sampler.PerDimGibbsSampler(data_dim, rand=True)
        elif "bg-" in sampler:
            block_size = int(sampler.split('-')[1])
            sampler = block_sampler.BlockGibbsSampler(data_dim, block_size)
        elif "hb-" in sampler:
            block_size, hamming_dist = [int(v) for v in sampler.split('-')[1:]]
            sampler = block_sampler.HammingBallSampler(data_dim, block_size, hamming_dist)
        else:
            raise ValueError("Invalid sampler...")
    else:
        if sampler == "gwg":
            sampler = gwg_sampler.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
        elif sampler.startswith("mscorrect-"):
            radius = int(sampler.split('-')[1])
            sampler = pas_sampler.PathCatSampler(radius)
        else:
            raise ValueError("Invalid sampler...")        
    return sampler
