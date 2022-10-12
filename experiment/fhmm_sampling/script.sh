#!/bin/bash

nohup python main.py --n_steps 10000 --L 1000 --K 5 --alpha 0.1 --beta 0.8 --sigma 1.4142 --gpu 0 --ess_ratio 0.5 > v0.out &
nohup python main.py --n_steps 10000 --L 1000 --K 5 --alpha 0.1 --beta 0.8 --sigma 1.0 --gpu 0 --ess_ratio 0.5 > v1.out &
nohup python main.py --n_steps 10000 --L 1000 --K 5 --alpha 0.1 --beta 0.8 --sigma 0.7071 --gpu 0 --ess_ratio 0.5 > v2.out &
nohup python main.py --n_steps 20000 --L 1000 --K 10 --alpha 0.1 --beta 0.8 --sigma 1.4142 --gpu 1 --ess_ratio 0.5 > v3.out &
nohup python main.py --n_steps 20000 --L 1000 --K 10 --alpha 0.1 --beta 0.8 --sigma 1.0 --gpu 2 --ess_ratio 0.5 > v4.out &
nohup python main.py --n_steps 20000 --L 1000 --K 10  --alpha 0.1 --beta 0.8 --sigma 0.7071 --gpu 3 --ess_ratio 0.5 > v5.out &
nohup python main.py --n_steps 40000 --L 1000 --K 20 --alpha 0.1 --beta 0.8 --sigma 1.4142 --gpu 4 --ess_ratio 0.5 > v6.out &
nohup python main.py --n_steps 40000 --L 1000 --K 20 --alpha 0.1 --beta 0.8 --sigma 1.0 --gpu 5 --ess_ratio 0.5 > v7.out &
nohup python main.py --n_steps 40000 --L 1000 --K 20 --alpha 0.1 --beta 0.8 --sigma 0.7071 --gpu 6 --ess_ratio 0.5 > v8.out &