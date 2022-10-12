#!/bin/bash

nohup python main.py --n_steps 10000 --p 20 --mu 0.1 --sigma 0.3 --lamda 0.1 --gpu 0 --ess_ratio 0.5 > v0.out &
nohup python main.py --n_steps 10000 --p 20 --mu 0.15 --sigma 0.45 --lamda 0.15 --gpu 0 --ess_ratio 0.5 > v1.out &
nohup python main.py --n_steps 10000 --p 20 --mu 0.2 --sigma 0.6 --lamda 0.2 --gpu 0 --ess_ratio 0.5 > v2.out &
nohup python main.py --n_steps 40000 --p 50 --mu 0.1 --sigma 0.3 --lamda 0.1 --gpu 1 --ess_ratio 0.5 > v3.out &
nohup python main.py --n_steps 40000 --p 50 --mu 0.15 --sigma 0.45 --lamda 0.15 --gpu 2 --ess_ratio 0.5 > v4.out &
nohup python main.py --n_steps 40000 --p 50 --mu 0.2 --sigma 0.6 --lamda 0.2 --gpu 3 --ess_ratio 0.5 > v5.out &
nohup python main.py --n_steps 100000 --p 100 --mu 0.1 --sigma 0.3 --lamda 0.1 --gpu 4 --ess_ratio 0.5 > v6.out &
nohup python main.py --n_steps 100000 --p 100 --mu 0.15 --sigma 0.45 --lamda 0.15 --gpu 5 --ess_ratio 0.5 > v7.out &
nohup python main.py --n_steps 100000 --p 100 --mu 0.2 --sigma 0.6 --lamda 0.2 --gpu 6 --ess_ratio 0.5 > v8.out &