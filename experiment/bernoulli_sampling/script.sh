#!/bin/bash

nohup python main.py --n_steps 4000 --p 100 --alpha 0.25 --beta 0.5 --gpu 0 --ess_ratio 0.5 > v0.out &
nohup python main.py --n_steps 4000 --p 100 --alpha 0.15 --beta 0.7 --gpu 0 --ess_ratio 0.5 > v1.out &
nohup python main.py --n_steps 4000 --p 100 --alpha 0.05 --beta 0.9 --gpu 0 --ess_ratio 0.5 > v2.out &
nohup python main.py --n_steps 16000 --p 800 --alpha 0.25 --beta 0.5 --gpu 1 --ess_ratio 0.5 > v3.out &
nohup python main.py --n_steps 16000 --p 800 --alpha 0.15 --beta 0.7 --gpu 2 --ess_ratio 0.5 > v4.out &
nohup python main.py --n_steps 16000 --p 800 --alpha 0.05 --beta 0.9 --gpu 3 --ess_ratio 0.5 > v5.out &
nohup python main.py --n_steps 64000 --p 6400 --alpha 0.25 --beta 0.5 --gpu 4 --ess_ratio 0.5 > v6.out &
nohup python main.py --n_steps 64000 --p 6400 --alpha 0.15 --beta 0.7 --gpu 5 --ess_ratio 0.5 > v7.out &
nohup python main.py --n_steps 64000 --p 6400 --alpha 0.05 --beta 0.9 --gpu 6 --ess_ratio 0.5 > v8.out &