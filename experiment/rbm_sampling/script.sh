#!/bin/bash

# python main.py --n_hidden 100 --gpu 5 --seed 0
# python main.py --n_hidden 400 --gpu 6 --seed 1
# python main.py --n_hidden 1000 --gpu 7 --seed 2

nohup python main.py --n_steps 40000 --model_dump results/mnist_0.ckpt --n_hidden 100 --gpu 0 --ess_ratio 0.5 > v0.out &
nohup python main.py --n_steps 40000 --model_dump results/mnist_1.ckpt --n_hidden 400 --gpu 1 --ess_ratio 0.5 > v1.out &
nohup python main.py --n_steps 40000 --model_dump results/mnist_2.ckpt --n_hidden 1000 --gpu 1 --ess_ratio 0.5 > v2.out &
