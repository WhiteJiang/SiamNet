#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py --dataset cub-2011 --root /home/jx/data/CUB_200_2011 --max-epoch 30 --gpu 1 --arch semicon --batch-size 16 --max-iter 20 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 16 --num-samples 2000 --info 'SiamNet-128hash-soft_mask' --momen 0.91
