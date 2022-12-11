#!/bin/bash

 python run.py --dataset cub-2011 --root /home/jx/data/CUB_200_2011 --gpu 0 --arch test --batch-size 16 --code-length 12 --wd 1e-4 --info 'SiamNet-mid-layer'