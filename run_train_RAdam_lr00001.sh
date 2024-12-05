#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:tesla:3
#SBATCH --time=7-0
#SBATCH -o logs/RAdam_out_lr.out
#SBATCH -e logs/RAdam_lr.err

uv run src/main.py --name lr_00001 --lr 0.00001 --batch_size 1024 --optimizer RAdam --masking_ratio 0.1 --mean_mask_length 10