#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:3
#SBATCH --time=7-0
#SBATCH -o logs/RAdam_out.out
#SBATCH -e logs/RAdam.err

uv run src/main.py --name RAdam --lr 0.0001 --batch_size 1024 --optimizer RAdam --masking_ratio 0.1 --mean_mask_length 10 --num_workers 3