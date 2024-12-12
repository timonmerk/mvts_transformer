#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:A40:3
#SBATCH --time=7-0
#SBATCH --cpus-per-task=4
#SBATCH -o logs/250.out
#SBATCH -e logs/250.err
uv run src/main.py --name RAdam250 --lr 0.00001 --batch_size 2056 --optimizer RAdam --masking_ratio 0.1 --mean_mask_length 10 --mask_mode concurrent --num_workers 3 --max_seq_len 250