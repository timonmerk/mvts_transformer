#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:3
#SBATCH --time=7-0
#SBATCH --cpus-per-task=4
#SBATCH -o logs/25.out
#SBATCH -e logs/25.err
uv run src/main.py --name RAdam25 --lr 0.00001 --batch_size 2056 --optimizer RAdam --masking_ratio 0.1 --mean_mask_length 10 --mask_mode separate --num_workers 3 --max_seq_len 25