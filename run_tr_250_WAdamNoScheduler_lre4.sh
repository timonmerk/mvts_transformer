#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --time=7-0
#SBATCH --cpus-per-task=4
#SBATCH -o logs/W250AdamNoSched.out
#SBATCH -e logs/W250AdamNoSched.err
uv run src/main.py --name WAdam250 --lr 0.0001 --batch_size 1024 --optimizer WAdam --masking_ratio 0.4 --mean_mask_length 50 --mask_mode concurrent --num_workers 3 --max_seq_len 250 --num_heads 8 --d_model 256 --dim_feedforward 512 --num_layers 6 --epochs 200 --loss MSE --scheduler None
