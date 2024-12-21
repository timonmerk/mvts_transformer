#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --time=7-0
#SBATCH --cpus-per-task=4
#SBATCH -o logs/W250AdamMSEe5.out
#SBATCH -e logs/W250AdamMSEe5.err
uv run src/main.py --name WAdam250 --lr 0.00001 --batch_size 1024 --optimizer AdamW --masking_ratio 0.4 --mean_mask_length 50 --mask_mode concurrent --num_workers 3 --max_seq_len 250 --num_heads 8 --d_model 256 --dim_feedforward 512 --num_layers 6 --epochs 200 --loss MSE --concat_fft