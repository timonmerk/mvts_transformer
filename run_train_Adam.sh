#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:3
#SBATCH --time=7-0
#SBATCH -o logs/Adam_out.out
#SBATCH -e logs/Adam.err
uv run src/main.py --name Adam --lr 0.0001 --batch_size 1024 --optimizer Adam --masking_ratio 0.1 --mean_mask_length 10 --num_workers 3 --load_model data/output/Adam/checkpoints/model_last.pth