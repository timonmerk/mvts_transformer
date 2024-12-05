#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:tesla:3
#SBATCH --time=7-0
# Extract the base name of the script file (without extension)
SCRIPT_NAME=$(basename "$0" .sh)
#SBATCH -o logs/${SCRIPT_NAME}.out
#SBATCH -e logs/${SCRIPT_NAME}.err

uv run src/main.py --name lr_0001 --lr 0.0001 --batch_size 1024 --optimizer Aadam --masking_ratio 0.1 --mean_mask_length 10