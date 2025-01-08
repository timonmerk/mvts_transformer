#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --cpus-per-task=2
#SBATCH -o logs/pred.out
#SBATCH -e logs/pred.err
uv run src/get_pred.py