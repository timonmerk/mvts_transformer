#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --time=1-0
#SBATCH --cpus-per-task=4
#SBATCH -o logs/pred.out
#SBATCH -e logs/pred.err
uv run src/get_pred.py