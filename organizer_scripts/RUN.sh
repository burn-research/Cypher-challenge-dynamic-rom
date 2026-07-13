#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err

#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00

echo "Start:"
date

source ~/envs/cypher2026/bin/activate

python3 -u prepare_data.py

echo "End:"
date