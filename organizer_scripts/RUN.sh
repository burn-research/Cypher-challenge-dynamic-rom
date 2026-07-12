#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_data.out
#SBATCH --error=logs/prepare_data.err
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00

echo "Start:"
date

module load Python/3.11.3
source /globalsc/ulb/atm/baffetti/envs/apriori/bin/activate

python3 -u prepare_data.py

echo "End:"
date