#!/bin/bash
#SBATCH --job-name=icecube_sims
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/neutrinos
mkdir -p logs

source ~/miniconda/etc/profile.d/conda.sh
conda activate icecube_clean

python run_generate_smoke.py
