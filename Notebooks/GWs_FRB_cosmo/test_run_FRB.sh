#!/bin/bash
#SBATCH --job-name="test_FRB"
#SBATCH --output="%j.test_FRB.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --account=unv116
#SBATCH --ntasks-per-node=96
#SBATCH -t 48:00:00
#SBATCH --mem=50G
#SBATCH --constraint="lustre"

module purge
module load shared
module load slurm
module load cpu/0.17.3b

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

python3 -W ignore::DeprecationWarning test_DM_diff_only_FRB.py

python3 ../../../automail.py -task 'DM_diff'