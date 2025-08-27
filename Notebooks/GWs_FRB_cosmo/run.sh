#!/bin/bash
#SBATCH --job-name="MCMC_all"
#SBATCH --output="%j.MCMC_all.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --account=unv114
#SBATCH --ntasks-per-node=108
#SBATCH -t 48:00:00
#SBATCH --mem=0
#SBATCH --constraint="lustre"

# --mem=2055552M
# large-shared
module purge
module load shared
module load slurm
module load cpu/0.17.3b

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

python3 -W ignore::DeprecationWarning Cosmo_constraints_DM_ext_MCMC.py
