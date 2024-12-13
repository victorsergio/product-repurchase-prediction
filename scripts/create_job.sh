#!/bin/bash
#SBATCH -J "VPENALOZA" # Job name with timestamp
#SBATCH -n 1 # number of jobs
#SBATCH -c 1 # number of cores
#SBATCH --mem=5G
#SBATCH -p GPU-DEPINFO
#SBATCH -t 7-0  # means a maximum of 7 days to run the job
#SBATCH -o /home/pv10123z/mldm-project/logs/train_%j.out
#SBATCH -e /home/pv10123z/mldm-project/logs/train_%j.err

source /home/pv10123z/mldm-project/environment/bin/activate

srun --exclusive python3 /home/pv10123z/mldm-project/src/train_model.py
