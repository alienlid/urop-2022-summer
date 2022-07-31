#!/bin/bash
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --output=/mnt/nfs/home/ddl/urop/transfer_learning/gaussian_blur/ll-zs-%a.out
#SBATCH --array=0
srun python3 mix.py
