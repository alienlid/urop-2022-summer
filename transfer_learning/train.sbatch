#!/bin/bash
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --output=/mnt/nfs/home/ddl/urop/transfer_learning/gaussian_blur/%a-fn.out
#SBATCH --array=19
srun python3 train.py
