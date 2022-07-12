#!/bin/bash
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --output=/mnt/nfs/home/ddl/urop/transfer_learning/gaussian_blur/%a-ll.out
#SBATCH --array=0-104%15
srun python3 train.py
