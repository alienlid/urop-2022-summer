#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name c10
#SBATCH --output=/mnt/cfs/home/harshay/repos/dmgen/logs/c10varylr_%a.out
#SBATCH --error=/mnt/cfs/home/harshay/repos/dmgen/logs/c10varylr_%a.err
#SBATCH --array=0-3

logdir=~/out/dmgen/cifar10/cifar10_lrbs-$SLURM_ARRAY_JOB_ID
cd /mnt/cfs/home/harshay/repos/dmgen 

python -m run.run_cifar --config run/configs/cifar.yaml \
                        --training.device_id 0 \
                        --expt.expt_name cifar10 \
                        --training.train_stop_epsilon 0.02 \
                        --training.save_root_dir $logdir \
                        --expt.num_runs 4 \
                        --expt.sbatch 1 \
                        --expt.sbatch_expt lr_bs \
                        --expt.sbatch_jobid $SLURM_ARRAY_TASK_ID