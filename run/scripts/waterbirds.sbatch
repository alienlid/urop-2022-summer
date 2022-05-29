#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --job-name wb
#SBATCH --output=/mnt/cfs/home/harshay/repos/dmgen/logs/wb_%a.out
#SBATCH --error=/mnt/cfs/home/harshay/repos/dmgen/logs/wb_%a.err
#SBATCH --array=0-1

logdir=~/out/dmgen/waterbirds/final-$SLURM_ARRAY_JOB_ID
cd /mnt/cfs/home/harshay/repos/dmgen/

python -m run.run_waterbirds --config configs/waterbirds_final.yaml \
                             --training.save_root_dir $logdir \
                             --expt.num_runs 8 \
                             --expt.sbatch 1 \
                             --expt.sbatch_expt final \
                             --expt.sbatch_jobid $SLURM_ARRAY_TASK_ID