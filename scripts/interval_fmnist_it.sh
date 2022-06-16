#!/bin/bash

#SBATCH --job-name=interval_cl
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 10

source .env

ns=True
elr=1.
mr=1.
lam=100.
si=5
lr=0.001

idx=0
for seed in 2001 2002 2003 2004 2005; do
  for lam in 100; do
    for elr in 1; do
      singularity exec --nv $SIF_PATH python3.9 train.py cfg=it cfg.dataset=FASHION_MNIST cfg.optimizer=SGD cfg.seed=$seed cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220124_fmnist_incremental_task_hps"] &
    done
  done
  wait
done
