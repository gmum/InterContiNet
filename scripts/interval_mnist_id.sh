#!/bin/bash

#SBATCH --job-name=interval_cl
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 10

source .env

ns=True
elr=10.
mr=1.
lam=100.
si=5
lr=1.

for elr in 10; do
  for seed in 2001 2002 2003 2004 2005; do
    singularity exec --nv $SIF_PATH python3.9 train.py cfg=id cfg.seed=$seed cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=0.8 cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220111_incremental_domain_semifinal"]
  done
done