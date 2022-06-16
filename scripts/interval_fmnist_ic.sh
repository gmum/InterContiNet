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
lam=1000.
si=5
lr=0.001
rat=0.9

for seed in 2001 2002 2003 2004 2005; do
  for elr in 0.001; do
    for lam in 100; do
      singularity exec --nv $SIF_PATH python3.9 train.py cfg=it cfg.dataset=FASHION_MNIST cfg.seed=$seed cfg.scenario=INC_CLASS cfg.optimizer=SGD cfg.interval.max_radius=${mr} cfg.interval.robust_accuracy_threshold=$rat cfg.interval.robust_lambda=$lam cfg.interval.expansion_learning_rate=$elr cfg.learning_rate=$lr cfg.interval.scale_init=5.0 tags=["20220124_fmnist_incremental_class_semifinal"] &
    done
  done
  wait
done
