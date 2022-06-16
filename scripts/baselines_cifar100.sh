#!/bin/bash

#SBATCH --job-name=interval_cl
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 10

source .env

trap "kill 0" INT

# SGD
for seed in 2001 2002 2003 2004 2005; do
  for offline in False; do
    for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
      singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
        tags=["20220302_rerun_drop_last_sgd_${scenario}"]
    done
  done
done

## ADAM
for offline in False; do
  for seed in 2001 2002 2003 2004 2005; do
    for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
      singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.optimizer=ADAM cfg.learning_rate=0.01 \
        cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
        tags=["20220302_rerun_drop_last_adam_${scenario}"]
    done
  done
done

# EWC
for seed in 2001 2002 2003 2004 2005; do
  for lambda in 0.2; do
    #    for optimizer in SGD ADAM; do
    for optimizer in SGD; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=EWC \
          cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
          cfg.scenario=${scenario} \
          tags=["20220302_rerun_drop_last_ewc_${scenario}"]
      done
    done
  done
done

# EWCOnline
for seed in 2001 2002 2003 2004 2005; do
  for lambda in 0.2; do
    #    for optimizer in SGD ADAM; do
    for optimizer in SGD; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=EWC cfg.ewc_mode=online cfg.seed=${seed} \
          cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
          cfg.scenario=${scenario} \
          tags=["20220302_rerun_drop_last_ewconline_${scenario}"]
      done
    done
  done
done

# L2
for seed in 2001 2002 2003 2004 2005; do
  #  for lambda in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8196 16384; do
  for lambda in 0.001; do
    #    for optimizer in SGD ADAM; do
    for optimizer in SGD; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=L2 cfg.seed=${seed} \
          cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
          cfg.scenario=${scenario} \
          tags=["20220302_rerun_drop_last_l2_${scenario}"]
      done
    done
  done
done

# LWF
for seed in 2001 2002 2003 2004 2005; do
  for temperature in 1; do
    for alpha in 2; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=LWF cfg.seed=${seed} cfg.lwf_alpha=${alpha} \
          cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} \
          tags=["20220302_rerun_drop_last_lwf_${scenario}"]
      done
    done
  done
done


# ## Synaptic Intelligence
# for seed in 2001 2002 2003 2004 2005; do
# #  for lambda in 0.001 0.1 1 8 32 128; do
#   for lambda in 0.001 0.1 1 8 32 128; do
#     for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=SI \
#         cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}","stdruns"] &
#       idx=$((idx+1))
#       export CUDA_VISIBLE_DEVICES=$idx
#       if [ $idx -eq 4 ]; then
#         idx=0
#         wait
#       fi
#     done
#     wait
#   done
# done
# 
# ## MAS
# for seed in 2001 2002 2003 2004 2005; do
#    for lambda in 0.001; do
# #  for lambda in 0.0001 0.001 0.01 0.1 1.0 10 100; do
#     for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.strategy=MAS cfg.seed=${seed} cfg.ewc_lambda=${lambda} \
#         cfg.scenario=${scenario} tags=["${scenario}","stdruns"] &
#     done
#     wait
#   done
# done
# 
