#!/usr/bin/env bash

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p data/cl

seed="${SEED:-12}"
batch_size="${BATCH_SIZE:-256}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-1}"
replay_sample_batch_size="${REPLAY_SAMPLE_BATCH_SIZE:-1000}"
experiment_suffix="${EXPERIMENT_SUFFIX:-}"

if [[ -z "${experiment_suffix}" && ("${batch_size}" != "256" || "${grad_accum_steps}" != "1") ]]; then
    experiment_suffix="_B${batch_size}_A${grad_accum_steps}"
fi

config_dir="configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10"
common_args=(--seed "${seed}" --batch-size "${batch_size}" --accumulate-grad-batches "${grad_accum_steps}" --replay-sample-batch-size "${replay_sample_batch_size}")

task1="CL_CIFAR10_SUP_TASK1_SEED${seed}${experiment_suffix}"
task2="CL_CIFAR10_SUP_TASK2_SEED${seed}${experiment_suffix}"
task3="CL_CIFAR10_SUP_TASK3_SEED${seed}${experiment_suffix}"
task4="CL_CIFAR10_SUP_TASK4_SEED${seed}${experiment_suffix}"
task5="CL_CIFAR10_SUP_TASK5_SEED${seed}${experiment_suffix}"
joint12="CL_CIFAR10_SUP_TASK1-2_SEED${seed}${experiment_suffix}"
joint13="CL_CIFAR10_SUP_TASK1-3_SEED${seed}${experiment_suffix}"
joint14="CL_CIFAR10_SUP_TASK1-4_SEED${seed}${experiment_suffix}"
joint15="CL_CIFAR10_SUP_TASK1-5_SEED${seed}${experiment_suffix}"

python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_sigle_task.yaml" -t 0 -d "${task1}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_sigle_task.yaml" -t 1 -d "${task2}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_kl_t2.yaml" -t 1 -o "logs/${task1}/checkpoints/last.ckpt" -l 0 -n "logs/${task2}/checkpoints/last.ckpt" -c "logs/${task1}/checkpoints/last.ckpt" -d "${joint12}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_sigle_task.yaml" -t 2 -d "${task3}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_kl_t3.yaml" -t 2 -o "logs/${joint12}/checkpoints/last.ckpt" -l 0 1 -n "logs/${task3}/checkpoints/last.ckpt" -c "logs/${joint12}/checkpoints/last.ckpt" -d "${joint13}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_sigle_task.yaml" -t 3 -d "${task4}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_kl_t4.yaml" -t 3 -o "logs/${joint13}/checkpoints/last.ckpt" -l 0 1 2 -n "logs/${task4}/checkpoints/last.ckpt" -c "logs/${joint13}/checkpoints/last.ckpt" -d "${joint14}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_sigle_task.yaml" -t 4 -d "${task5}" "${common_args[@]}"
python train_joint_diffusion_cl.py -p "${config_dir}/cifar10_kl_t5.yaml" -t 4 -o "logs/${joint14}/checkpoints/last.ckpt" -l 0 1 2 3 -n "logs/${task5}/checkpoints/last.ckpt" -c "logs/${joint14}/checkpoints/last.ckpt" -d "${joint15}" "${common_args[@]}"
