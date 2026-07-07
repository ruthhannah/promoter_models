#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:1


eval "$(${CONDA_EXE:-/data/programs/conda/bin/conda} shell.bash hook)"
conda activate /home/hyewon/.conda/envs/MTLucifer
export PYTHONPATH=/home/hyewon/MTLucifer:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
# Slurm이 할당한 GPU를 0번으로 매핑 (SLURM_JOB_GPUS는 개수이지 ID가 아님)
export CUDA_VISIBLE_DEVICES=0
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
unset MPLBACKEND

cd /home/hyewon/MTLucifer/promoter_models

single_task=$1
modelling_strategy=$2
model_name=$3
batch_size=$4
max_epochs=$5
input_csv_path=$6
metric_to_monitor=$7
metric_direction_which_is_optimal=$8
val_chr=$9
test_chr=${10}
train_sampling_ratio=${11}
patience=${12}
wandb_project_name=${13}


python -u /home/hyewon/MTLucifer/promoter_models/train_models.py \
  --single_task $1 \
  --modelling_strategy $2 \
  --model_name $3 \
  --batch_size $4 \
  --max_epochs $5 \
  --input_csv_path $6 \
  --metric_to_monitor $7 \
  --metric_direction_which_is_optimal $8 \
  --val_chr $9 \
  --test_chr ${10} \
  --train_sampling_ratio ${11} \
  --patience ${12} \
  --wandb_project_name ${13}