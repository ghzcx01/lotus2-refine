#!/bin/bash

#SBATCH --job-name=lotus2_replica_depth
#SBATCH --account=cuuser_duan_deep_learning_for_3d_scene_modeling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=lotus2_replica_depth_%j.log
#SBATCH --error=lotus2_replica_depth_%j.err

set -euo pipefail

echo "Job Start Time: $(date)"
echo "Allocated Node: ${SLURM_JOB_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-none}"

cd /scratch/chenxiz/Lotus-2

export NVIDIA_DRIVER_CAPABILITIES=all
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/chenxiz/.conda/envs/lotus2/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}

PYTHON=/home/chenxiz/.conda/envs/lotus2/bin/python
OUTPUT_DIR=outputs/finetune_replica_depth

echo ">>> Start Lotus-2 Replica depth finetuning"
echo ">>> Python: ${PYTHON}"
echo ">>> Output: ${OUTPUT_DIR}"

${PYTHON} finetuning.py \
    --task_name=depth \
    --replica_root=replica \
    --replica_scale=0.5 \
    --crop_mode=original \
    --depth_normalization=minmax \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=20 \
    --train_batch_size=1 \
    --eval_split_ratio=0.1 \
    --eval_steps=100 \
    --eval_max_batches=32 \
    --visualization_steps=200 \
    --visualization_inference_steps=4 \
    --loss_curve_steps=200 \
    --dataloader_num_workers=0 \
    --mixed_precision=bf16 \
    --report_to=none \
    --gradient_checkpointing

echo "Job End Time: $(date)"
