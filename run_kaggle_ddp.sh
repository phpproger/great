#!/usr/bin/env bash
set -e

echo "===================================================================="
echo "  KAGGLE T4x2 DISTRIBUTED DATA PARALLEL (DDP) RUNNER"
echo "===================================================================="

# Check GPU availability
nvidia-smi

# Check Python & PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}')"

# Launch 2-GPU DDP training with torchrun (Tuned for Tesla T4, utilizes ~11.6 GB VRAM per GPU)
torchrun --nproc_per_node=2 train_ddp.py \
    --arch compare \
    --model_size 114m \
    --seq_len 512 \
    --batch_size 12 \
    --grad_accum 4 \
    --lr 4e-4 \
    --min_steps 400 \
    --max_steps 4000 \
    --eval_interval 50 \
    --patience 8 \
    --min_delta 0.001 \
    --output_dir checkpoints_ddp

echo "===================================================================="
echo "  DDP TRAINING COMPLETE! Checkpoints saved to checkpoints_ddp/"
echo "===================================================================="
