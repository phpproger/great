# Banach Contraction LLaMA: 100M+ Multi-GPU Distributed Training

Production-grade Distributed Data Parallel (DDP) training of 100M+ parameter Banach-contractive LLaMA language models on WikiText-103, optimized to fully utilize dual Tesla T4 GPUs (16GB VRAM each) on Kaggle.

---

## Architecture Overview

This repository implements the optimized **Banach Contraction Baseline**:
* **Plain 2-Matrix MLP ($W_2(\text{SiLU}(W_1 x))$)**: Replaces standard SwiGLU 3-matrix bilinear gating with a monotonic Lipschitz operator, eliminating quadratic non-linear coupling $\mathcal{O}(x^2)$ and cutting 4-layer inversion error by $3\times$.
* **Spectral Barrier Regularization**: Soft barrier penalty $\max(0, \sigma(W) - 1.0)^2$ on $W_O$ and $W_2$ via 1-step power iteration Rayleigh quotients.
* **Multi-Step Banach Fixed-Point Inversion**: Reversible representation unrolling supporting both Picard iteration and **Anderson Acceleration ($m=2$)**.
* **$\sqrt{d}$ Embedding Scaling**: Preserves token embedding norm geometry.
* **Depth-Weighted Single-Layer Auxiliary Contraction Loss**: Stochastically supervises layer-wise reversibility with only $\sim 1.5\%$ compute overhead.

### Model Parameter Specifications

| Configuration | Parameters (Total) | Non-Embedding Backbone | Layers | Hidden Dim | Heads (Q / KV) | Context Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLaMA-114M** | **114,114,048** | **75,516,672** | 12 | 768 | 12 / 4 (GQA) | 1024 tokens |
| **LLaMA-Small** | 31,888,128 | 12,589,440 | 8 | 384 | 6 / 2 (GQA) | 512 tokens |
| **LLaMA-Tiny** | 7,220,480 | 787,584 | 4 | 128 | 4 / 2 (GQA) | 256 tokens |

---

## Kaggle 2x Tesla T4 (16GB x 2 VRAM) Execution

Tuned to utilize **~14.5 GB of the 16.0 GB VRAM** on each Tesla T4 GPU:
* **Context Length**: `1024 tokens`
* **Micro-Batch Size per GPU**: `24 sequences` ($24{,}576$ tokens per GPU)
* **Gradient Accumulation**: `2 steps`
* **Effective Batch Size**: `2 GPUs × 24 batch × 2 accum = 96 sequences = 98,304 tokens per optimizer step`

### 1. Launch via Shell Script
```bash
bash run_kaggle_ddp.sh
```

### 2. Launch via `torchrun`
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --arch compare \
    --model_size 114m \
    --seq_len 1024 \
    --batch_size 24 \
    --grad_accum 2 \
    --lr 4e-4 \
    --min_steps 400 \
    --max_steps 4000 \
    --eval_interval 50 \
    --patience 6 \
    --min_delta 0.005 \
    --output_dir checkpoints_ddp
```

### 3. Run in Kaggle Notebook
Open `kaggle_train_114m.ipynb` directly in Kaggle, select **Accelerator: GPU T4 x 2**, and click **Run All**.

---

## Directory Structure
```
great/
├── llama/
│   ├── config.py           # LlamaConfig with 114M architecture and Plain MLP defaults
│   ├── model.py            # LlamaModel, PlainMLP, TransformerBlock, Anderson Acceleration
│   ├── dataset.py          # WikiText-103 zero-copy memory-mapped dataset
│   └── __init__.py
├── train_ddp.py            # Multi-GPU Distributed Data Parallel training script
├── run_kaggle_ddp.sh       # Kaggle dual-GPU runner script
├── kaggle_train_114m.ipynb # Ready-to-run Kaggle Jupyter notebook
├── benchmark_combinations.py # Controlled convergence benchmark
└── README.md
```
