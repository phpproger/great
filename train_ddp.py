import os
import sys
import math
import time
import copy
import json
import argparse
import random
import tempfile
from contextlib import nullcontext
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import tiktoken

from llama.config import LlamaConfig
from llama.model import LlamaForCausalLM, estimate_spectral_norm
from llama.dataset import WikiTextDataset


# ============================================================================
# 1. DISTRIBUTED SETUP AND UTILITIES
# ============================================================================
def safe_makedirs(dir_path: str) -> str:
    """Creates directory safely with fallback to temp directory if permission is denied."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    except PermissionError:
        alt_path = os.path.join(tempfile.gettempdir(), os.path.basename(dir_path))
        os.makedirs(alt_path, exist_ok=True)
        return alt_path


def init_distributed():
    """Initializes PyTorch DDP environment with NCCL or Gloo backend.
    Gracefully handles multi-GPU torchrun environments (Kaggle 2x Tesla T4)
    and single-GPU fallback for local testing.
    """
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        torch.cuda.set_device(local_rank)
        # NCCL on Linux (Kaggle T4x2), Gloo on Windows
        backend = "nccl" if torch.cuda.is_available() and sys.platform != "win32" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return is_distributed, global_rank, local_rank, world_size, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int = 0):
    if rank == 0:
        print(msg, flush=True)


# ============================================================================
# 2. DATASET INITIALIZATION WITH RANK-0 BARRIER PREPARATION
# ============================================================================
def prepare_wikitext_data(data_dir: str, rank: int, is_distributed: bool):
    """Ensures WikiText binary token files exist. If missing (e.g. on fresh Kaggle instance),
    downloads wikitext-103 on rank 0 and tokenizes it, while other ranks wait at barrier.
    """
    train_tokens_path = os.path.join(data_dir, "train-tokens.bin")
    if os.path.exists(train_tokens_path):
        return

    if rank == 0:
        print_rank0(f"[*] Preparing WikiText-103 in {data_dir}...", rank)
        safe_makedirs(data_dir)
        try:
            from datasets import load_dataset
            enc = tiktoken.get_encoding("gpt2")
            splits = ["validation", "test", "train"]
            for split_name in splits:
                hf_split = "validation" if split_name == "validation" else split_name
                print_rank0(f"    - Downloading & tokenizing {split_name} split...", rank)
                ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split)
                offsets = [0]
                all_tokens = []
                for row in ds:
                    text = row["text"]
                    if not text or not text.strip():
                        continue
                    tokens = enc.encode_ordinary(text)
                    all_tokens.extend(tokens)
                    offsets.append(len(all_tokens))

                bin_file = os.path.join(data_dir, f"{split_name}-tokens.bin")
                off_file = os.path.join(data_dir, f"{split_name}-offsets.npy")
                import numpy as np
                arr = np.array(all_tokens, dtype=np.uint32)
                arr.tofile(bin_file)
                np.save(off_file, np.array(offsets, dtype=np.int64))
                print_rank0(f"    - Saved {len(all_tokens):,} tokens to {bin_file}", rank)
        except Exception as e:
            print_rank0(f"[!] Auto-dataset download failed: {e}. Ensure {data_dir} is mounted.", rank)

    if is_distributed:
        dist.barrier()


# ============================================================================
# 3. EVALUATION ACROSS DISTRIBUTED PROCESSES
# ============================================================================
def evaluate_loss(model, val_loader, vocab_size: int, device, is_distributed: bool, max_batches: int = 15) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            tokens = batch["input_ids"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
                _, loss, _ = model(input_ids=tokens, labels=tokens)

            total_loss += loss.item()
            total_batches += 1.0

    if is_distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    model.train()
    return (total_loss / max(total_batches, torch.tensor(1.0, device=device))).item()


# ============================================================================
# 4. TRAINING LOOP TO PLATEAU (TUNED TO MAXIMIZE 16GB VRAM ON T4x2)
# ============================================================================
def train_model(
    config: LlamaConfig,
    arch_name: str,
    args,
    train_dataset,
    val_dataset,
    is_distributed: bool,
    global_rank: int,
    local_rank: int,
    world_size: int,
    device,
) -> Dict[str, Any]:
    output_dir = safe_makedirs(args.output_dir)

    print_rank0("\n" + "=" * 115, global_rank)
    print_rank0(f"  STARTING DDP TRAINING: {arch_name.upper()} (114M PARAMETERS ON KAGGLE T4x2)", global_rank)
    print_rank0(f"  Architecture: {config.num_hidden_layers} Layers | {config.hidden_size} Hidden | {config.num_attention_heads} Heads", global_rank)
    print_rank0(f"  MLP Type: {config.mlp_type} | Spectral Barrier: {config.use_spectral_barrier}", global_rank)
    print_rank0(f"  GPUs: {world_size}x Tesla T4 (16GB VRAM each) | Context Window: {args.seq_len} tokens", global_rank)
    print_rank0(f"  Batch/GPU: {args.batch_size} | Gradient Accumulation: {args.grad_accum}", global_rank)
    effective_batch = args.batch_size * world_size * args.grad_accum
    effective_tokens = effective_batch * args.seq_len
    print_rank0(f"  Effective Batch Size: {effective_batch} sequences = {effective_tokens:,} tokens per optimizer step", global_rank)
    print_rank0("=" * 115, global_rank)

    # Initialize model
    torch.manual_seed(42 + global_rank)
    raw_model = LlamaForCausalLM(config).to(device)
    total_params = raw_model.get_num_params()
    non_emb_params = raw_model.get_num_params(non_embedding=True)
    print_rank0(f"  Total Model Parameters: {total_params:,} | Non-Embedding Backbone: {non_emb_params:,}", global_rank)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False, # Optimized: all transformer layers are in active residual path
        )
    else:
        model = raw_model

    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.lr * 0.1,
    )

    # Auto-resume from existing checkpoint if explicitly requested
    save_path = os.path.join(output_dir, f"{arch_name}_114m_best.pt")
    start_step = 0
    resumed = False
    resumed_best_val = None

    if getattr(args, "resume", False) and os.path.exists(save_path):
        try:
            print_rank0(f"  [Auto-Resume] Found existing checkpoint at {save_path}! Loading...", global_rank)
            ckpt_obj = torch.load(save_path, map_location=device)
            if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
                raw_model.load_state_dict(ckpt_obj["model_state_dict"])
                start_step = ckpt_obj.get("step", 0)
                resumed_best_val = ckpt_obj.get("best_val", None)
                if "optimizer_state_dict" in ckpt_obj and ckpt_obj["optimizer_state_dict"] is not None:
                    try:
                        optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
                    except Exception:
                        pass
            else:
                raw_model.load_state_dict(ckpt_obj)
            resumed = True
            print_rank0(f"  [Auto-Resume Success] Restored model weights from {save_path} (resuming from step {start_step})", global_rank)
        except Exception as e:
            print_rank0(f"  [Auto-Resume Warning] Could not load {save_path}: {e}. Starting fresh.", global_rank)

    # Data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if is_distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=1 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_loader) == 0:
        raise RuntimeError(f"Train DataLoader has 0 batches! Dataset size ({len(train_dataset)}) is too small for batch_size={args.batch_size} and world_size={world_size}.")

    # Plateau tracking
    vocab_size = config.vocab_size
    init_val = evaluate_loss(model, val_loader, vocab_size, device, is_distributed)
    if resumed and resumed_best_val is not None:
        best_val = min(init_val, resumed_best_val)
    else:
        best_val = init_val
    best_step = start_step
    patience_counter = 0
    plateau_reached = False
    best_state_dict = None

    history = {
        "train_steps": [],
        "train_losses": [],
        "lrs": [],
        "val_steps": [],
        "val_losses": [],
        "val_ppls": [],
    }

    print_rank0(f"  Initial Validation Loss: {init_val:.4f} (Initial PPL: {math.exp(min(init_val, 20)):.2f})", global_rank)
    print_rank0(f"  Criterion: Train dynamically until plateau (min_steps={args.min_steps}, patience={args.patience} checks x {args.eval_interval} st = {args.patience * args.eval_interval} st)", global_rank)
    print_rank0("-" * 115, global_rank)

    step = start_step
    micro_step = start_step * args.grad_accum
    epoch = (start_step * args.grad_accum) // max(1, len(train_loader))
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps and not plateau_reached:
        epoch += 1
        if is_distributed:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            tokens = batch["input_ids"].to(device, non_blocking=True)

            is_accumulating = (micro_step + 1) % args.grad_accum != 0

            # Suppress DDP all-reduce on accumulation micro-steps (huge speedup)
            ddp_context = model.no_sync() if (is_distributed and is_accumulating) else nullcontext()
            with ddp_context:
                with torch.amp.autocast("cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
                    logits, loss, _ = model(input_ids=tokens, labels=tokens)
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()

            micro_step += 1

            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                # Logging & VRAM monitoring every 20 optimizer steps
                if step % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    print_rank0(f"  [Step {step:5d}/{args.max_steps}] Loss: {loss.item():.4f} | VRAM: {vram_gb:.2f} GB / 16.0 GB | LR: {current_lr:.2e}", global_rank)
                    if global_rank == 0:
                        history["train_steps"].append(step)
                        history["train_losses"].append(round(loss.item(), 4))
                        history["lrs"].append(float(current_lr))

                # Evaluation & Plateau Check every args.eval_interval optimizer steps
                if step % args.eval_interval == 0:
                    v_loss = evaluate_loss(model, val_loader, vocab_size, device, is_distributed)
                    v_ppl = math.exp(min(v_loss, 20))
                    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    print_rank0(f"  --> [Check @ Step {step:5d}] Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:.2f} | Peak VRAM: {vram_gb:.2f} GB", global_rank)

                    if global_rank == 0:
                        history["val_steps"].append(step)
                        history["val_losses"].append(round(v_loss, 4))
                        history["val_ppls"].append(round(v_ppl, 2))

                    if v_loss < best_val - args.min_delta:
                        best_val = v_loss
                        best_step = step
                        patience_counter = 0
                        if global_rank == 0:
                            best_state_dict = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                            print_rank0(f"      [Best Model Updated] New best validation loss {best_val:.4f} (PPL {v_ppl:.2f})", global_rank)
                            # Periodic snapshot every 200 optimizer steps so auto-resume can recover if interrupted
                            if step % 200 == 0:
                                ckpt_obj = {
                                    "model_state_dict": best_state_dict,
                                    "step": best_step,
                                    "best_val": best_val,
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "history": history,
                                }
                                torch.save(ckpt_obj, save_path)
                                hist_path = os.path.join(output_dir, f"{arch_name}_history.json")
                                with open(hist_path, "w") as f:
                                    json.dump(history, f, indent=2)
                                print_rank0(f"      [Periodic Checkpoint Saved] Saved snapshot & history to {save_path}", global_rank)
                    else:
                        if step >= args.min_steps:
                            patience_counter += 1
                            print_rank0(f"      [Plateau Alert] No improvement >= {args.min_delta} for {patience_counter}/{args.patience} checks", global_rank)
                            if patience_counter >= args.patience:
                                plateau_reached = True
                                print_rank0(f"      [CONVERGED] Dynamic plateau achieved at step {best_step}!", global_rank)
                                break

            if step >= args.max_steps:
                break

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print_rank0(f"  Training for {arch_name} finished in {total_time:.1f}s ({total_time/60:.2f} min). Peak VRAM: {peak_vram:.2f} GB", global_rank)
    print_rank0(f"  Best Validation Loss: {best_val:.4f} | Best Validation PPL: {math.exp(min(best_val, 20)):.2f} at Step {best_step}", global_rank)

    # Save final best checkpoint to disk when model converges to plateau
    if global_rank == 0 and best_state_dict is not None:
        save_path = os.path.join(output_dir, f"{arch_name}_114m_best.pt")
        ckpt_obj = {
            "model_state_dict": best_state_dict,
            "step": best_step,
            "best_val": best_val,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        torch.save(ckpt_obj, save_path)
        hist_path = os.path.join(output_dir, f"{arch_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print_rank0(f"  [Checkpoint Saved] Wrote final best plateau checkpoint & history to {save_path}", global_rank)

    if is_distributed:
        dist.barrier()

    return {
        "arch": arch_name,
        "best_step": best_step,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(min(best_val, 20)),
        "total_time_sec": total_time,
        "peak_vram_gb": peak_vram,
        "total_params": total_params,
        "non_emb_params": non_emb_params,
        "history": history,
    }


# ============================================================================
# 5. CLI & MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU DDP Training for 100M+ Banach Contraction LLaMA Models on Kaggle T4x2")
    parser.add_argument("--arch", type=str, default="compare", choices=["compare", "plain_spectral", "swiglu_baseline"],
                        help="Architecture to train: 'plain_spectral' (champion), 'swiglu_baseline', or 'compare' (both)")
    parser.add_argument("--model_size", type=str, default="114m", choices=["114m", "small", "tiny"])
    parser.add_argument("--data_dir", type=str, default="data/wikitext", help="Path to pretokenized WikiText-103 files")
    parser.add_argument("--output_dir", type=str, default="checkpoints_ddp", help="Directory to save checkpoints")
    # Tuned for Tesla T4 (16GB VRAM each):
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence context length (default 512 tokens)")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per GPU (default 12, uses ~10 GB VRAM; use 16 for ~13.5 GB)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective 49,152 tokens/step)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--min_steps", type=int, default=400, help="Minimum steps before plateau patience activates")
    parser.add_argument("--max_steps", type=int, default=4000, help="Maximum training steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Steps between validation evaluations")
    parser.add_argument("--patience", type=int, default=8, help="Number of evaluations without improvement before plateau")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum validation loss improvement delta (default 0.001)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available")
    args = parser.parse_args()

    is_distributed, global_rank, local_rank, world_size, device = init_distributed()

    print_rank0("=" * 115, global_rank)
    print_rank0("  LLAMA 100M+ BANACH CONTRACTION: DISTRIBUTED DATA PARALLEL (DDP) TRAINING PIPELINE", global_rank)
    print_rank0(f"  PyTorch: {torch.__version__} | Distributed: {is_distributed} | World Size: {world_size} | Device: {device}", global_rank)
    print_rank0("=" * 115, global_rank)

    # 1. Dataset setup
    prepare_wikitext_data(args.data_dir, global_rank, is_distributed)

    train_ds = WikiTextDataset(data_dir=args.data_dir, split="train", seq_len=args.seq_len)
    val_ds = WikiTextDataset(data_dir=args.data_dir, split="validation", seq_len=args.seq_len, max_samples=1000)

    print_rank0(f"  WikiText Datasets Loaded: Train={len(train_ds):,} samples | Val={len(val_ds):,} samples (seq_len={args.seq_len})", global_rank)

    # 2. Configurations
    def get_cfg(mlp_type: str, use_spectral: bool):
        if args.model_size == "114m":
            cfg = LlamaConfig.llama_114m(vocab_size=50257)
        elif args.model_size == "small":
            cfg = LlamaConfig.small(vocab_size=50257)
        else:
            cfg = LlamaConfig.tiny(vocab_size=50257)
        cfg.mlp_type = mlp_type
        cfg.use_spectral_barrier = use_spectral
        cfg.max_position_embeddings = max(cfg.max_position_embeddings, args.seq_len + 64)
        return cfg

    results = {}

    # 3. Train requested architectures
    if args.arch in ["swiglu_baseline", "compare"]:
        cfg_swiglu = get_cfg("swiglu", use_spectral=False)
        cfg_swiglu.use_contraction_aux = False
        cfg_swiglu.scale_embeddings = False
        res_swi = train_model(
            cfg_swiglu,
            "swiglu_baseline",
            args,
            train_ds,
            val_ds,
            is_distributed,
            global_rank,
            local_rank,
            world_size,
            device,
        )
        results["swiglu_baseline"] = res_swi

    if args.arch in ["plain_spectral", "compare"]:
        cfg_plain_spec = get_cfg("plain", use_spectral=True)
        cfg_plain_spec.use_contraction_aux = True
        cfg_plain_spec.scale_embeddings = True
        res_plain = train_model(
            cfg_plain_spec,
            "plain_spectral",
            args,
            train_ds,
            val_ds,
            is_distributed,
            global_rank,
            local_rank,
            world_size,
            device,
        )
        results["plain_spectral"] = res_plain

    # 4. Final Comparison Matrix (Rank 0)
    if global_rank == 0 and len(results) > 1:
        print("\n" + "=" * 105)
        print("  114M DDP PRODUCTION COMPARISON MATRIX (TRAINED TO PLATEAU ON WIKITEXT-103 ON KAGGLE T4x2)")
        print("=" * 105)
        print(f"  {'Metric':<32} | {'0. SwiGLU Baseline':<28} | {'1. Plain MLP + Spectral (Champion)':<34}")
        print("-" * 105)
        r0 = results["swiglu_baseline"]
        r1 = results["plain_spectral"]
        print(f"  {'Parameters (Total)':<32} | {r0['total_params']:<28,d} | {r1['total_params']:<34,d}")
        print(f"  {'Parameters (Non-Embedding)':<32} | {r0['non_emb_params']:<28,d} | {r1['non_emb_params']:<34,d}")
        print(f"  {'Plateau Step':<32} | {r0['best_step']:<28d} | {r1['best_step']:<34d}")
        print(f"  {'Peak VRAM Per GPU':<32} | {r0['peak_vram_gb']:<25.2f} GB | {r1['peak_vram_gb']:<31.2f} GB")
        print(f"  {'Validation Cross-Entropy':<32} | {r0['best_val_loss']:<28.4f} | {r1['best_val_loss']:<34.4f}")
        print(f"  {'Validation Perplexity (PPL)':<32} | {r0['best_val_ppl']:<28.2f} | {r1['best_val_ppl']:<34.2f}")
        delta = r1['best_val_ppl'] - r0['best_val_ppl']
        print(f"  {'Delta PPL vs Baseline':<32} | {'0.00 (Reference)':<28} | {delta:<+34.2f}")
        print(f"  {'Total Training Time':<32} | {r0['total_time_sec']/60:<25.1f} min | {r1['total_time_sec']/60:<31.1f} min")
        print("=" * 105)

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
