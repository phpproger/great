import os
import re
import json
import math
import argparse
from typing import Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import torch

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="deep")
except ImportError:
    pass


def parse_history_from_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Loads recorded training history from JSON."""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning loading {json_path}: {e}")
    return None


def parse_history_from_checkpoint(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Loads history saved directly inside checkpoint."""
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "history" in ckpt:
                return ckpt["history"]
        except Exception as e:
            print(f"[!] Warning reading checkpoint {ckpt_path}: {e}")
    return None


def parse_history_from_log_text(log_text: str) -> Dict[str, Any]:
    """Parses raw console / Kaggle stdout log text using regex."""
    history = {
        "train_steps": [],
        "train_losses": [],
        "lrs": [],
        "val_steps": [],
        "val_losses": [],
        "val_ppls": [],
    }

    # Match train steps: [Step    20/4000] Loss: 7.6563 | VRAM: 9.51 GB / 16.0 GB | LR: 4.00e-04
    train_pattern = re.compile(r"\[Step\s+(\d+)/\d+\]\s+Loss:\s+([\d\.]+)\s+\|\s+VRAM:.*?\|\s+LR:\s+([\d\.e\-\+]+)")
    for match in train_pattern.finditer(log_text):
        step = int(match.group(1))
        loss = float(match.group(2))
        lr = float(match.group(3))
        history["train_steps"].append(step)
        history["train_losses"].append(loss)
        history["lrs"].append(lr)

    # Match val checks: --> [Check @ Step    50] Val Loss: 6.9188 | Val PPL: 1011.13
    val_pattern = re.compile(r"Check @ Step\s+(\d+)\].*?Val Loss:\s+([\d\.]+)\s+\|\s+Val PPL:\s+([\d\.]+)")
    for match in val_pattern.finditer(log_text):
        step = int(match.group(1))
        loss = float(match.group(2))
        ppl = float(match.group(3))
        history["val_steps"].append(step)
        history["val_losses"].append(loss)
        history["val_ppls"].append(ppl)

    return history


def load_architecture_history(output_dir: str, arch_name: str, fallback_log_text: Optional[str] = None) -> Dict[str, Any]:
    """Tries JSON -> Checkpoint -> Raw Log parsing in order of preference."""
    # 1. JSON
    json_path = os.path.join(output_dir, f"{arch_name}_history.json")
    hist = parse_history_from_json(json_path)
    if hist and len(hist.get("val_steps", [])) > 0:
        return hist

    # 2. Checkpoint
    ckpt_path = os.path.join(output_dir, f"{arch_name}_114m_best.pt")
    hist = parse_history_from_checkpoint(ckpt_path)
    if hist and len(hist.get("val_steps", [])) > 0:
        return hist

    # 3. Log file on disk
    log_file = os.path.join(output_dir, f"{arch_name}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            hist = parse_history_from_log_text(f.read())
            if len(hist.get("val_steps", [])) > 0:
                return hist

    # 4. Provided fallback log text
    if fallback_log_text:
        hist = parse_history_from_log_text(fallback_log_text)
        if len(hist.get("val_steps", [])) > 0:
            return hist

    return {"train_steps": [], "train_losses": [], "lrs": [], "val_steps": [], "val_losses": [], "val_ppls": []}


def plot_comparison_curves(
    hist_swiglu: Dict[str, Any],
    hist_plain: Dict[str, Any],
    save_path: str = "checkpoints_ddp/comparison_curves.png",
    show_plot: bool = True,
):
    """Generates 2x2 publication-quality comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    fig.patch.set_facecolor("#FAFAFA")

    # Colors
    c_swiglu = "#E63946"   # Crimson red
    c_plain = "#1D3557"    # Deep navy blue
    c_gpt2 = "#6C757D"     # Slate grey reference

    # -------------------------------------------------------------------------
    # Subplot 1: Validation Perplexity (PPL) vs Step
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")
    
    if hist_swiglu.get("val_steps"):
        ax1.plot(
            hist_swiglu["val_steps"],
            hist_swiglu["val_ppls"],
            label="0. SwiGLU Baseline (114M)",
            color=c_swiglu,
            linewidth=2.2,
            marker="o",
            markersize=5,
            alpha=0.9,
        )
        min_swi = min(hist_swiglu["val_ppls"])
        min_swi_step = hist_swiglu["val_steps"][hist_swiglu["val_ppls"].index(min_swi)]
        ax1.scatter([min_swi_step], [min_swi], color=c_swiglu, s=120, zorder=5, edgecolors="black", linewidth=1.5)
        ax1.annotate(
            f"Best SwiGLU: {min_swi:.2f} PPL",
            (min_swi_step, min_swi),
            textcoords="offset points",
            xytext=(-20, 15),
            fontweight="bold",
            color=c_swiglu,
            fontsize=10,
        )

    if hist_plain.get("val_steps"):
        ax1.plot(
            hist_plain["val_steps"],
            hist_plain["val_ppls"],
            label="1. Plain MLP + Spectral (Champion)",
            color=c_plain,
            linewidth=2.4,
            marker="s",
            markersize=5,
            alpha=0.95,
        )
        min_pl = min(hist_plain["val_ppls"])
        min_pl_step = hist_plain["val_steps"][hist_plain["val_ppls"].index(min_pl)]
        ax1.scatter([min_pl_step], [min_pl], color=c_plain, s=120, zorder=5, edgecolors="black", linewidth=1.5)
        ax1.annotate(
            f"Best Champion: {min_pl:.2f} PPL",
            (min_pl_step, min_pl),
            textcoords="offset points",
            xytext=(-20, -20),
            fontweight="bold",
            color=c_plain,
            fontsize=10,
        )

    # Reference line for GPT-2 Small baseline (37.5 PPL on WikiText-103)
    ax1.axhline(37.5, color=c_gpt2, linestyle="--", linewidth=1.4, alpha=0.8, label="OpenAI GPT-2 Small Baseline (37.50 PPL)")

    ax1.set_title("Validation Perplexity (PPL) vs Optimizer Step", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Optimizer Steps (Effective Batch = 48-96 sequences)", fontsize=11)
    ax1.set_ylabel("Validation Perplexity (PPL)", fontsize=11)
    ax1.set_ylim(bottom=max(15, min(min_swi if 'min_swi' in locals() else 30, min_pl if 'min_pl' in locals() else 30) - 10))
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 2: Validation Cross-Entropy Loss vs Step
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")

    if hist_swiglu.get("val_steps"):
        ax2.plot(
            hist_swiglu["val_steps"],
            hist_swiglu["val_losses"],
            label="SwiGLU Baseline",
            color=c_swiglu,
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    if hist_plain.get("val_steps"):
        ax2.plot(
            hist_plain["val_steps"],
            hist_plain["val_losses"],
            label="Plain MLP + Spectral Barrier",
            color=c_plain,
            linewidth=2.4,
            marker="s",
            markersize=4,
        )

    # Highlight delta between curves if both present
    if hist_swiglu.get("val_steps") and hist_plain.get("val_steps"):
        common_steps = sorted(list(set(hist_swiglu["val_steps"]).intersection(set(hist_plain["val_steps"]))))
        if common_steps:
            s_map = dict(zip(hist_swiglu["val_steps"], hist_swiglu["val_losses"]))
            p_map = dict(zip(hist_plain["val_steps"], hist_plain["val_losses"]))
            y1 = [s_map[s] for s in common_steps]
            y2 = [p_map[s] for s in common_steps]
            ax2.fill_between(common_steps, y1, y2, color=c_plain, alpha=0.15, label="Δ Cross-Entropy")

    ax2.set_title("Validation Cross-Entropy Loss vs Step", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Optimizer Steps", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 3: Training Loss Dynamics (Smoothed)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")

    def smooth(vals, window=5):
        if len(vals) < window:
            return vals
        import numpy as np
        return np.convolve(vals, np.ones(window)/window, mode="valid").tolist()

    if hist_swiglu.get("train_steps"):
        steps = hist_swiglu["train_steps"]
        raw = hist_swiglu["train_losses"]
        sm = smooth(raw, window=5)
        ax3.plot(steps[:len(raw)], raw, color=c_swiglu, alpha=0.25, linewidth=1.0)
        ax3.plot(steps[len(steps)-len(sm):], sm, label="SwiGLU Train Loss (Smoothed)", color=c_swiglu, linewidth=2.0)

    if hist_plain.get("train_steps"):
        steps = hist_plain["train_steps"]
        raw = hist_plain["train_losses"]
        sm = smooth(raw, window=5)
        ax3.plot(steps[:len(raw)], raw, color=c_plain, alpha=0.25, linewidth=1.0)
        ax3.plot(steps[len(steps)-len(sm):], sm, label="Plain+Spectral Train Loss (Smoothed)", color=c_plain, linewidth=2.0)

    ax3.set_title("Training Loss Dynamics & Stability", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Optimizer Steps", fontsize=11)
    ax3.set_ylabel("Training Cross-Entropy Loss", fontsize=11)
    ax3.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 4: Learning Rate Decay & Convergence Points
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")

    if hist_swiglu.get("train_steps") and hist_swiglu.get("lrs"):
        ax4.plot(hist_swiglu["train_steps"], hist_swiglu["lrs"], label="Learning Rate Schedule (Cosine)", color="#2A9D8F", linewidth=2.0)

    if hist_swiglu.get("val_steps"):
        min_s = min(hist_swiglu["val_ppls"])
        min_s_step = hist_swiglu["val_steps"][hist_swiglu["val_ppls"].index(min_s)]
        ax4.axvline(min_s_step, color=c_swiglu, linestyle="--", linewidth=1.5, label=f"SwiGLU Best Step ({min_s_step})")

    if hist_plain.get("val_steps"):
        min_p = min(hist_plain["val_ppls"])
        min_p_step = hist_plain["val_steps"][hist_plain["val_ppls"].index(min_p)]
        ax4.axvline(min_p_step, color=c_plain, linestyle="-.", linewidth=1.8, label=f"Champion Best Step ({min_p_step})")

    ax4.set_title("Learning Rate Decay & Plateau Convergence Steps", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Optimizer Steps", fontsize=11)
    ax4.set_ylabel("Learning Rate", fontsize=11)
    ax4.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("114M Banach Contraction Transformer vs SwiGLU Baseline on Kaggle 2x Tesla T4\nWikiText-103 Convergence Comparison", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"[✓] Saved comparison figure to {save_path}")

    if show_plot:
        plt.show()


def print_executive_summary(hist_swiglu: Dict[str, Any], hist_plain: Dict[str, Any]):
    """Prints formatted comparison table and delta metrics."""
    swi_best_ppl = min(hist_swiglu["val_ppls"]) if hist_swiglu.get("val_ppls") else None
    swi_best_loss = min(hist_swiglu["val_losses"]) if hist_swiglu.get("val_losses") else None
    swi_best_step = hist_swiglu["val_steps"][hist_swiglu["val_ppls"].index(swi_best_ppl)] if swi_best_ppl else None

    pl_best_ppl = min(hist_plain["val_ppls"]) if hist_plain.get("val_ppls") else None
    pl_best_loss = min(hist_plain["val_losses"]) if hist_plain.get("val_losses") else None
    pl_best_step = hist_plain["val_steps"][hist_plain["val_ppls"].index(pl_best_ppl)] if pl_best_ppl else None

    print("\n" + "=" * 105)
    print("  EXECUTIVE COMPARISON MATRIX: 114M BANACH CONTRACTION VS SWIGLU BASELINE ON WIKITEXT-103")
    print("=" * 105)
    print(f"  {'Metric':<35} | {'0. SwiGLU Baseline':<28} | {'1. Plain MLP + Spectral':<32}")
    print("-" * 105)
    print(f"  {'Total Parameters':<35} | {'114,114,048':<28} | {'114,114,048 (Matched 1.5x)':<32}")
    print(f"  {'Non-Embedding Backbone Params':<35} | {'75,516,672':<28} | {'75,516,672':<32}")
    print(f"  {'Context Window':<35} | {'512 tokens':<28} | {'512 tokens':<32}")
    print(f"  {'Hardware Target':<35} | {'2x Tesla T4 (16GB VRAM)':<28} | {'2x Tesla T4 (16GB VRAM)':<32}")
    
    s_loss_str = f"{swi_best_loss:.4f}" if swi_best_loss else "N/A"
    p_loss_str = f"{pl_best_loss:.4f}" if pl_best_loss else "N/A"
    print(f"  {'Best Validation Cross-Entropy':<35} | {s_loss_str:<28} | {p_loss_str:<32}")

    s_ppl_str = f"{swi_best_ppl:.2f}" if swi_best_ppl else "N/A"
    p_ppl_str = f"{pl_best_ppl:.2f}" if pl_best_ppl else "N/A"
    print(f"  {'Best Validation Perplexity (PPL)':<35} | {s_ppl_str:<28} | {p_ppl_str:<32}")

    s_step_str = f"Step {swi_best_step}" if swi_best_step else "N/A"
    p_step_str = f"Step {pl_best_step}" if pl_best_step else "N/A"
    print(f"  {'Convergence Step':<35} | {s_step_str:<28} | {p_step_str:<32}")

    if swi_best_ppl and pl_best_ppl:
        delta_ppl = pl_best_ppl - swi_best_ppl
        rel_imp = (swi_best_ppl - pl_best_ppl) / swi_best_ppl * 100.0
        print("-" * 105)
        print(f"  {'Delta PPL (Champion - Baseline)':<35} | {'0.00 (Reference)':<28} | {delta_ppl:<+32.2f}")
        print(f"  {'Relative Perplexity Improvement':<35} | {'0.00%':<28} | {rel_imp:<+31.2f}%")
        print(f"  {'OpenAI GPT-2 Small Baseline Delta':<35} | {37.50 - swi_best_ppl:<+28.2f} PPL | {37.50 - pl_best_ppl:<+32.2f} PPL")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Build Comparison Curves and Difference Matrix for 114M DDP Models")
    parser.add_argument("--dir", type=str, default="checkpoints_ddp", help="Directory containing history JSONs or checkpoints")
    parser.add_argument("--save", type=str, default="checkpoints_ddp/comparison_curves.png", help="Path to save output plot")
    args = parser.parse_args()

    hist_swiglu = load_architecture_history(args.dir, "swiglu_baseline")
    hist_plain = load_architecture_history(args.dir, "plain_spectral")

    print(f"[*] Loaded SwiGLU history: {len(hist_swiglu.get('val_steps', []))} val checks, {len(hist_swiglu.get('train_steps', []))} train logs")
    print(f"[*] Loaded Plain+Spectral history: {len(hist_plain.get('val_steps', []))} val checks, {len(hist_plain.get('train_steps', []))} train logs")

    print_executive_summary(hist_swiglu, hist_plain)
    plot_comparison_curves(hist_swiglu, hist_plain, save_path=args.save, show_plot=False)


if __name__ == "__main__":
    main()
