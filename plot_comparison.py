import os
import re
import json
import math
import argparse
from typing import Dict, Any, Optional, Tuple, List

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
    """Loads history saved directly inside checkpoint, with fallback to best_val scalar."""
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "history" in ckpt:
                    return ckpt["history"]
                # Fallback: extract summary endpoint from checkpoint metadata
                if "best_val" in ckpt and "step" in ckpt:
                    bv = float(ckpt["best_val"])
                    st = int(ckpt["step"])
                    return {
                        "train_steps": [st],
                        "train_losses": [bv],
                        "lrs": [],
                        "val_steps": [st],
                        "val_losses": [bv],
                        "val_ppls": [math.exp(min(bv, 20))],
                    }
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
    train_pattern = re.compile(r"\[Step\s+(\d+)/\d+\]\s+Loss:\s+([\d\.]+).*?\|\s+LR:\s+([\d\.e\-\+]+)")
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


def load_all_histories_from_log(log_path_or_text: str) -> Dict[str, Dict[str, Any]]:
    """Splits a combined multi-architecture DDP log into per-architecture histories."""
    if os.path.exists(log_path_or_text):
        with open(log_path_or_text, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = log_path_or_text

    # Split by STARTING DDP TRAINING: <NAME>
    sections = re.split(r"STARTING DDP TRAINING:\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE)
    results = {}
    if len(sections) > 1:
        for i in range(1, len(sections), 2):
            raw_arch = sections[i].lower().strip()
            if "swiglu_baseline" in raw_arch:
                arch_key = "swiglu_baseline"
            elif "plain_spectral" in raw_arch:
                arch_key = "plain_spectral"
            elif "swiglu_aux" in raw_arch:
                arch_key = "swiglu_aux"
            else:
                arch_key = raw_arch
            sec_text = sections[i + 1]
            hist = parse_history_from_log_text(sec_text)
            if len(hist.get("val_steps", [])) > 0:
                results[arch_key] = hist
    return results


def load_architecture_history(output_dir: str, arch_name: str, fallback_log_text: Optional[str] = None) -> Dict[str, Any]:
    """Tries JSON -> Checkpoint -> Log files -> Fallback text in order of preference."""
    # 1. JSON history
    json_path = os.path.join(output_dir, f"{arch_name}_history.json")
    hist = parse_history_from_json(json_path)
    if hist and len(hist.get("val_steps", [])) > 0:
        return hist

    # 2. Combined training.log on disk
    comb_log = os.path.join(output_dir, "training.log")
    if os.path.exists(comb_log):
        all_h = load_all_histories_from_log(comb_log)
        if arch_name in all_h and len(all_h[arch_name].get("val_steps", [])) > 0:
            return all_h[arch_name]

    # 3. Individual arch log file on disk
    log_file = os.path.join(output_dir, f"{arch_name}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            hist = parse_history_from_log_text(f.read())
            if len(hist.get("val_steps", [])) > 0:
                return hist

    # 4. Provided fallback log text
    if fallback_log_text:
        all_h = load_all_histories_from_log(fallback_log_text)
        if arch_name in all_h and len(all_h[arch_name].get("val_steps", [])) > 0:
            return all_h[arch_name]
        hist = parse_history_from_log_text(fallback_log_text)
        if len(hist.get("val_steps", [])) > 0:
            return hist

    # 5. Checkpoint metadata fallback
    ckpt_path = os.path.join(output_dir, f"{arch_name}_114m_best.pt")
    hist = parse_history_from_checkpoint(ckpt_path)
    if hist and len(hist.get("val_steps", [])) > 0:
        return hist

    return {"train_steps": [], "train_losses": [], "lrs": [], "val_steps": [], "val_losses": [], "val_ppls": []}


def plot_comparison_curves(
    hist_swiglu_or_dict: Any,
    hist_plain: Optional[Dict[str, Any]] = None,
    hist_swiglu_aux: Optional[Dict[str, Any]] = None,
    save_path: str = "checkpoints_ddp/comparison_curves.png",
    show_plot: bool = True,
):
    """Generates 2x2 publication-quality comparison figure across all available models."""
    if isinstance(hist_swiglu_or_dict, dict) and "val_steps" not in hist_swiglu_or_dict:
        arch_map = hist_swiglu_or_dict
    else:
        arch_map = {}
        if hist_swiglu_or_dict and hist_swiglu_or_dict.get("val_steps"):
            arch_map["0. SwiGLU Baseline"] = hist_swiglu_or_dict
        if hist_plain and hist_plain.get("val_steps"):
            arch_map["1. Plain MLP + Spectral"] = hist_plain
        if hist_swiglu_aux and hist_swiglu_aux.get("val_steps"):
            arch_map["2. SwiGLU + Aux"] = hist_swiglu_aux

    if not arch_map:
        print("[!] No architecture histories with validation steps found to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    fig.patch.set_facecolor("#FAFAFA")

    palette = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#9B5DE5", "#E76F51"]
    markers = ["o", "s", "^", "D", "v", "P"]

    def smooth(vals, window=5):
        if len(vals) < window:
            return vals
        import numpy as np
        return np.convolve(vals, np.ones(window)/window, mode="valid").tolist()

    all_min_ppl = []

    # -------------------------------------------------------------------------
    # Subplot 1: Validation Perplexity (PPL) vs Step
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")

    for idx, (name, h) in enumerate(arch_map.items()):
        if not h.get("val_steps"):
            continue
        c = palette[idx % len(palette)]
        m = markers[idx % len(markers)]
        v_steps = h["val_steps"]
        v_ppls = h["val_ppls"]
        min_p = min(v_ppls)
        all_min_ppl.append(min_p)
        min_p_step = v_steps[v_ppls.index(min_p)]

        if len(v_steps) > 1:
            ax1.plot(v_steps, v_ppls, label=f"{name} (Best: {min_p:.2f})", color=c, linewidth=2.2, marker=m, markersize=4, alpha=0.9)
        ax1.scatter([min_p_step], [min_p], color=c, s=110, zorder=5, edgecolors="black", linewidth=1.2, label=f"{name} Best ({min_p:.2f})" if len(v_steps) == 1 else None)

    # Reference line for GPT-2 Small baseline (37.5 PPL on WikiText-103)
    ax1.axhline(37.5, color="#6C757D", linestyle="--", linewidth=1.4, alpha=0.8, label="OpenAI GPT-2 Small (37.50 PPL)")
    ax1.set_title("Validation Perplexity (PPL) vs Step", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Optimizer Steps", fontsize=11)
    ax1.set_ylabel("Validation Perplexity (PPL)", fontsize=11)
    if all_min_ppl:
        ax1.set_ylim(bottom=max(15, min(all_min_ppl) - 8))
    h_l, l_l = ax1.get_legend_handles_labels()
    if h_l:
        ax1.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 2: Validation Cross-Entropy Loss vs Step
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")

    for idx, (name, h) in enumerate(arch_map.items()):
        if not h.get("val_steps"):
            continue
        c = palette[idx % len(palette)]
        m = markers[idx % len(markers)]
        if len(h["val_steps"]) > 1:
            ax2.plot(h["val_steps"], h["val_losses"], label=name, color=c, linewidth=2.2, marker=m, markersize=4)
        else:
            ax2.scatter(h["val_steps"], h["val_losses"], color=c, s=110, zorder=5, edgecolors="black", label=name)

    ax2.set_title("Validation Cross-Entropy Loss vs Step", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Optimizer Steps", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
    h_l, l_l = ax2.get_legend_handles_labels()
    if h_l:
        ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 3: Training Loss Dynamics (Smoothed)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")

    for idx, (name, h) in enumerate(arch_map.items()):
        if not h.get("train_steps"):
            continue
        c = palette[idx % len(palette)]
        steps = h["train_steps"]
        raw = h["train_losses"]
        if len(raw) > 5:
            sm = smooth(raw, window=5)
            ax3.plot(steps[:len(raw)], raw, color=c, alpha=0.18, linewidth=0.8)
            ax3.plot(steps[len(steps)-len(sm):], sm, label=f"{name} (Train)", color=c, linewidth=2.0)
        else:
            ax3.scatter(steps, raw, color=c, label=f"{name} (Train)")

    ax3.set_title("Training Loss Dynamics & Stability", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Optimizer Steps", fontsize=11)
    ax3.set_ylabel("Training Cross-Entropy Loss", fontsize=11)
    h_l, l_l = ax3.get_legend_handles_labels()
    if h_l:
        ax3.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 4: Learning Rate Schedule & Convergence Steps
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")

    for idx, (name, h) in enumerate(arch_map.items()):
        if h.get("train_steps") and h.get("lrs") and idx == 0:
            ax4.plot(h["train_steps"], h["lrs"], label="Learning Rate (Cosine Annealing)", color="#457B9D", linewidth=2.0)
        if h.get("val_steps") and h.get("val_ppls"):
            c = palette[idx % len(palette)]
            min_p = min(h["val_ppls"])
            min_step = h["val_steps"][h["val_ppls"].index(min_p)]
            ax4.axvline(min_step, color=c, linestyle="--", linewidth=1.5, label=f"{name} Plateau (@ {min_step})")

    ax4.set_title("Learning Rate Decay & Convergence Points", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Optimizer Steps", fontsize=11)
    ax4.set_ylabel("Learning Rate", fontsize=11)
    h_l, l_l = ax4.get_legend_handles_labels()
    if h_l:
        ax4.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("114M Parameter Model Comparison on WikiText-103 (Kaggle 2x Tesla T4)\nEmpirical Perplexity & Convergence Trajectories", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"[✓] Saved comparison figure to {save_path}")

    if show_plot:
        plt.show()


def print_executive_summary(hist_swiglu_or_dict: Any, hist_plain: Optional[Dict[str, Any]] = None, hist_swiglu_aux: Optional[Dict[str, Any]] = None):
    """Prints formatted comparison table across all models."""
    if isinstance(hist_swiglu_or_dict, dict) and "val_steps" not in hist_swiglu_or_dict:
        arch_map = hist_swiglu_or_dict
    else:
        arch_map = {}
        if hist_swiglu_or_dict and hist_swiglu_or_dict.get("val_steps"):
            arch_map["0. SwiGLU Baseline"] = hist_swiglu_or_dict
        if hist_plain and hist_plain.get("val_steps"):
            arch_map["1. Plain MLP + Spectral"] = hist_plain
        if hist_swiglu_aux and hist_swiglu_aux.get("val_steps"):
            arch_map["2. SwiGLU + Aux"] = hist_swiglu_aux

    if not arch_map:
        return

    print("\n" + "=" * 115)
    print("  EXECUTIVE COMPARISON MATRIX: 114M MODELS ON WIKITEXT-103")
    print("=" * 115)
    print(f"  {'Metric':<32} | " + " | ".join([f"{name:<24}" for name in arch_map]))
    print("-" * 115)

    best_ppls = {}
    best_losses = {}
    best_steps = {}

    for name, h in arch_map.items():
        if h.get("val_ppls"):
            bp = min(h["val_ppls"])
            best_ppls[name] = bp
            best_losses[name] = min(h["val_losses"])
            best_steps[name] = h["val_steps"][h["val_ppls"].index(bp)]
        else:
            best_ppls[name] = None
            best_losses[name] = None
            best_steps[name] = None

    def fmt_cell(val, is_float=True, is_step=False):
        if val is None:
            return f"{'N/A':<24}"
        if is_step:
            return f"Step {val:<19d}"
        if is_float:
            return f"{val:<24.4f}"
        return f"{val:<24.2f}"

    print(f"  {'Best Validation Cross-Entropy':<32} | " + " | ".join([fmt_cell(best_losses[name], is_float=True) for name in arch_map]))
    print(f"  {'Best Validation Perplexity (PPL)':<32} | " + " | ".join([fmt_cell(best_ppls[name], is_float=False) for name in arch_map]))
    print(f"  {'Convergence Step':<32} | " + " | ".join([fmt_cell(best_steps[name], is_step=True) for name in arch_map]))

    base_ppl = list(best_ppls.values())[0]
    if base_ppl is not None:
        print("-" * 115)
        print(f"  {'Delta PPL vs Baseline':<32} | " + " | ".join([f"{best_ppls[name] - base_ppl:<+24.2f}" if best_ppls[name] is not None else f"{'N/A':<24}" for name in arch_map]))
        print(f"  {'GPT-2 Small Delta (37.50 ref)':<32} | " + " | ".join([f"{37.50 - best_ppls[name]:<+20.2f} PPL" if best_ppls[name] is not None else f"{'N/A':<24}" for name in arch_map]))
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Build Comparison Curves for 114M DDP Models")
    parser.add_argument("--dir", type=str, default="checkpoints_ddp", help="Directory containing history JSONs or checkpoints")
    parser.add_argument("--save", type=str, default="checkpoints_ddp/comparison_curves.png", help="Path to save output plot")
    args = parser.parse_args()

    archs = ["swiglu_baseline", "plain_spectral", "swiglu_aux", "plain_spectral_cycle"]
    histories = {}
    for a in archs:
        h = load_architecture_history(args.dir, a)
        if h and len(h.get("val_steps", [])) > 0:
            histories[a] = h

    print(f"[*] Found {len(histories)} trained architectures to plot: {list(histories.keys())}")
    print_executive_summary(histories)
    plot_comparison_curves(histories, save_path=args.save, show_plot=False)


if __name__ == "__main__":
    main()
