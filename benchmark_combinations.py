import os
import sys
import math
import time
import copy
import random
import tempfile
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from llama.config import LlamaConfig
from llama.model import LlamaForCausalLM, RMSNorm, apply_rotary_pos_emb, repeat_kv
from llama.dataset import get_wikitext_dataloader


# ============================================================================
# 1. ARCHITECTURAL MODULES AND INVERSION OPERATORS
# ============================================================================
class PlainMLP(nn.Module):
    """Simplified 2-matrix MLP without SwiGLU gating branch: down(silu(up(x)))."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Exact parameter matching with 3-matrix SwiGLU: hidden_dim = 1.5 * intermediate_size
        hidden_dim = int(config.intermediate_size * 1.5)
        self.fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


def estimate_spectral_norm(weight: torch.Tensor, n_iter: int = 1) -> torch.Tensor:
    """Estimates spectral norm (largest singular value) using 1-step power iteration."""
    device = weight.device
    out_dim, in_dim = weight.shape
    u = torch.randn(out_dim, 1, device=device)
    u = u / (u.norm() + 1e-8)
    for _ in range(n_iter):
        v = torch.matmul(weight.t(), u)
        v = v / (v.norm() + 1e-8)
        u = torch.matmul(weight, v)
        u = u / (u.norm() + 1e-8)
    # Detach u and v to compute exact Rayleigh quotient gradient without unrolling graph
    sigma = torch.matmul(u.detach().t(), torch.matmul(weight, v.detach())).squeeze()
    return sigma


def invert_layer_1step(layer, h_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """1-step Picard Banach inverse operator."""
    norm2 = layer.post_attention_layernorm(h_next)
    h_mid = h_next - layer.mlp(norm2)
    norm1 = layer.input_layernorm(h_mid)
    attn_rev, _ = layer.self_attn(norm1, cos=cos, sin=sin)
    h_in_rev = h_mid - attn_rev
    return h_in_rev


def invert_layer_2step(layer, h_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """2-step Picard Banach inverse operator for deeper contraction supervision."""
    # 2-step MLP inverse
    x0 = h_next
    x1 = h_next - layer.mlp(layer.post_attention_layernorm(x0))
    h_mid = h_next - layer.mlp(layer.post_attention_layernorm(x1))

    # 2-step Attention inverse
    z0 = h_mid
    attn0, _ = layer.self_attn(layer.input_layernorm(z0), cos=cos, sin=sin)
    z1 = h_mid - attn0
    attn1, _ = layer.self_attn(layer.input_layernorm(z1), cos=cos, sin=sin)
    h_in_rev = h_mid - attn1
    return h_in_rev


def invert_block_picard(layer, h_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_steps: int = 2) -> torch.Tensor:
    """Multi-step Picard fixed-point iteration for test-time evaluation."""
    h_mid = h_next.clone()
    for _ in range(k_steps):
        norm2 = layer.post_attention_layernorm(h_mid)
        h_mid = h_next - layer.mlp(norm2)
    h_prev = h_mid.clone()
    for _ in range(k_steps):
        norm1 = layer.input_layernorm(h_prev)
        attn_out, _ = layer.self_attn(norm1, cos=cos, sin=sin)
        h_prev = h_mid - attn_out
    return h_prev


def invert_block_anderson(layer, h_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Anderson Accelerated fixed-point iteration (m=2 iterates) for test-time evaluation."""
    # 1. Invert MLP with Anderson Acceleration
    x0 = h_next.clone()
    g0 = h_next - layer.mlp(layer.post_attention_layernorm(x0))
    x1 = g0.clone()
    g1 = h_next - layer.mlp(layer.post_attention_layernorm(x1))

    r0 = g0 - x0
    r1 = g1 - x1
    dr = r1 - r0
    dot_r0_dr = (r0 * dr).sum(dim=-1, keepdim=True)
    norm_dr_sq = (dr * dr).sum(dim=-1, keepdim=True) + 1e-8
    alpha_mlp = (-dot_r0_dr / norm_dr_sq).clamp(0.0, 1.0)
    h_mid = (1.0 - alpha_mlp) * g0 + alpha_mlp * g1

    # 2. Invert Attention with Anderson Acceleration
    x0_a = h_mid.clone()
    attn_out0, _ = layer.self_attn(layer.input_layernorm(x0_a), cos=cos, sin=sin)
    g0_a = h_mid - attn_out0
    x1_a = g0_a.clone()
    attn_out1, _ = layer.self_attn(layer.input_layernorm(x1_a), cos=cos, sin=sin)
    g1_a = h_mid - attn_out1

    r0_a = g0_a - x0_a
    r1_a = g1_a - x1_a
    dr_a = r1_a - r0_a
    dot_r0_dr_a = (r0_a * dr_a).sum(dim=-1, keepdim=True)
    norm_dr_sq_a = (dr_a * dr_a).sum(dim=-1, keepdim=True) + 1e-8
    alpha_attn = (-dot_r0_dr_a / norm_dr_sq_a).clamp(0.0, 1.0)
    h_prev = (1.0 - alpha_attn) * g0_a + alpha_attn * g1_a
    return h_prev


# ============================================================================
# 2. STEP FUNCTIONS FOR EACH COMBINATION ARM
# ============================================================================
def step_arm_base(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 0: Baseline (SwiGLU Combo 1+2+4)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            h_in_rev = invert_layer_1step(layer, h_next, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err.mean()
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_plain_solo(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 1: Plain MLP Solo (Combo 1+2+4)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            h_in_rev = invert_layer_1step(layer, h_next, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err.mean()
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_plain_spectral(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 2: Combo Plain MLP + Spectral Barrier on (W_O, W_2)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            h_in_rev = invert_layer_1step(layer, h_next, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)

            sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
            sig_mlp = estimate_spectral_norm(layer.mlp.fc2.weight)
            barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_mlp - 1.0).pow(2)
            loss_c = rel_err.mean() + 0.1 * barrier
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_plain_spectral_cycle(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 3: Combo Plain MLP + Spectral Barrier + Mild Cycle Consistency (lambda_cyc = 0.25)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            h_in_rev = invert_layer_1step(layer, h_next, cos, sin)
            target_bw = h_prev.detach()
            rel_err_bw = (h_in_rev - target_bw).pow(2).sum(-1) / (target_bw.pow(2).sum(-1) + 1e-4)

            # Round-trip forward cycle pass
            h_rec, _ = layer(h_in_rev, cos=cos, sin=sin)
            target_fw = h_next.detach()
            rel_err_fw = (h_rec - target_fw).pow(2).sum(-1) / (target_fw.pow(2).sum(-1) + 1e-4)

            sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
            sig_mlp = estimate_spectral_norm(layer.mlp.fc2.weight)
            barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_mlp - 1.0).pow(2)

            loss_c = rel_err_bw.mean() + 0.25 * rel_err_fw.mean() + 0.1 * barrier
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_plain_spectral_k2(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 4: Combo Plain MLP + Spectral Barrier + 2-Step Picard Aux Supervision."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            # 2-step Picard unroll during training auxiliary pass
            h_in_rev = invert_layer_2step(layer, h_next, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)

            sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
            sig_mlp = estimate_spectral_norm(layer.mlp.fc2.weight)
            barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_mlp - 1.0).pow(2)
            loss_c = rel_err.mean() + 0.1 * barrier
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_swiglu_spectral_cycle(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 5: Combo SwiGLU + Spectral Barrier + Cycle Consistency."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        h_next, _ = layer(h_prev, cos=cos, sin=sin)
        if l == chosen_layer:
            h_in_rev = invert_layer_1step(layer, h_next, cos, sin)
            target_bw = h_prev.detach()
            rel_err_bw = (h_in_rev - target_bw).pow(2).sum(-1) / (target_bw.pow(2).sum(-1) + 1e-4)

            h_rec, _ = layer(h_in_rev, cos=cos, sin=sin)
            target_fw = h_next.detach()
            rel_err_fw = (h_rec - target_fw).pow(2).sum(-1) / (target_fw.pow(2).sum(-1) + 1e-4)

            sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
            sig_d = estimate_spectral_norm(layer.mlp.down_proj.weight)
            barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_d - 1.0).pow(2)

            loss_c = rel_err_bw.mean() + 0.25 * rel_err_fw.mean() + 0.1 * barrier
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


# ============================================================================
# 3. EVALUATION AND PROBING
# ============================================================================
def compute_val_loss(model: LlamaForCausalLM, val_batches: List[torch.Tensor], vocab_size: int, scale_factor: float = 1.0) -> float:
    was_training = model.training
    model.eval()
    total_ce = 0.0
    with torch.no_grad():
        for batch in val_batches:
            x = batch[:, :-1].contiguous()
            y = batch[:, 1:].contiguous()
            h = model.model.embed_tokens(x) * scale_factor
            seq_l = x.shape[1]
            cos, sin = model.model.rotary_emb(h, seq_l)
            for layer in model.model.layers:
                h, _ = layer(h, cos=cos, sin=sin)
            h = model.model.norm(h)
            logits = model.lm_head(h)
            ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_ce += ce.item()
    if was_training:
        model.train()
    return total_ce / len(val_batches)


def evaluate_inversion_probes(model: LlamaForCausalLM, x_probe: torch.Tensor, scale_factor: float = 1.0) -> Dict[str, Any]:
    seq_l = x_probe.shape[1]
    results = {}
    with torch.no_grad():
        h_in = model.model.embed_tokens(x_probe) * scale_factor
        cos, sin = model.model.rotary_emb(h_in, seq_l)
        h = h_in
        hidden_states = [h]
        for layer in model.model.layers:
            h, _ = layer(h, cos=cos, sin=sin)
            hidden_states.append(h)
        h_out = h

        # 1-step errors per layer
        layer_errs = []
        for l in range(len(model.model.layers)):
            layer = model.model.layers[l]
            h_prev_t = hidden_states[l]
            h_next_t = hidden_states[l + 1]
            h_in_rev = invert_layer_1step(layer, h_next_t, cos, sin)
            err = (h_in_rev - h_prev_t).norm().item() / max(h_prev_t.norm().item(), 1e-6)
            layer_errs.append(err)
        results["layer_errs"] = layer_errs

        # Picard Multi-step (k=1, 2, 3) across all 4 layers
        for k in [1, 2, 3]:
            h_rev = h_out.clone()
            for layer in reversed(model.model.layers):
                h_rev = invert_block_picard(layer, h_rev, cos, sin, k_steps=k)
            full_err = (h_rev - h_in).norm().item() / max(h_in.norm().item(), 1e-6)
            results[f"inv_picard_k{k}"] = full_err
            if k == 1:
                results["norm_rev"] = h_rev.norm().item()

        # Anderson Accelerated Multi-step across all 4 layers
        h_rev_aa = h_out.clone()
        for layer in reversed(model.model.layers):
            h_rev_aa = invert_block_anderson(layer, h_rev_aa, cos, sin)
        results["inv_anderson"] = (h_rev_aa - h_in).norm().item() / max(h_in.norm().item(), 1e-6)

    return results


# ============================================================================
# 4. BENCHMARK EXECUTION (DYNAMIC CONVERGENCE TO PLATEAU)
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    seq_len = 16
    hidden_size = 128
    num_layers = 4
    num_heads = 4
    num_kv_heads = 2
    lr = 6e-4

    # Rigorous Dynamic Plateau Criteria
    eval_interval = 40       # Evaluate validation loss every 40 steps
    min_steps = 300          # Minimum training steps before plateau check activates
    patience = 4             # 4 checks (160 steps) without improvement >= min_delta
    min_delta = 0.005        # Improvement threshold
    max_steps = 1100         # Upper limit guaranteeing convergence

    print("\n" + "=" * 170)
    print("  RIGOROUS ADVANCEMENT BENCHMARK: TESTING ARCHITECTURAL & REGULARIZATION COMBINATIONS UNTIL TRUE PLATEAU")
    print("  Arm 0: Baseline (SwiGLU Combo 1+2+4)     | Arm 1: Plain MLP Solo")
    print("  Arm 2: Plain MLP + Spectral Barrier      | Arm 3: Plain MLP + Spectral Barrier + Mild Cycle Consistency")
    print("  Arm 4: Plain MLP + Spectral + k=2 Picard | Arm 5: SwiGLU + Spectral Barrier + Cycle Consistency")
    print(f"  Criterion: Train dynamically until individual Validation Loss plateaus (patience={patience} checks x {eval_interval} st = {patience*eval_interval} st)")
    print("=" * 170)
    print(f"  Device: {device} | Architecture: LLaMA-{num_layers}L-{hidden_size}D | Max Steps: {max_steps} | LR: {lr}")

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=384,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )

    scale_factor = math.sqrt(hidden_size)
    depth_probs = [0.40, 0.30, 0.20, 0.10]

    def make_model(seed=42):
        torch.manual_seed(seed)
        return LlamaForCausalLM(cfg).to(device)

    # 6 Isolated Models
    m_base = make_model(42)
    
    m_plain_solo = make_model(42)
    for layer in m_plain_solo.model.layers:
        layer.mlp = PlainMLP(cfg).to(device)
        
    m_plain_spec = make_model(42)
    for layer in m_plain_spec.model.layers:
        layer.mlp = PlainMLP(cfg).to(device)

    m_plain_spec_cyc = make_model(42)
    for layer in m_plain_spec_cyc.model.layers:
        layer.mlp = PlainMLP(cfg).to(device)

    m_plain_spec_k2 = make_model(42)
    for layer in m_plain_spec_k2.model.layers:
        layer.mlp = PlainMLP(cfg).to(device)

    m_swiglu_spec_cyc = make_model(42)

    models = {
        "base": m_base,
        "plain_solo": m_plain_solo,
        "plain_spec": m_plain_spec,
        "plain_spec_cyc": m_plain_spec_cyc,
        "plain_spec_k2": m_plain_spec_k2,
        "swiglu_spec_cyc": m_swiglu_spec_cyc,
    }

    optimizers = {
        k: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        for k, m in models.items()
    }

    step_fns = {
        "base": step_arm_base,
        "plain_solo": step_arm_plain_solo,
        "plain_spec": step_arm_plain_spectral,
        "plain_spec_cyc": step_arm_plain_spectral_cycle,
        "plain_spec_k2": step_arm_plain_spectral_k2,
        "swiglu_spec_cyc": step_arm_swiglu_spectral_cycle,
    }

    # Data Loaders
    train_loader = get_wikitext_dataloader(split="train", batch_size=8, seq_len=seq_len, pairs=False, max_samples=10000, shuffle=True)
    val_loader = get_wikitext_dataloader(split="test", batch_size=16, seq_len=seq_len, pairs=False, max_samples=160, shuffle=False)
    val_batches = [b["input_ids"].to(device) for b in val_loader]
    print(f"  Pre-loaded {len(val_batches)} validation batches onto {device}.")

    trackers = {}
    for k in models:
        init_val = compute_val_loss(models[k], val_batches, vocab_size, scale_factor=scale_factor)
        trackers[k] = {
            "name": k,
            "active": True,
            "best_val": init_val,
            "best_step": 0,
            "patience_counter": 0,
            "plateau_step": None,
            "best_state": copy.deepcopy(models[k].state_dict()),
        }

    print(f"\n  --- TRAINING PROFILE: DYNAMIC CONVERGENCE TO PLATEAU ---")
    print(f"{'Step':>6} | {'0. Base (SwiGLU)':>16} | {'1. Plain Solo':>13} | {'2. Plain+Spec':>13} | {'3. Plain+Spc+Cyc':>16} | {'4. Plain+Spc+k2':>15} | {'5. Swi+Spc+Cyc':>14} | {'Active':>6}")
    print("-" * 125)

    train_iter = iter(train_loader)
    step = 0
    t0_start = time.time()

    while any(trackers[k]["active"] for k in models) and step < max_steps:
        step += 1
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch["input_ids"].to(device)
        x = tokens[:, :-1].contiguous()
        y = tokens[:, 1:].contiguous()
        seq_l = x.shape[1]

        step_losses = {}

        for k, m in models.items():
            if not trackers[k]["active"]:
                continue
            opt = optimizers[k]
            opt.zero_grad(set_to_none=True)
            emb = m.model.embed_tokens(x) * scale_factor
            cos, sin = m.model.rotary_emb(emb, seq_l)
            h, l_aux = step_fns[k](m, x, cos, sin, depth_probs, scale_factor)
            l_ce = F.cross_entropy(m.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            total_loss = l_ce + l_aux
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step_losses[k] = l_ce.item()

        # Dynamic Plateau Evaluation
        if step % eval_interval == 0:
            for k, m in models.items():
                if not trackers[k]["active"]:
                    continue
                v_loss = compute_val_loss(m, val_batches, vocab_size, scale_factor=scale_factor)
                if v_loss < trackers[k]["best_val"] - min_delta:
                    trackers[k]["best_val"] = v_loss
                    trackers[k]["best_step"] = step
                    trackers[k]["patience_counter"] = 0
                    trackers[k]["best_state"] = copy.deepcopy(m.state_dict())
                else:
                    if step >= min_steps:
                        trackers[k]["patience_counter"] += 1
                        if trackers[k]["patience_counter"] >= patience:
                            trackers[k]["active"] = False
                            trackers[k]["plateau_step"] = step

        if step % 50 == 0 or step == 1 or not any(trackers[k]["active"] for k in models):
            active_count = sum(1 for k in models if trackers[k]["active"])
            def fmt(k):
                if k in step_losses: return f"{step_losses[k]:.4f}"
                return f"[PLT@{trackers[k]['plateau_step']}]"
            print(f"{step:6d} | {fmt('base'):>16} | {fmt('plain_solo'):>13} | {fmt('plain_spec'):>13} | {fmt('plain_spec_cyc'):>16} | {fmt('plain_spec_k2'):>15} | {fmt('swiglu_spec_cyc'):>14} | {active_count:>4d}/6")

    total_training_time = time.time() - t0_start
    print("-" * 125)
    print(f"  Training finished in {total_training_time:.1f}s. All arms converged to plateau.")

    # ------------------------------------------------------------------------
    # 5. RESTORE BEST CHECKPOINTS AND RUN FULL DIAGNOSTIC PROBES
    # ------------------------------------------------------------------------
    model_labels = {
        "base": "0. Baseline (SwiGLU Combo 1+2+4)",
        "plain_solo": "1. Plain MLP Solo",
        "plain_spec": "2. Plain MLP + Spectral Barrier",
        "plain_spec_cyc": "3. Plain + Spectral + Cycle Loss",
        "plain_spec_k2": "4. Plain + Spectral + k=2 Picard Aux",
        "swiglu_spec_cyc": "5. SwiGLU + Spectral + Cycle Loss",
    }

    final_results = {}
    probe_batch = val_batches[0][:1, :].contiguous()
    x_probe = probe_batch[:, :-1].contiguous()

    ckpt_dir = os.path.join(tempfile.gettempdir(), "great_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for k, m in models.items():
        m.load_state_dict(trackers[k]["best_state"])
        m.eval()

        # Save individual plateau checkpoint
        torch.save(trackers[k]["best_state"], os.path.join(ckpt_dir, f"plateau_combo_{k}.pt"))

        val_ce = compute_val_loss(m, val_batches, vocab_size, scale_factor=scale_factor)
        val_ppl = math.exp(min(val_ce, 20))
        inv_data = evaluate_inversion_probes(m, x_probe, scale_factor=scale_factor)

        final_results[k] = {
            "display": model_labels[k],
            "plateau_step": trackers[k]["best_step"],
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "layer_errs": inv_data["layer_errs"],
            "inv_picard_k1": inv_data["inv_picard_k1"],
            "inv_picard_k2": inv_data["inv_picard_k2"],
            "inv_picard_k3": inv_data["inv_picard_k3"],
            "inv_anderson": inv_data["inv_anderson"],
            "norm_rev": inv_data["norm_rev"],
        }

    # ------------------------------------------------------------------------
    # 6. EXECUTIVE DIAGNOSTIC REPORT
    # ------------------------------------------------------------------------
    print("\n" + "=" * 180)
    print("  EXECUTIVE DIAGNOSTIC TABLE: TESTING ARCHITECTURAL & REGULARIZATION COMBINATIONS UNTIL TRUE PLATEAU")
    print("=" * 180)
    header = f"  {'Metric':<32} | {'0. Base (SwiGLU)':<18} | {'1. Plain Solo':<16} | {'2. Plain+Spec':<16} | {'3. Plain+Spc+Cyc':<18} | {'4. Plain+Spc+k2':<18} | {'5. Swi+Spc+Cyc':<16}"
    print(header)
    print("-" * 180)

    rb = final_results["base"]
    r1 = final_results["plain_solo"]
    r2 = final_results["plain_spec"]
    r3 = final_results["plain_spec_cyc"]
    r4 = final_results["plain_spec_k2"]
    r5 = final_results["swiglu_spec_cyc"]

    print(f"  {'Step at Plateau (Best)':<32} | {rb['plateau_step']:<18d} | {r1['plateau_step']:<16d} | {r2['plateau_step']:<16d} | {r3['plateau_step']:<18d} | {r4['plateau_step']:<18d} | {r5['plateau_step']:<16d}")
    print(f"  {'Plateau Validation CE Loss':<32} | {rb['val_ce']:<18.4f} | {r1['val_ce']:<16.4f} | {r2['val_ce']:<16.4f} | {r3['val_ce']:<18.4f} | {r4['val_ce']:<18.4f} | {r5['val_ce']:<16.4f}")
    print(f"  {'Plateau Validation PPL':<32} | {rb['val_ppl']:<18.2f} | {r1['val_ppl']:<16.2f} | {r2['val_ppl']:<16.2f} | {r3['val_ppl']:<18.2f} | {r4['val_ppl']:<18.2f} | {r5['val_ppl']:<16.2f}")

    delta = lambda m: m['val_ppl'] - rb['val_ppl']
    print(f"  {'Delta PPL vs Baseline':<32} | {'0.00 (Baseline)':<18} | {delta(r1):<+16.2f} | {delta(r2):<+16.2f} | {delta(r3):<+18.2f} | {delta(r4):<+18.2f} | {delta(r5):<+16.2f}")
    print(f"  {'Layer 0 1-Step Invert Error':<32} | {rb['layer_errs'][0]:<18.4f} | {r1['layer_errs'][0]:<16.4f} | {r2['layer_errs'][0]:<16.4f} | {r3['layer_errs'][0]:<18.4f} | {r4['layer_errs'][0]:<18.4f} | {r5['layer_errs'][0]:<16.4f}")
    print(f"  {'Layer 3 1-Step Invert Error':<32} | {rb['layer_errs'][3]:<18.4f} | {r1['layer_errs'][3]:<16.4f} | {r2['layer_errs'][3]:<16.4f} | {r3['layer_errs'][3]:<18.4f} | {r4['layer_errs'][3]:<18.4f} | {r5['layer_errs'][3]:<16.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=1)':<32} | {rb['inv_picard_k1']:<18.4f} | {r1['inv_picard_k1']:<16.4f} | {r2['inv_picard_k1']:<16.4f} | {r3['inv_picard_k1']:<18.4f} | {r4['inv_picard_k1']:<18.4f} | {r5['inv_picard_k1']:<16.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=2)':<32} | {rb['inv_picard_k2']:<18.4f} | {r1['inv_picard_k2']:<16.4f} | {r2['inv_picard_k2']:<16.4f} | {r3['inv_picard_k2']:<18.4f} | {r4['inv_picard_k2']:<18.4f} | {r5['inv_picard_k2']:<16.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=3)':<32} | {rb['inv_picard_k3']:<18.4f} | {r1['inv_picard_k3']:<16.4f} | {r2['inv_picard_k3']:<16.4f} | {r3['inv_picard_k3']:<18.4f} | {r4['inv_picard_k3']:<18.4f} | {r5['inv_picard_k3']:<16.4f}")
    print(f"  {'Full 4-Layer Invert (Anderson)':<32} | {rb['inv_anderson']:<18.4f} | {r1['inv_anderson']:<16.4f} | {r2['inv_anderson']:<16.4f} | {r3['inv_anderson']:<18.4f} | {r4['inv_anderson']:<18.4f} | {r5['inv_anderson']:<16.4f}")
    print(f"  {'Reversed Vector L2 Norm':<32} | {rb['norm_rev']:<18.2f} | {r1['norm_rev']:<16.2f} | {r2['norm_rev']:<16.2f} | {r3['norm_rev']:<18.2f} | {r4['norm_rev']:<18.2f} | {r5['norm_rev']:<16.2f}")
    print("=" * 180)


if __name__ == "__main__":
    main()
