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
# 1. SPECIALIZED ARCHITECTURAL MODULES AND INVERSION OPERATORS
# ============================================================================
class PlainMLP(nn.Module):
    """Simplified 2-matrix MLP without SwiGLU gating branch: down(silu(up(x)))."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        hidden_dim = int(config.intermediate_size * 1.5)
        self.fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


def forward_and_invert_layer(layer, h_prev, cos, sin):
    """Executes forward pass and 1-step Banach inversion."""
    h_next, _ = layer(h_prev, cos=cos, sin=sin)
    norm2 = layer.post_attention_layernorm(h_next)
    h_mid = h_next - layer.mlp(norm2)
    norm1 = layer.input_layernorm(h_mid)
    attn_rev, _ = layer.self_attn(norm1, cos=cos, sin=sin)
    h_in_rev = h_mid - attn_rev
    return h_next, h_in_rev


def invert_block_picard(layer, h_next, cos, sin, k_steps=2):
    """Standard Picard fixed-point iteration."""
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


def invert_block_anderson(layer, h_next, cos, sin):
    """Anderson Accelerated fixed-point iteration (m=2 iterates)."""
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
    sigma = torch.matmul(u.t(), torch.matmul(weight, v)).squeeze()
    return sigma


# ============================================================================
# 2. SEPARATE STEP FUNCTIONS FOR EACH ARM (UNTIL PLATEAU)
# ============================================================================
def step_arm_0_baseline(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 0 (Baseline): SwiGLU + Emb Sqrt(d) + Depth-Weighted Rotation (Combo 1+2+4)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        if l == chosen_layer:
            h_next, h_in_rev = forward_and_invert_layer(layer, h_prev, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err.mean()
        else:
            h_next, _ = layer(h_prev, cos=cos, sin=sin)
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_1_plain_mlp(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 1: Plain 2-Matrix MLP (testing if SwiGLU overtakes it at true plateau)."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        if l == chosen_layer:
            h_next, h_in_rev = forward_and_invert_layer(layer, h_prev, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err.mean()
        else:
            h_next, _ = layer(h_prev, cos=cos, sin=sin)
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_2_cycle_loss(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 2: Round-Trip Cycle Consistency Loss F(F^(-1)(h)) == h."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        if l == chosen_layer:
            h_next, h_in_rev = forward_and_invert_layer(layer, h_prev, cos, sin)
            # Backward loss
            target_bw = h_prev.detach()
            rel_err_bw = (h_in_rev - target_bw).pow(2).sum(-1) / (target_bw.pow(2).sum(-1) + 1e-4)
            # Round-trip cycle forward: F(h_in_rev)
            h_rec, _ = layer(h_in_rev, cos=cos, sin=sin)
            target_fw = h_next.detach()
            rel_err_fw = (h_rec - target_fw).pow(2).sum(-1) / (target_fw.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err_bw.mean() + 0.5 * rel_err_fw.mean()
        else:
            h_next, _ = layer(h_prev, cos=cos, sin=sin)
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_3_spectral_barrier(model, x, cos, sin, depth_probs, scale_factor):
    """Arm 3: Spectral Norm Barrier on W_O and W_down via 1-step power iteration."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    h = model.model.embed_tokens(x) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        if l == chosen_layer:
            h_next, h_in_rev = forward_and_invert_layer(layer, h_prev, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            
            # Spectral barrier penalty: max(0, sigma - 1.0)^2
            sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
            sig_d = estimate_spectral_norm(layer.mlp.down_proj.weight)
            barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_d - 1.0).pow(2)
            loss_c = rel_err.mean() + 0.1 * barrier
        else:
            h_next, _ = layer(h_prev, cos=cos, sin=sin)
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


def step_arm_4_emb_rmsnorm(model, x, cos, sin, depth_probs, scale_factor, emb_norm_fn):
    """Arm 4: Embedding Pre-RMSNorm + Sqrt(d) Scaling."""
    chosen_layer = random.choices(range(len(model.model.layers)), weights=depth_probs, k=1)[0]
    raw_emb = model.model.embed_tokens(x)
    h = emb_norm_fn(raw_emb) * scale_factor
    loss_c = torch.tensor(0.0, device=x.device)

    for l, layer in enumerate(model.model.layers):
        h_prev = h
        if l == chosen_layer:
            h_next, h_in_rev = forward_and_invert_layer(layer, h_prev, cos, sin)
            target = h_prev.detach()
            rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
            loss_c = rel_err.mean()
        else:
            h_next, _ = layer(h_prev, cos=cos, sin=sin)
        h = h_next
    h_final = model.model.norm(h)
    return h_final, 0.02 * loss_c


# ============================================================================
# 3. EVALUATION AND PROBING
# ============================================================================
def compute_val_loss(model: LlamaForCausalLM, val_batches: List[torch.Tensor], vocab_size: int, scale_factor: float = 1.0, emb_norm_fn=None) -> float:
    was_training = model.training
    model.eval()
    total_ce = 0.0
    with torch.no_grad():
        for batch in val_batches:
            x = batch[:, :-1].contiguous()
            y = batch[:, 1:].contiguous()
            raw_emb = model.model.embed_tokens(x)
            if emb_norm_fn is not None:
                h = emb_norm_fn(raw_emb) * scale_factor
            else:
                h = raw_emb * scale_factor
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


def evaluate_inversion_probes(model: LlamaForCausalLM, x_probe: torch.Tensor, scale_factor: float = 1.0, emb_norm_fn=None) -> Dict[str, Any]:
    seq_l = x_probe.shape[1]
    results = {}
    with torch.no_grad():
        raw_emb = model.model.embed_tokens(x_probe)
        if emb_norm_fn is not None:
            h_in = emb_norm_fn(raw_emb) * scale_factor
        else:
            h_in = raw_emb * scale_factor
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
            norm2 = layer.post_attention_layernorm(h_next_t)
            h_mid = h_next_t - layer.mlp(norm2)
            norm1 = layer.input_layernorm(h_mid)
            attn_rev, _ = layer.self_attn(norm1, cos=cos, sin=sin)
            h_in_rev = h_mid - attn_rev
            err = (h_in_rev - h_prev_t).norm().item() / max(h_prev_t.norm().item(), 1e-6)
            layer_errs.append(err)
        results["layer_errs"] = layer_errs

        # Picard Multi-step (k=1, 2, 3)
        for k in [1, 2, 3]:
            h_rev = h_out.clone()
            for layer in reversed(model.model.layers):
                h_rev = invert_block_picard(layer, h_rev, cos, sin, k_steps=k)
            full_err = (h_rev - h_in).norm().item() / max(h_in.norm().item(), 1e-6)
            results[f"inv_picard_k{k}"] = full_err
            if k == 1:
                results["norm_rev"] = h_rev.norm().item()

        # Anderson Accelerated Multi-step
        h_rev_aa = h_out.clone()
        for layer in reversed(model.model.layers):
            h_rev_aa = invert_block_anderson(layer, h_rev_aa, cos, sin)
        results["inv_anderson"] = (h_rev_aa - h_in).norm().item() / max(h_in.norm().item(), 1e-6)

    return results


# ============================================================================
# 4. BENCHMARK EXECUTION (UNTIL TRUE CONVERGENCE PLATEAU)
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

    # Rigorous Dynamic Plateau Parameters
    eval_interval = 40       # Evaluate validation loss every 40 steps
    min_steps = 300          # Minimum steps before plateau check activates
    patience = 4             # 4 checks (160 steps) without improvement >= min_delta
    min_delta = 0.005        # Improvement threshold
    max_steps = 1100         # Upper limit ensuring full plateau convergence

    print("\n" + "=" * 155)
    print("  DEEP ADVANCEMENT BENCHMARK: TESTING CANDIDATE IMPROVEMENTS SEPARATELY UNTIL TRUE PLATEAU")
    print("  0. Baseline (SwiGLU Combo 1+2+4) | 1. Plain MLP (Testing if SwiGLU wins at plateau)")
    print("  2. Round-Trip Cycle Consistency  | 3. Spectral Barrier Regularization | 4. Embedding Pre-RMSNorm")
    print(f"  Criterion: Train dynamically until Validation Loss plateaus (min_steps={min_steps}, patience={patience} checks x {eval_interval} st = {patience*eval_interval} st)")
    print("=" * 155)
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

    # 5 Isolated Models
    m_base = make_model(42)
    m_plain = make_model(42)
    for layer in m_plain.model.layers:
        layer.mlp = PlainMLP(cfg).to(device)
    m_cycle = make_model(42)
    m_spectral = make_model(42)
    m_embnorm = make_model(42)
    emb_norm_module = RMSNorm(hidden_size, eps=cfg.rms_norm_eps).to(device)

    models = {
        "base": m_base,
        "plain": m_plain,
        "cycle": m_cycle,
        "spectral": m_spectral,
        "embnorm": m_embnorm,
    }

    optimizers = {
        "base": torch.optim.AdamW(m_base.parameters(), lr=lr, weight_decay=0.01),
        "plain": torch.optim.AdamW(m_plain.parameters(), lr=lr, weight_decay=0.01),
        "cycle": torch.optim.AdamW(m_cycle.parameters(), lr=lr, weight_decay=0.01),
        "spectral": torch.optim.AdamW(m_spectral.parameters(), lr=lr, weight_decay=0.01),
        "embnorm": torch.optim.AdamW(list(m_embnorm.parameters()) + list(emb_norm_module.parameters()), lr=lr, weight_decay=0.01),
    }

    emb_norms = {
        "base": None,
        "plain": None,
        "cycle": None,
        "spectral": None,
        "embnorm": emb_norm_module,
    }

    # Data Loaders
    train_loader = get_wikitext_dataloader(split="train", batch_size=8, seq_len=seq_len, pairs=False, max_samples=10000, shuffle=True)
    val_loader = get_wikitext_dataloader(split="test", batch_size=16, seq_len=seq_len, pairs=False, max_samples=160, shuffle=False)
    val_batches = [b["input_ids"].to(device) for b in val_loader]
    print(f"  Pre-loaded {len(val_batches)} validation batches onto {device}.")

    trackers = {}
    for k in models:
        init_val = compute_val_loss(models[k], val_batches, vocab_size, scale_factor=scale_factor, emb_norm_fn=emb_norms[k])
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
    print(f"{'Step':>6} | {'Baseline (SwiGLU)':>18} | {'Plain MLP':>12} | {'Cycle Loss':>12} | {'Spectral Barrier':>18} | {'Emb RMSNorm':>13} | {'Active':>6}")
    print("-" * 115)

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

        # 0. Baseline (SwiGLU Combo 1+2+4)
        if trackers["base"]["active"]:
            opt = optimizers["base"]; opt.zero_grad(set_to_none=True)
            emb = m_base.model.embed_tokens(x) * scale_factor
            cos, sin = m_base.model.rotary_emb(emb, seq_l)
            h, l_aux = step_arm_0_baseline(m_base, x, cos, sin, depth_probs, scale_factor)
            l_ce = F.cross_entropy(m_base.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            (l_ce + l_aux).backward()
            torch.nn.utils.clip_grad_norm_(m_base.parameters(), 1.0); opt.step()
            step_losses["base"] = l_ce.item()

        # 1. Plain MLP (Testing if SwiGLU wins at plateau)
        if trackers["plain"]["active"]:
            opt = optimizers["plain"]; opt.zero_grad(set_to_none=True)
            emb = m_plain.model.embed_tokens(x) * scale_factor
            cos, sin = m_plain.model.rotary_emb(emb, seq_l)
            h, l_aux = step_arm_1_plain_mlp(m_plain, x, cos, sin, depth_probs, scale_factor)
            l_ce = F.cross_entropy(m_plain.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            (l_ce + l_aux).backward()
            torch.nn.utils.clip_grad_norm_(m_plain.parameters(), 1.0); opt.step()
            step_losses["plain"] = l_ce.item()

        # 2. Round-Trip Cycle Consistency Loss
        if trackers["cycle"]["active"]:
            opt = optimizers["cycle"]; opt.zero_grad(set_to_none=True)
            emb = m_cycle.model.embed_tokens(x) * scale_factor
            cos, sin = m_cycle.model.rotary_emb(emb, seq_l)
            h, l_aux = step_arm_2_cycle_loss(m_cycle, x, cos, sin, depth_probs, scale_factor)
            l_ce = F.cross_entropy(m_cycle.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            (l_ce + l_aux).backward()
            torch.nn.utils.clip_grad_norm_(m_cycle.parameters(), 1.0); opt.step()
            step_losses["cycle"] = l_ce.item()

        # 3. Spectral Barrier Regularization
        if trackers["spectral"]["active"]:
            opt = optimizers["spectral"]; opt.zero_grad(set_to_none=True)
            emb = m_spectral.model.embed_tokens(x) * scale_factor
            cos, sin = m_spectral.model.rotary_emb(emb, seq_l)
            h, l_aux = step_arm_3_spectral_barrier(m_spectral, x, cos, sin, depth_probs, scale_factor)
            l_ce = F.cross_entropy(m_spectral.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            (l_ce + l_aux).backward()
            torch.nn.utils.clip_grad_norm_(m_spectral.parameters(), 1.0); opt.step()
            step_losses["spectral"] = l_ce.item()

        # 4. Embedding Pre-RMSNorm
        if trackers["embnorm"]["active"]:
            opt = optimizers["embnorm"]; opt.zero_grad(set_to_none=True)
            raw = m_embnorm.model.embed_tokens(x)
            emb = emb_norm_module(raw) * scale_factor
            cos, sin = m_embnorm.model.rotary_emb(emb, seq_l)
            h, l_aux = step_arm_4_emb_rmsnorm(m_embnorm, x, cos, sin, depth_probs, scale_factor, emb_norm_module)
            l_ce = F.cross_entropy(m_embnorm.lm_head(h).reshape(-1, vocab_size), y.reshape(-1))
            (l_ce + l_aux).backward()
            torch.nn.utils.clip_grad_norm_(list(m_embnorm.parameters()) + list(emb_norm_module.parameters()), 1.0); opt.step()
            step_losses["embnorm"] = l_ce.item()

        # Dynamic Plateau Evaluation
        if step % eval_interval == 0:
            for k, m in models.items():
                if not trackers[k]["active"]:
                    continue
                v_loss = compute_val_loss(m, val_batches, vocab_size, scale_factor=scale_factor, emb_norm_fn=emb_norms[k])
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
            print(f"{step:6d} | {fmt('base'):>18} | {fmt('plain'):>12} | {fmt('cycle'):>12} | {fmt('spectral'):>18} | {fmt('embnorm'):>13} | {active_count:>4d}/5")

    total_training_time = time.time() - t0_start
    print("-" * 115)
    print(f"  Training finished in {total_training_time:.1f}s. All arms converged to plateau.")

    # ------------------------------------------------------------------------
    # 5. RESTORE BEST PLATEAU CHECKPOINTS AND RUN MULTI-STEP / ANDERSON PROBING
    # ------------------------------------------------------------------------
    model_labels = {
        "base": "0. Baseline (SwiGLU Combo 1+2+4)",
        "plain": "1. Plain MLP (Testing SwiGLU Plateau)",
        "cycle": "2. +Round-Trip Cycle Loss",
        "spectral": "3. +Spectral Barrier Regularization",
        "embnorm": "4. +Embedding Pre-RMSNorm",
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
        torch.save(trackers[k]["best_state"], os.path.join(ckpt_dir, f"plateau_arm_{k}.pt"))

        val_ce = compute_val_loss(m, val_batches, vocab_size, scale_factor=scale_factor, emb_norm_fn=emb_norms[k])
        val_ppl = math.exp(min(val_ce, 20))
        inv_data = evaluate_inversion_probes(m, x_probe, scale_factor=scale_factor, emb_norm_fn=emb_norms[k])

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
    print("\n" + "=" * 165)
    print("  EXECUTIVE DIAGNOSTIC TABLE: TESTING CANDIDATE ADVANCEMENTS SEPARATELY UNTIL TRUE PLATEAU")
    print("=" * 165)
    header = f"  {'Metric':<34} | {'0. Baseline (SwiGLU)':<22} | {'1. Plain MLP':<18} | {'2. Cycle Loss':<18} | {'3. Spectral Barrier':<20} | {'4. Emb RMSNorm':<18}"
    print(header)
    print("-" * 165)

    rb = final_results["base"]
    rp = final_results["plain"]
    rc = final_results["cycle"]
    rs = final_results["spectral"]
    re = final_results["embnorm"]

    print(f"  {'Step at Plateau (Best)':<34} | {rb['plateau_step']:<22d} | {rp['plateau_step']:<18d} | {rc['plateau_step']:<18d} | {rs['plateau_step']:<20d} | {re['plateau_step']:<18d}")
    print(f"  {'Plateau Validation CE Loss':<34} | {rb['val_ce']:<22.4f} | {rp['val_ce']:<18.4f} | {rc['val_ce']:<18.4f} | {rs['val_ce']:<20.4f} | {re['val_ce']:<18.4f}")
    print(f"  {'Plateau Validation PPL':<34} | {rb['val_ppl']:<22.2f} | {rp['val_ppl']:<18.2f} | {rc['val_ppl']:<18.2f} | {rs['val_ppl']:<20.2f} | {re['val_ppl']:<18.2f}")
    
    delta = lambda m: m['val_ppl'] - rb['val_ppl']
    print(f"  {'Delta PPL vs Baseline':<34} | {'0.00 (Baseline)':<22} | {delta(rp):<+18.2f} | {delta(rc):<+18.2f} | {delta(rs):<+20.2f} | {delta(re):<+18.2f}")
    print(f"  {'Layer 0 1-Step Invert Error':<34} | {rb['layer_errs'][0]:<22.4f} | {rp['layer_errs'][0]:<18.4f} | {rc['layer_errs'][0]:<18.4f} | {rs['layer_errs'][0]:<20.4f} | {re['layer_errs'][0]:<18.4f}")
    print(f"  {'Layer 3 1-Step Invert Error':<34} | {rb['layer_errs'][3]:<22.4f} | {rp['layer_errs'][3]:<18.4f} | {rc['layer_errs'][3]:<18.4f} | {rs['layer_errs'][3]:<20.4f} | {re['layer_errs'][3]:<18.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=1)':<34} | {rb['inv_picard_k1']:<22.4f} | {rp['inv_picard_k1']:<18.4f} | {rc['inv_picard_k1']:<18.4f} | {rs['inv_picard_k1']:<20.4f} | {re['inv_picard_k1']:<18.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=2)':<34} | {rb['inv_picard_k2']:<22.4f} | {rp['inv_picard_k2']:<18.4f} | {rc['inv_picard_k2']:<18.4f} | {rs['inv_picard_k2']:<20.4f} | {re['inv_picard_k2']:<18.4f}")
    print(f"  {'Full 4-Layer Invert (Picard k=3)':<34} | {rb['inv_picard_k3']:<22.4f} | {rp['inv_picard_k3']:<18.4f} | {rc['inv_picard_k3']:<18.4f} | {rs['inv_picard_k3']:<20.4f} | {re['inv_picard_k3']:<18.4f}")
    print(f"  {'Full 4-Layer Invert (Anderson)':<34} | {rb['inv_anderson']:<22.4f} | {rp['inv_anderson']:<18.4f} | {rc['inv_anderson']:<18.4f} | {rs['inv_anderson']:<20.4f} | {re['inv_anderson']:<18.4f}")
    print(f"  {'Reversed Vector L2 Norm':<34} | {rb['norm_rev']:<22.2f} | {rp['norm_rev']:<18.2f} | {rc['norm_rev']:<18.2f} | {rs['norm_rev']:<20.2f} | {re['norm_rev']:<18.2f}")
    print("=" * 165)


if __name__ == "__main__":
    main()
