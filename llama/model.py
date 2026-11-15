import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequencies: base^(-2i / dim) for i in 0 .. dim/2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute table for max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    # cos, sin shape: (seq_len, dim) or (batch, seq_len, dim)
    if position_ids is not None:
        # position_ids shape: (batch, seq_len)
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = sin[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
    else:
        # Default unsqueeze for heads: (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands key/value heads for Grouped Query Attention (GQA).
    Shape: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """Multi-Head / Grouped-Query Attention with RoPE and KV Cache support."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # Handle KV Caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_key_value = (k, v) if use_cache else None

        # Repeat KV for Grouped-Query Attention (GQA)
        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        # Attention computation using PyTorch scaled_dot_product_attention (SDPA)
        kv_seq_len = k_expanded.shape[2]
        is_causal = (seq_len > 1) and (past_key_value is None)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, current_key_value


class PlainMLP(nn.Module):
    """Monotonic 2-matrix Plain MLP: down_proj(SiLU(up_proj(x))).
    Replaces SwiGLU quadratic bilinear gating with a monotonic Lipschitz operator,
    improving contraction conditioning and multi-layer invertibility.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        hidden_dim = config.mlp_hidden_dim
        self.fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)

    @property
    def up_proj(self) -> nn.Linear:
        return self.fc1

    @property
    def down_proj(self) -> nn.Linear:
        return self.fc2

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
    return torch.matmul(u.detach().t(), torch.matmul(weight, v.detach())).squeeze()


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single LLaMA Transformer Decoder layer."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "mlp_type", "plain") == "plain":
            self.mlp = PlainMLP(config)
        else:
            self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with pre-RMSNorm and residual
        residual = x
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + hidden_states

        # MLP with pre-RMSNorm and residual
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x, present_key_value

    def invert(
        self,
        h_next: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_steps: int = 2,
        method: str = "picard",
    ) -> torch.Tensor:
        """Inverts this transformer block using Picard iteration or Anderson acceleration."""
        if method == "anderson":
            return self.invert_anderson(h_next, cos, sin)
        # Default Picard fixed-point iteration
        h_mid = h_next.clone()
        for _ in range(k_steps):
            norm2 = self.post_attention_layernorm(h_mid)
            h_mid = h_next - self.mlp(norm2)
        h_prev = h_mid.clone()
        for _ in range(k_steps):
            norm1 = self.input_layernorm(h_prev)
            attn_out, _ = self.self_attn(norm1, cos=cos, sin=sin)
            h_prev = h_mid - attn_out
        return h_prev

    def invert_anderson(self, h_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Anderson Accelerated fixed-point iteration (m=2 iterates)."""
        # 1. Invert MLP with Anderson Acceleration
        x0 = h_next.clone()
        g0 = h_next - self.mlp(self.post_attention_layernorm(x0))
        x1 = g0.clone()
        g1 = h_next - self.mlp(self.post_attention_layernorm(x1))
        r0 = g0 - x0
        r1 = g1 - x1
        dr = r1 - r0
        dot_r0_dr = (r0 * dr).sum(dim=-1, keepdim=True)
        norm_dr_sq = (dr * dr).sum(dim=-1, keepdim=True) + 1e-8
        alpha_mlp = (-dot_r0_dr / norm_dr_sq).clamp(0.0, 1.0)
        h_mid = (1.0 - alpha_mlp) * g0 + alpha_mlp * g1

        # 2. Invert Attention with Anderson Acceleration
        x0_a = h_mid.clone()
        attn_out0, _ = self.self_attn(self.input_layernorm(x0_a), cos=cos, sin=sin)
        g0_a = h_mid - attn_out0
        x1_a = g0_a.clone()
        attn_out1, _ = self.self_attn(self.input_layernorm(x1_a), cos=cos, sin=sin)
        g1_a = h_mid - attn_out1
        r0_a = g0_a - x0_a
        r1_a = g1_a - x1_a
        dr_a = r1_a - r0_a
        dot_r0_dr_a = (r0_a * dr_a).sum(dim=-1, keepdim=True)
        norm_dr_sq_a = (dr_a * dr_a).sum(dim=-1, keepdim=True) + 1e-8
        alpha_attn = (-dot_r0_dr_a / norm_dr_sq_a).clamp(0.0, 1.0)
        h_prev = (1.0 - alpha_attn) * g0_a + alpha_attn * g1_a
        return h_prev

    def forward_and_invert(self, h_prev: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes forward pass and k-step Banach inversion for auxiliary loss."""
        h_next, _ = self.forward(h_prev, cos=cos, sin=sin)
        if k_steps == 1:
            norm2 = self.post_attention_layernorm(h_next)
            h_mid = h_next - self.mlp(norm2)
            norm1 = self.input_layernorm(h_mid)
            attn_rev, _ = self.self_attn(norm1, cos=cos, sin=sin)
            h_in_rev = h_mid - attn_rev
            return h_next, h_in_rev
        elif k_steps == 2:
            # 2-step Picard MLP inverse
            x0 = h_next
            x1 = h_next - self.mlp(self.post_attention_layernorm(x0))
            h_mid = h_next - self.mlp(self.post_attention_layernorm(x1))
            # 2-step Picard Attention inverse
            z0 = h_mid
            attn0, _ = self.self_attn(self.input_layernorm(z0), cos=cos, sin=sin)
            z1 = h_mid - attn0
            attn1, _ = self.self_attn(self.input_layernorm(z1), cos=cos, sin=sin)
            h_in_rev = h_mid - attn1
            return h_next, h_in_rev
        else:
            return h_next, self.invert(h_next, cos=cos, sin=sin, k_steps=k_steps, method="picard")


class LlamaModel(nn.Module):
    """Core LLaMA Transformer stack."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
            hidden_states = inputs_embeds
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            hidden_states = self.embed_tokens(input_ids)
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # Scale embeddings by sqrt(hidden_size) if configured [Combo 1+2+4 baseline]
        if getattr(self.config, "scale_embeddings", False):
            hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        # Compute position IDs if not provided
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # RoPE embeddings
        max_pos = position_ids.max().item() + 1
        cos, sin = self.rotary_emb(x=hidden_states, seq_len=max(max_pos, seq_len))

        next_decoder_cache = [] if use_cache else None

        # Stochastic single-layer rotation for Banach Contraction Loss B [Opt 4 & Opt 2]
        self.last_aux_loss = None
        chosen_layer = -1
        k_steps = getattr(self.config, "aux_k_steps", 1)
        if self.training and getattr(self.config, "use_contraction_aux", False) and not use_cache:
            probs = getattr(self.config, "contraction_layer_probs", None)
            if probs is not None and len(probs) == len(self.layers):
                chosen_layer = random.choices(range(len(self.layers)), weights=probs, k=1)[0]
            else:
                chosen_layer = random.randint(0, len(self.layers) - 1)

        aux_loss = None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if idx == chosen_layer:
                h_prev = hidden_states
                hidden_states, h_in_rev = layer.forward_and_invert(h_prev, cos=cos, sin=sin, k_steps=k_steps)
                target = h_prev.detach()
                rel_err = (h_in_rev - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-4)
                layer_aux = rel_err.mean()

                # Optional Round-Trip Cycle Consistency Loss: h -> h_rev -> h_rec == h_next
                if getattr(self.config, "use_cycle_consistency", False):
                    h_rec, _ = layer(h_in_rev, cos=cos, sin=sin)
                    target_fw = hidden_states.detach()
                    rel_err_fw = (h_rec - target_fw).pow(2).sum(-1) / (target_fw.pow(2).sum(-1) + 1e-4)
                    cycle_weight = getattr(self.config, "cycle_consistency_weight", 0.25)
                    layer_aux = layer_aux + cycle_weight * rel_err_fw.mean()

                # Single-layer spectral barrier fallback (when not global)
                if getattr(self.config, "use_spectral_barrier", False) and not getattr(self.config, "global_spectral_barrier", False):
                    sig_o = estimate_spectral_norm(layer.self_attn.o_proj.weight)
                    sig_d = estimate_spectral_norm(layer.mlp.down_proj.weight)
                    barrier = F.relu(sig_o - 1.0).pow(2) + F.relu(sig_d - 1.0).pow(2)
                    layer_aux = layer_aux + 0.1 * barrier

                aux_loss = layer_aux
                layer_cache = None
            else:
                hidden_states, layer_cache = layer(
                    hidden_states,
                    cos=cos,
                    sin=sin,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if use_cache:
                next_decoder_cache.append(layer_cache)

        # Global All-Layer Spectral Barrier Regularization (computed across all layers)
        if self.training and getattr(self.config, "use_spectral_barrier", False) and getattr(self.config, "global_spectral_barrier", False):
            total_barrier = torch.tensor(0.0, device=hidden_states.device)
            for l in self.layers:
                sig_o = estimate_spectral_norm(l.self_attn.o_proj.weight)
                sig_d = estimate_spectral_norm(l.mlp.down_proj.weight)
                total_barrier = total_barrier + F.relu(sig_o - 1.0).pow(2) + F.relu(sig_d - 1.0).pow(2)
            avg_barrier = total_barrier / len(self.layers)
            if aux_loss is not None:
                aux_loss = aux_loss + 0.1 * avg_barrier
            else:
                aux_loss = 0.1 * avg_barrier

        self.last_aux_loss = aux_loss

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

    def invert(
        self,
        h_out: torch.Tensor,
        seq_len: int,
        k_steps: Optional[int] = None,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Runs multi-step sequential Banach unrolling (h_L -> h_0) back to token embedding space."""
        k = k_steps if k_steps is not None else getattr(self.config, "inversion_steps", 2)
        m = method if method is not None else getattr(self.config, "default_inversion_method", "anderson")
        cos, sin = self.rotary_emb(x=h_out, seq_len=seq_len)
        h_rev = h_out.clone()
        for layer in reversed(self.layers):
            h_rev = layer.invert(h_rev, cos=cos, sin=sin, k_steps=k, method=m)
        if getattr(self.config, "scale_embeddings", False):
            h_rev = h_rev / math.sqrt(self.config.hidden_size)
        return h_rev


class LlamaForCausalLM(nn.Module):
    """LLaMA Model with Language Modeling head and autoregressive generation."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initializes weights with truncated normal / normal distribution."""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add Banach Contraction Auxiliary Loss if training [Combo 1+2+4 baseline]
            aux_loss = getattr(self.model, "last_aux_loss", None)
            if aux_loss is not None and self.training and (getattr(self.config, "use_contraction_aux", False) or getattr(self.config, "global_spectral_barrier", False)):
                weight = getattr(self.config, "contraction_aux_weight", 0.02)
                loss = ce_loss + weight * aux_loss
            else:
                loss = ce_loss

        return logits, loss, past_key_values

    def invert(
        self,
        h_out: torch.Tensor,
        seq_len: int,
        k_steps: Optional[int] = None,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Inverts model representations back to token embedding space using multi-step Banach unrolling."""
        return self.model.invert(h_out, seq_len=seq_len, k_steps=k_steps, method=method)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive text generation with top-k / top-p (nucleus) sampling and KV caching.
        Supports starting from discrete token IDs (prompt_ids) or continuous vector embeddings (prompt_embeds).
        """
        self.eval()
        if prompt_embeds is not None:
            device = prompt_embeds.device
            current_ids = None
        elif prompt_ids is not None:
            device = prompt_ids.device
            current_ids = prompt_ids.clone()
        else:
            raise ValueError("Must provide either prompt_ids or prompt_embeds")

        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0 and prompt_embeds is not None:
                logits, _, past_key_values = self.forward(
                    inputs_embeds=prompt_embeds,
                    past_key_values=None,
                    use_cache=use_cache,
                )
            else:
                if use_cache and past_key_values is not None:
                    model_input = current_ids[:, -1:]
                else:
                    model_input = current_ids

                logits, _, past_key_values = self.forward(
                    input_ids=model_input,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache,
                )

            # Get logits of the last token
            next_token_logits = logits[:, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0 and current_ids is not None:
                for b_idx in range(current_ids.shape[0]):
                    for token_id in set(current_ids[b_idx].tolist()):
                        if next_token_logits[b_idx, token_id] > 0:
                            next_token_logits[b_idx, token_id] /= repetition_penalty
                        else:
                            next_token_logits[b_idx, token_id] *= repetition_penalty

            # Apply temperature
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if current_ids is None:
                current_ids = next_token
            else:
                current_ids = torch.cat([current_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return current_ids
