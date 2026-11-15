import math
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .model import RMSNorm


class AffineCouplingBlock(nn.Module):
    """Affine Coupling Reversible Layer (RealNVP / Glow style).
    Breaks the identity shortcut by learning both scaling s(z) and translation t(z).
    
    Forward (Y -> X):
        z1, z2 = split(z)
        s1, t1 = net1(z2)
        z1' = z1 * exp(s1) + t1
        s2, t2 = net2(z1')
        z2' = z2 * exp(s2) + t2
        return concat(z1', z2')
    
    Inverse (^(-1)) (X -> Y):
        z1', z2' = split(z')
        s2, t2 = net2(z1')
        z2 = (z2' - t2) * exp(-s2)
        s1, t1 = net1(z2)
        z1 = (z1' - t1) * exp(-s1)
        return concat(z1, z2)
        
    Exact mathematical invertibility: M^(-1)(M(y)) == y to machine precision (< 1e-6).
    """
    def __init__(self, dim: int, max_seq_len: int = 64, num_heads: int = 4, scale_clamp: float = 0.8):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for 2-way splitting"
        self.half_dim = dim // 2
        self.scale_clamp = scale_clamp

        # Position embeddings for coupling networks to distinguish token positions
        self.pos_emb1 = nn.Parameter(torch.randn(1, max_seq_len, self.half_dim) * 0.02)
        self.pos_emb2 = nn.Parameter(torch.randn(1, max_seq_len, self.half_dim) * 0.02)

        # Network 1: processes z2 to transform z1
        self.norm1 = RMSNorm(self.half_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim=self.half_dim, num_heads=num_heads, batch_first=True)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim * 2),
            nn.GELU(),
            nn.Linear(self.half_dim * 2, self.half_dim * 2), # outputs [s1, t1]
        )

        # Network 2: processes z1' to transform z2
        self.norm2 = RMSNorm(self.half_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.half_dim, num_heads=num_heads, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim * 2),
            nn.GELU(),
            nn.Linear(self.half_dim * 2, self.half_dim * 2), # outputs [s2, t2]
        )

    def _get_st1(self, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = z2.shape[1]
        norm_z2 = self.norm1(z2) + self.pos_emb1[:, :seq_len, :]
        attn_out, _ = self.attn1(norm_z2, norm_z2, norm_z2)
        st = self.mlp1(norm_z2 + attn_out)
        s, t = torch.chunk(st, 2, dim=-1)
        s = self.scale_clamp * torch.tanh(s / self.scale_clamp)
        return s, t

    def _get_st2(self, z1_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = z1_prime.shape[1]
        norm_z1 = self.norm2(z1_prime) + self.pos_emb2[:, :seq_len, :]
        attn_out, _ = self.attn2(norm_z1, norm_z1, norm_z1)
        st = self.mlp2(norm_z1 + attn_out)
        s, t = torch.chunk(st, 2, dim=-1)
        s = self.scale_clamp * torch.tanh(s / self.scale_clamp)
        return s, t

    def forward_y_to_x(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: Answer Y -> Context X."""
        z1, z2 = torch.chunk(z, 2, dim=-1)
        s1, t1 = self._get_st1(z2)
        z1_prime = z1 * torch.exp(s1) + t1
        s2, t2 = self._get_st2(z1_prime)
        z2_prime = z2 * torch.exp(s2) + t2
        return torch.cat([z1_prime, z2_prime], dim=-1)

    def inverse_x_to_y(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Inverse pass (^(-1)): Context X -> Answer Y."""
        z1_prime, z2_prime = torch.chunk(z_prime, 2, dim=-1)
        s2, t2 = self._get_st2(z1_prime)
        z2 = (z2_prime - t2) * torch.exp(-s2)
        s1, t1 = self._get_st1(z2)
        z1 = (z1_prime - t1) * torch.exp(-s1)
        return torch.cat([z1, z2], dim=-1)


class InvertibleContextModel(nn.Module):
    """Invertible neural architecture that learns Y -> X and is inverted (^(-1)) for X -> Y.
    
    1. Training direction (Y -> X):
       - Input: Answer tokens Y (embedded)
       - Passes through stack of AffineCouplingBlocks to predict Context vector X
       - Weights are optimized ONLY on the reverse task: min || M(Y) - X ||^2
    
    2. Generation direction (^(-1) : X -> Y):
       - Input: Context vector X
       - Passes backwards through the EXACT SAME weights using analytical inverse:
         Y_hat = M^(-1)(X)
       - Projects Y_hat to vocabulary tokens to generate the answer Y!
    """
    def __init__(self, config: LlamaConfig, seq_len: int = 8, num_layers: int = 4):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.hidden_size = config.hidden_size

        self.blocks = nn.ModuleList([
            AffineCouplingBlock(dim=config.hidden_size, max_seq_len=max(64, seq_len), num_heads=config.num_attention_heads // 2)
            for _ in range(num_layers)
        ])

    def forward_y_to_x(self, answer_vectors: torch.Tensor) -> torch.Tensor:
        """Runs Y -> X: maps answer vector to context vector."""
        z = answer_vectors
        for i, block in enumerate(self.blocks):
            z = block.forward_y_to_x(z)
            if i < len(self.blocks) - 1:
                # Permute channels between layers so all features are transformed
                z = torch.roll(z, shifts=self.hidden_size // 4, dims=-1)
        return z

    def inverse_x_to_y(self, context_vectors: torch.Tensor) -> torch.Tensor:
        """Runs (M)^(-1): inverts context vector back into answer vector using the same weights."""
        z = context_vectors
        for i, block in enumerate(reversed(self.blocks)):
            if i > 0:
                # Reverse permutation
                z = torch.roll(z, shifts=-(self.hidden_size // 4), dims=-1)
            z = block.inverse_x_to_y(z)
        return z

    def compute_logits(
        self,
        vectors: torch.Tensor,
        embedding_matrix: torch.Tensor,
        temperature: float = 0.20,
    ) -> torch.Tensor:
        """Projects continuous hidden vectors to vocabulary logits via temperature-scaled cosine similarity."""
        norm_we = F.normalize(embedding_matrix.detach().float(), p=2, dim=-1)
        norm_v = F.normalize(vectors.float(), p=2, dim=-1)
        return torch.matmul(norm_v, norm_we.t()) / temperature

    @staticmethod
    def compute_loss_and_ppl(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[float, float]:
        """Calculates Cross-Entropy Loss and Perplexity (PPL = exp(loss))."""
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        ppl = math.exp(min(loss.item(), 20.0))
        return loss.item(), ppl

    def decode_vectors_to_tokens(
        self,
        vectors: torch.Tensor,
        embedding_matrix: torch.Tensor,
        tokenizer,
        top_k: int = 1,
    ) -> List[str]:
        """Projects continuous vectors to discrete tokens via embedding cosine similarity."""
        if vectors.ndim == 3:
            vectors = vectors.squeeze(0)
            
        norm_we = F.normalize(embedding_matrix.detach().float(), p=2, dim=-1)
        norm_v = F.normalize(vectors.detach().float(), p=2, dim=-1)
        sim = torch.matmul(norm_v, norm_we.t())
        best_token_ids = torch.argmax(sim, dim=-1).tolist()
        return [tokenizer.decode([tid]).strip() for tid in best_token_ids]
