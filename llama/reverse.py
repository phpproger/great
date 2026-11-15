import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .model import LlamaForCausalLM, RMSNorm


class ContextPredictor(nn.Module):
    """Inverse Neural Network: Predicts the context representation vector given an answer sequence.
    Given an answer sequence Y (shape: B, L_ans), predicts context embeddings
    Z_pred (shape: B, context_len, hidden_size) such that LLaMA conditioned on Z_pred
    assigns high likelihood to Y.
    """
    def __init__(
        self,
        config: LlamaConfig,
        context_len: int = 4,
        encoder_layers: int = 2,
    ):
        super().__init__()
        self.config = config
        self.context_len = context_len
        self.hidden_size = config.hidden_size

        # Embed answer tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Lightweight bidirectional encoder to process the answer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Learnable queries to extract `context_len` context vectors from the answer representation
        self.context_queries = nn.Parameter(torch.randn(1, context_len, config.hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, answer_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            answer_ids: LongTensor of shape (batch_size, answer_len)
        Returns:
            predicted_context: FloatTensor of shape (batch_size, context_len, hidden_size)
        """
        batch_size = answer_ids.shape[0]

        # 1. Embed answer tokens
        ans_emb = self.embed_tokens(answer_ids)

        # 2. Encode answer context
        ans_encoded = self.encoder(ans_emb)
        ans_encoded = self.norm(ans_encoded)

        # 3. Query cross-attention to reconstruct context vectors
        queries = self.context_queries.expand(batch_size, -1, -1)
        context_vecs, _ = self.cross_attn(
            query=queries,
            key=ans_encoded,
            value=ans_encoded,
        )
        predicted_context = self.out_proj(context_vecs)
        return predicted_context


class DirectContextInverter:
    """Optimization-based Inversion (Continuous Prefix / Soft-Prompt Inversion).
    Given a frozen LLaMA model and target answer tokens Y, optimizes a continuous context
    vector Z to directly minimize the negative log-likelihood of Y:
        min_Z -log P(Y | Z)
    """
    def __init__(
        self,
        model: LlamaForCausalLM,
        context_len: int = 4,
        lr: float = 0.05,
        num_steps: int = 40,
    ):
        self.model = model
        self.context_len = context_len
        self.lr = lr
        self.num_steps = num_steps

    def invert(
        self,
        answer_ids: torch.Tensor,
        init_context: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Finds the optimal context vector for the target answer tokens."""
        self.model.eval()
        device = answer_ids.device
        batch_size = answer_ids.shape[0]
        hidden_size = self.model.config.hidden_size

        # Initialize context vector: random or from prior
        if init_context is not None:
            z = init_context.detach().clone().requires_grad_(True)
        else:
            z = torch.randn(
                batch_size, self.context_len, hidden_size,
                device=device, dtype=torch.float32, requires_grad=True
            )

        optimizer = torch.optim.AdamW([z], lr=self.lr, weight_decay=1e-4)
        answer_embeds = self.model.model.embed_tokens(answer_ids).detach()

        loss_history = []


        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Concatenate continuous context vector with answer embeddings
            combined_embeds = torch.cat([z, answer_embeds], dim=1)

            # Targets: ignore context positions (-100), supervise answer positions
            # When predicting token at position i, input is token i-1.
            # In combined_embeds: positions 0..context_len-1 are context, context_len..end are answer.
            # Next token after context_len-1 should be answer_ids[0].
            full_targets = torch.full(
                (batch_size, self.context_len + answer_ids.shape[1]),
                fill_value=-100,
                dtype=torch.long,
                device=device,
            )
            # Answer tokens as targets:
            full_targets[:, self.context_len :] = answer_ids

            _, loss, _ = self.model(
                inputs_embeds=combined_embeds,
                labels=full_targets,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if verbose and (step + 1) % 10 == 0:
                print(f"      Inversion step {step + 1}/{self.num_steps} - Answer NLL: {loss_val:.4f}")

        return {
            "context_vector": z.detach(),
            "final_loss": loss_history[-1],
            "initial_loss": loss_history[0],
            "loss_history": loss_history,
        }


class ContextAnalyzer:
    """Utilities for interpreting, decoding, and scoring context vectors."""

    @staticmethod
    def compute_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> Dict[str, float]:
        """Computes cosine similarity and MSE between two context vectors."""
        flat1 = vec1.flatten()
        flat2 = vec2.flatten()
        cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
        mse = F.mse_loss(flat1, flat2).item()
        return {"cosine_similarity": cos_sim, "mse": mse}

    @staticmethod
    def decode_to_tokens(
        context_vector: torch.Tensor,
        embedding_matrix: torch.Tensor,
        tokenizer,
        top_k: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        """Projects continuous context vectors back to vocabulary tokens via cosine similarity.
        Reveals what words/concepts the predicted context vector encodes.
        """
        # context_vector shape: (1, context_len, hidden_dim) or (context_len, hidden_dim)
        if context_vector.ndim == 3:
            context_vector = context_vector.squeeze(0)

        # Normalize embeddings for cosine similarity
        norm_we = F.normalize(embedding_matrix.detach().float(), p=2, dim=-1)
        norm_ctx = F.normalize(context_vector.detach().float(), p=2, dim=-1)

        # Similarity: (context_len, vocab_size)
        sim = torch.matmul(norm_ctx, norm_we.t())

        decoded_positions = []
        for pos in range(context_vector.shape[0]):
            top_scores, top_indices = torch.topk(sim[pos], k=top_k)
            pos_tokens = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                token_text = tokenizer.decode([idx]).strip()
                pos_tokens.append((token_text, score))
            decoded_positions.append(pos_tokens)

        return decoded_positions

    @staticmethod
    def evaluate_steering_loss(
        model: LlamaForCausalLM,
        context_vector: Optional[torch.Tensor],
        answer_ids: torch.Tensor,
    ) -> float:
        """Evaluates the cross-entropy loss of the answer tokens conditioned on a context vector."""
        model.eval()
        device = answer_ids.device
        batch_size = answer_ids.shape[0]

        with torch.no_grad():
            answer_embeds = model.model.embed_tokens(answer_ids)
            if context_vector is not None:
                combined_embeds = torch.cat([context_vector.to(device), answer_embeds], dim=1)
                context_len = context_vector.shape[1]
                full_targets = torch.full(
                    (batch_size, context_len + answer_ids.shape[1]),
                    fill_value=-100,
                    dtype=torch.long,
                    device=device,
                )
                full_targets[:, context_len:] = answer_ids
            else:
                combined_embeds = answer_embeds
                full_targets = answer_ids

            _, loss, _ = model(
                inputs_embeds=combined_embeds,
                labels=full_targets,
            )
            return loss.item()
