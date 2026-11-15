import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlamaConfig:
    """Configuration class for LLaMA model architecture."""
    vocab_size: int = 49152           # Default matches SmolLM2 / LLaMA tokenizer
    hidden_size: int = 256            # Hidden dimension d_model
    num_hidden_layers: int = 6        # Number of transformer decoder layers
    num_attention_heads: int = 8      # Number of query heads
    num_key_value_heads: int = 4      # Number of key/value heads for GQA (Grouped Query Attention)
    intermediate_size: Optional[int] = None  # SwiGLU MLP hidden dimension
    max_position_embeddings: int = 2048      # Max context sequence length
    rms_norm_eps: float = 1e-5        # RMSNorm epsilon
    rope_theta: float = 10000.0       # Base frequency for RoPE
    attention_dropout: float = 0.0    # Dropout on attention weights
    hidden_dropout: float = 0.0       # Dropout on MLP/residual
    initializer_range: float = 0.02   # Stddev for weight initialization
    tie_word_embeddings: bool = True  # Weight tying between embed_tokens and lm_head
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    # Production Banach Contraction Baseline Configuration
    # Architecture: Plain 2-Matrix MLP + Spectral Barrier + Sqrt(d) Scaling + Depth Rotation
    mlp_type: str = "plain"                # "plain" (2-matrix down(silu(up(x)))) or "swiglu" (3-matrix)
    scale_embeddings: bool = True          # Scales token embeddings by sqrt(hidden_size)
    use_contraction_aux: bool = True       # Enables Banach Contraction Auxiliary Loss B
    contraction_aux_weight: float = 0.02   # Lambda weight for Banach Contraction Loss
    use_spectral_barrier: bool = True      # Spectral Barrier Regularization (sigma(W) <= 1.0) on output projections
    spectral_barrier_weight: float = 0.002 # Effective weight for spectral barrier penalty
    contraction_layer_probs: Optional[list] = None # Depth-weighted sampling [Opt 2 & 4]
    inversion_steps: int = 2               # Default test-time Banach fixed-point iterations
    default_inversion_method: str = "anderson" # Default test-time inversion solver ("anderson" or "picard")

    def __post_init__(self):
        if self.contraction_layer_probs is None:
            # Depth-weighted exponential decay: [0.40, 0.30, 0.20, 0.10]
            decay = [math.exp(-0.25 * l) for l in range(self.num_hidden_layers)]
            s = sum(decay)
            self.contraction_layer_probs = [d / s for d in decay]
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0, (
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be a multiple of num_key_value_heads ({self.num_key_value_heads})"
        )
        if self.intermediate_size is None:
            # Default SwiGLU hidden dim heuristic from LLaMA: ~8/3 * hidden_size, rounded to multiple of 256
            raw_dim = int(2 * (4 * self.hidden_size) / 3)
            multiple_of = 256
            self.intermediate_size = multiple_of * ((raw_dim + multiple_of - 1) // multiple_of)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlp_hidden_dim(self) -> int:
        """Returns the internal MLP hidden dimension.
        For Plain MLP (2-matrix), expands by 1.5x to match the parameter count of 3-matrix SwiGLU.
        """
        if self.mlp_type == "plain":
            return int(self.intermediate_size * 1.5)
        return self.intermediate_size

    @classmethod
    def tiny(cls, vocab_size: int = 49152) -> "LlamaConfig":
        """A lightweight configuration (~4.5M params) ideal for fast unit tests and quick TinyStories experimentation."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=384,
            max_position_embeddings=512,
            tie_word_embeddings=True,
        )

    @classmethod
    def small(cls, vocab_size: int = 49152) -> "LlamaConfig":
        """Small configuration (~28M params) suitable for training on TinyStories on a consumer GPU."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=288,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=768,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
        )

    @classmethod
    def smollm2_135m(cls, vocab_size: int = 49152) -> "LlamaConfig":
        """Matches SmolLM2-135M architecture."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=576,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            intermediate_size=1536,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            rope_theta=100000.0,
            tie_word_embeddings=True,
        )

    @classmethod
    def llama_114m(cls, vocab_size: int = 50257) -> "LlamaConfig":
        """Standard 114M parameter LLaMA configuration for multi-GPU training on WikiText."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
            mlp_type="plain",
            use_spectral_barrier=True,
        )
