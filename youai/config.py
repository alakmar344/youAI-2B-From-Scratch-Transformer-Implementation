"""YouAI model configuration.

The :class:`YouAIConfig` dataclass fully describes a model's architecture.  It is
serialisable to / from JSON so a checkpoint can be reloaded without the original
Python code, and it validates its own fields so mistakes are caught early with a
clear error message instead of an obscure crash deep inside PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


# Activation / normalisation / positional-embedding choices we support.
_ACTIVATIONS = {"gelu", "gelu_new", "relu", "silu", "swiglu", "geglu"}
_NORMS = {"layernorm", "rmsnorm"}
_POS_EMB = {"learned", "rotary", "alibi"}


@dataclass
class YouAIConfig:
    """Configuration for a YouAI language model.

    Args:
        vocab_size: Size of the vocabulary (50257 for the GPT-2 tokenizer).
        max_position_embeddings: Maximum sequence length the model supports.
        hidden_size: Embedding / residual stream dimension.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for grouped-query
            attention (defaults to ``num_attention_heads``, i.e. standard MHA).
        intermediate_size: Hidden dimension of the feed-forward network.
        hidden_dropout_prob: Dropout on hidden states / residual connections.
        attention_dropout_prob: Dropout on the attention matrix.
        activation: Feed-forward activation: one of gelu, relu, silu, swiglu, geglu.
        norm_type: Normalisation layer: ``layernorm`` or ``rmsnorm``.
        position_embedding_type: ``learned``, ``rotary`` or ``alibi``.
        rope_theta: Base frequency for rotary embeddings.
        layer_norm_eps: Epsilon for normalisation layers.
        initializer_range: Std-dev of the truncated-normal weight initialiser.
        tie_word_embeddings: Share weights between the input embedding and the
            output projection (reduces parameters, usually improves quality).
        use_flash_attention: Use PyTorch's fused scaled-dot-product-attention
            when available (much faster / lower memory).
        gradient_checkpointing: Trade compute for memory during training.
        bos_token_id / eos_token_id / pad_token_id: Special token ids.
    """

    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    activation: str = "gelu"
    norm_type: str = "layernorm"
    position_embedding_type: str = "learned"
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.activation = str(self.activation).lower()
        self.norm_type = str(self.norm_type).lower()
        self.position_embedding_type = str(self.position_embedding_type).lower()
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "YouAIConfig":
        """Validate the configuration, raising ``ValueError`` on problems."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. Choose from {sorted(_ACTIVATIONS)}."
            )
        if self.norm_type not in _NORMS:
            raise ValueError(
                f"Unknown norm_type '{self.norm_type}'. Choose from {sorted(_NORMS)}."
            )
        if self.position_embedding_type not in _POS_EMB:
            raise ValueError(
                f"Unknown position_embedding_type '{self.position_embedding_type}'. "
                f"Choose from {sorted(_POS_EMB)}."
            )
        for name in ("vocab_size", "hidden_size", "num_hidden_layers",
                     "num_attention_heads", "intermediate_size",
                     "max_position_embeddings"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------
    @property
    def total_params(self) -> int:
        """Accurate analytical parameter count for the configured model."""
        h = self.hidden_size
        kv = self.num_key_value_heads * self.head_dim
        token_emb = self.vocab_size * h
        pos_emb = (self.max_position_embeddings * h
                   if self.position_embedding_type == "learned" else 0)

        # Attention: q (h*h) + k,v (h*kv each) + out (h*h) + biases.
        attn = h * h + 2 * h * kv + h * h + 2 * h + 2 * kv

        # Feed-forward: gated variants use 3 matrices, others use 2.
        gated = self.activation in ("swiglu", "geglu")
        if gated:
            # Gated MLPs (SwiGLU/GEGLU) use three bias-free projections.
            ffn = 3 * h * self.intermediate_size
        else:
            ffn = 2 * h * self.intermediate_size + self.intermediate_size + h

        norm_params = 2 * h if self.norm_type == "layernorm" else h
        per_layer = attn + ffn + 2 * norm_params
        all_layers = per_layer * self.num_hidden_layers
        final_norm = norm_params
        output_head = 0 if self.tie_word_embeddings else self.vocab_size * h
        return token_emb + pos_emb + all_layers + final_norm + output_head

    @property
    def total_params_formatted(self) -> str:
        n = self.total_params
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        return f"{n / 1e6:.1f}M"

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YouAIConfig":
        """Create a config from a dict, ignoring unknown keys for forward-compat."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in known}
        return cls(**filtered)

    def replace(self, **kwargs: Any) -> "YouAIConfig":
        """Return a copy with the given fields overridden."""
        data = self.to_dict()
        data.update(kwargs)
        return YouAIConfig.from_dict(data)


# ----------------------------------------------------------------------
# Preset configurations
# ----------------------------------------------------------------------
# Each preset is a modern decoder-only transformer (rotary embeddings, RMSNorm,
# SwiGLU, tied embeddings) — the recipe used by contemporary open LLMs.
_PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=512, max_position_embeddings=512),
    "micro": dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8,
                  intermediate_size=1024, max_position_embeddings=1024),
    "125m": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_position_embeddings=2048),
    "350m": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=4096, max_position_embeddings=2048),
    "750m": dict(hidden_size=1536, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=6144, max_position_embeddings=2048),
    "1.3b": dict(hidden_size=2048, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=8192, max_position_embeddings=2048),
    "2b": dict(hidden_size=2560, num_hidden_layers=32, num_attention_heads=20,
               intermediate_size=10240, max_position_embeddings=2048),
}

# Modern defaults applied on top of every preset.
_MODERN_DEFAULTS = dict(
    activation="swiglu",
    norm_type="rmsnorm",
    position_embedding_type="rotary",
    tie_word_embeddings=True,
    hidden_dropout_prob=0.0,
    attention_dropout_prob=0.0,
)


def list_presets() -> Dict[str, str]:
    """Return a mapping of preset name -> human-readable parameter count."""
    return {name: get_preset_config(name).total_params_formatted for name in _PRESETS}


def get_preset_config(preset: str, modern: bool = True, **overrides: Any) -> YouAIConfig:
    """Get a preset model configuration.

    Args:
        preset: One of ``nano``, ``micro``, ``125m``, ``350m``, ``750m``,
            ``1.3b`` or ``2b``.
        modern: Apply modern architecture defaults (rotary + RMSNorm + SwiGLU).
            Set to ``False`` for a classic GPT-2 style model.
        **overrides: Any config field to override.

    Returns:
        A validated :class:`YouAIConfig`.

    Raises:
        ValueError: If the preset name is not recognised.
    """
    key = preset.lower()
    if key not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {sorted(_PRESETS)}"
        )
    params = dict(_PRESETS[key])
    if modern:
        params.update(_MODERN_DEFAULTS)
    params.update(overrides)
    return YouAIConfig(**params)
