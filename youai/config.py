"""YouAI model configuration.

The :class:`YouAIConfig` dataclass fully describes a model's architecture.  It is
serialisable to / from JSON so a checkpoint can be reloaded without the original
Python code, and it validates its own fields so mistakes are caught early with a
clear error message instead of an obscure crash deep inside PyTorch.

Supports 12 architecture families with 25+ size presets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Optional


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
        architecture_family: Name of the architecture family this config belongs
            to (e.g. ``llama``, ``qwen2``, ``mistral``). Used by the pretrained
            weight loader to select the correct conversion logic.
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
    architecture_family: str = "youai"
    use_attention_bias: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.activation = str(self.activation).lower()
        self.norm_type = str(self.norm_type).lower()
        self.position_embedding_type = str(self.position_embedding_type).lower()
        self.architecture_family = str(self.architecture_family).lower()
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
        bias = 2 * h + 2 * kv if self.use_attention_bias else 0
        attn = h * h + 2 * h * kv + h * h + bias

        # Feed-forward: gated variants use 3 matrices without bias, others use 2 with bias.
        gated = self.activation in ("swiglu", "geglu")
        if gated:
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
# Preset configurations — size presets
# ----------------------------------------------------------------------
# Each preset is a modern decoder-only transformer (rotary embeddings, RMSNorm,
# SwiGLU, tied embeddings) — the recipe used by contemporary open LLMs.
_PRESETS: Dict[str, Dict[str, Any]] = {
    # ---- Tiny (for testing / learning) ----
    "nano": dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=512, max_position_embeddings=512),
    "micro": dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8,
                  intermediate_size=1024, max_position_embeddings=1024),
    "tiny": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                 intermediate_size=2048, max_position_embeddings=1024),

    # ---- Small–medium (trainable on a single GPU) ----
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

    # ---- Large (multi-GPU / cluster) ----
    "3b": dict(hidden_size=3200, num_hidden_layers=26, num_attention_heads=32,
               intermediate_size=12800, max_position_embeddings=4096),
    "7b": dict(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=11008,
               max_position_embeddings=4096),
    "13b": dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
                num_key_value_heads=8, intermediate_size=13824,
                max_position_embeddings=4096),
    "34b": dict(hidden_size=6656, num_hidden_layers=48, num_attention_heads=52,
                num_key_value_heads=4, intermediate_size=17920,
                max_position_embeddings=4096),
    "70b": dict(hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
                num_key_value_heads=8, intermediate_size=28672,
                max_position_embeddings=4096),
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


# ----------------------------------------------------------------------
# Architecture family presets — match real model configs
# ----------------------------------------------------------------------
# These let you create a YouAIConfig that matches a specific open-source model
# family's architecture exactly. Use ``create_from_family("llama2-7b")`` or
# ``from_pretrained("meta-llama/Llama-2-7b-hf")``.
#
# Each entry: { param_name: value } overrides applied on top of the base preset.
# The ``hf_config_map`` field tells the weight loader which HF config keys to
# read (so we don't hard-code the mapping elsewhere).
ARCHITECTURE_FAMILIES: Dict[str, Dict[str, Any]] = {
    # ---- Meta Llama ----
    "llama": dict(
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0, attention_dropout_prob=0.0,
    ),
    "llama2-7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=11008,
        max_position_embeddings=4096, vocab_size=32000,
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),
    "llama2-13b": dict(
        hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
        num_key_value_heads=8, intermediate_size=13824,
        max_position_embeddings=4096, vocab_size=32000,
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),
    "llama2-70b": dict(
        hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
        num_key_value_heads=8, intermediate_size=28672,
        max_position_embeddings=4096, vocab_size=32000,
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),
    "llama3-8b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=14336,
        max_position_embeddings=8192, vocab_size=128256,
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=500000.0, tie_word_embeddings=False,
    ),
    "llama3-70b": dict(
        hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
        num_key_value_heads=8, intermediate_size=28672,
        max_position_embeddings=8192, vocab_size=128256,
        architecture_family="llama",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=500000.0, tie_word_embeddings=False,
    ),

    # ---- Alibaba Qwen ----
    "qwen": dict(
        architecture_family="qwen2",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=1000000.0, tie_word_embeddings=False,
        layer_norm_eps=1e-6, hidden_dropout_prob=0.0, attention_dropout_prob=0.0,
    ),
    "qwen2-0.5b": dict(
        hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
        num_key_value_heads=2, intermediate_size=4864,
        max_position_embeddings=32768, vocab_size=151936,
        architecture_family="qwen2",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=1000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "qwen2-1.5b": dict(
        hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
        num_key_value_heads=2, intermediate_size=8960,
        max_position_embeddings=32768, vocab_size=151936,
        architecture_family="qwen2",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=1000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "qwen2-7b": dict(
        hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, intermediate_size=18944,
        max_position_embeddings=131072, vocab_size=152064,
        architecture_family="qwen2",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=1000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "qwen2-72b": dict(
        hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
        num_key_value_heads=8, intermediate_size=29568,
        max_position_embeddings=131072, vocab_size=152064,
        architecture_family="qwen2",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=1000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),

    # ---- Mistral ----
    "mistral": dict(
        architecture_family="mistral",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "mistral-7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=14336,
        max_position_embeddings=32768, vocab_size=32000,
        architecture_family="mistral",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),

    # ---- DeepSeek ----
    "deepseek": dict(
        architecture_family="deepseek",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "deepseek-7b": dict(
        hidden_size=4096, num_hidden_layers=30, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=11008,
        max_position_embeddings=4096, vocab_size=100015,
        architecture_family="deepseek",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),

    # ---- Google Gemma ----
    "gemma": dict(
        architecture_family="gemma",
        activation="gelu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=True, layer_norm_eps=1e-6,
    ),
    "gemma-2b": dict(
        hidden_size=2048, num_hidden_layers=18, num_attention_heads=8,
        num_key_value_heads=1, intermediate_size=16384,
        max_position_embeddings=8192, vocab_size=256000,
        architecture_family="gemma",
        activation="gelu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=True, layer_norm_eps=1e-6,
    ),
    "gemma-7b": dict(
        hidden_size=3072, num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=24576,
        max_position_embeddings=8192, vocab_size=256000,
        architecture_family="gemma",
        activation="gelu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=True, layer_norm_eps=1e-6,
    ),

    # ---- Microsoft Phi ----
    "phi": dict(
        architecture_family="phi",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "phi-2": dict(
        hidden_size=2560, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=10240,
        max_position_embeddings=2048, vocab_size=51200,
        architecture_family="phi",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),
    "phi-3-mini": dict(
        hidden_size=3072, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=8192,
        max_position_embeddings=131072, vocab_size=32064,
        architecture_family="phi",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),

    # ---- TII Falcon ----
    "falcon": dict(
        architecture_family="falcon",
        activation="silu", norm_type="layernorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "falcon-7b": dict(
        hidden_size=4544, num_hidden_layers=32, num_attention_heads=71,
        num_key_value_heads=1, intermediate_size=18176,
        max_position_embeddings=2048, vocab_size=65024,
        architecture_family="falcon",
        activation="silu", norm_type="layernorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),
    "falcon-40b": dict(
        hidden_size=8192, num_hidden_layers=60, num_attention_heads=64,
        num_key_value_heads=8, intermediate_size=32768,
        max_position_embeddings=2048, vocab_size=65024,
        architecture_family="falcon",
        activation="silu", norm_type="layernorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),

    # ---- 01.AI Yi ----
    "yi": dict(
        architecture_family="yi",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=5000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "yi-6b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=4, intermediate_size=11008,
        max_position_embeddings=4096, vocab_size=64000,
        architecture_family="yi",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=5000000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),

    # ---- Baichuan ----
    "baichuan": dict(
        architecture_family="baichuan",
        activation="silu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "baichuan-7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=11008,
        max_position_embeddings=4096, vocab_size=64000,
        architecture_family="baichuan",
        activation="silu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),

    # ---- InternLM ----
    "internlm": dict(
        architecture_family="internlm",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),
    "internlm-7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=11008,
        max_position_embeddings=2048, vocab_size=103168,
        architecture_family="internlm",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-6,
    ),

    # ---- MPT (MosaicML) ----
    "mpt": dict(
        architecture_family="mpt",
        activation="silu", norm_type="layernorm", position_embedding_type="alibi",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "mpt-7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=11008,
        max_position_embeddings=2048, vocab_size=50368,
        architecture_family="mpt",
        activation="silu", norm_type="layernorm", position_embedding_type="alibi",
        tie_word_embeddings=False,
    ),

    # ---- StableLM (Stability AI) ----
    "stablelm": dict(
        architecture_family="stablelm",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "stablelm-3b": dict(
        hidden_size=2560, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=6912,
        max_position_embeddings=4096, vocab_size=50304,
        architecture_family="stablelm",
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
        rope_theta=10000.0, tie_word_embeddings=False,
    ),

    # ---- StarCoder (BigCode) ----
    "starcoder": dict(
        architecture_family="starcoder",
        activation="gelu_new", norm_type="layernorm", position_embedding_type="learned",
        rope_theta=10000.0, tie_word_embeddings=False, layer_norm_eps=1e-5,
    ),
    "starcoder-3b": dict(
        hidden_size=3072, num_hidden_layers=24, num_attention_heads=24,
        num_key_value_heads=24, intermediate_size=12288,
        max_position_embeddings=8192, vocab_size=49152,
        architecture_family="starcoder",
        activation="gelu_new", norm_type="layernorm", position_embedding_type="learned",
        tie_word_embeddings=False,
    ),

    # ---- GPT-2 style (classic) ----
    "gpt2-style": dict(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, max_position_embeddings=1024, vocab_size=50257,
        architecture_family="gpt2",
        activation="gelu_new", norm_type="layernorm", position_embedding_type="learned",
        tie_word_embeddings=True, layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1, attention_dropout_prob=0.1,
        use_attention_bias=True,
    ),
}


def list_architecture_families() -> List[str]:
    """Return the names of all supported architecture families."""
    return sorted(set(v["architecture_family"] for v in ARCHITECTURE_FAMILIES.values()))


def list_family_presets() -> Dict[str, str]:
    """Return a mapping of family preset name -> human-readable parameter count."""
    result = {}
    for name, overrides in ARCHITECTURE_FAMILIES.items():
        try:
            cfg = YouAIConfig(**{k: v for k, v in overrides.items()
                                if k in {f.name for f in fields(YouAIConfig)}})
            result[name] = cfg.total_params_formatted
        except Exception:
            result[name] = "?"
    return result


def list_presets() -> Dict[str, str]:
    """Return a mapping of size preset name -> human-readable parameter count."""
    return {name: get_preset_config(name).total_params_formatted for name in _PRESETS}


def get_preset_config(preset: str, modern: bool = True, **overrides: Any) -> YouAIConfig:
    """Get a preset model configuration.

    Args:
        preset: One of the size presets (``nano``, ``micro``, ``125m``, etc.)
            or a family preset (``llama2-7b``, ``qwen2-7b``, ``mistral-7b``, etc.).
        modern: Apply modern architecture defaults (rotary + RMSNorm + SwiGLU).
            Ignored for family presets (they already define their own arch).
        **overrides: Any config field to override.

    Returns:
        A validated :class:`YouAIConfig`.

    Raises:
        ValueError: If the preset name is not recognised.
    """
    key = preset.lower()

    # Check family presets first.
    if key in ARCHITECTURE_FAMILIES:
        params = dict(ARCHITECTURE_FAMILIES[key])
        params.update(overrides)
        return YouAIConfig(**{k: v for k, v in params.items()
                              if k in {f.name for f in fields(YouAIConfig)}})

    if key in _PRESETS:
        params = dict(_PRESETS[key])
        if modern:
            params.update(_MODERN_DEFAULTS)
        params.update(overrides)
        return YouAIConfig(**params)

    raise ValueError(
        f"Unknown preset '{preset}'. Choose from: "
        f"size presets {sorted(_PRESETS)} or family presets {sorted(ARCHITECTURE_FAMILIES)}"
    )


def create_from_family(family: str, **overrides: Any) -> YouAIConfig:
    """Create a config from an architecture family name.

    This is the primary entry point for creating configs that match real
    open-source models.

    Args:
        family: A family preset name (e.g. ``llama2-7b``, ``qwen2-7b``).
        **overrides: Any config field to override.

    Example::

        config = create_from_family("llama2-7b")
        config = create_from_family("mistral-7b", max_position_embeddings=16384)
    """
    return get_preset_config(family, **overrides)
