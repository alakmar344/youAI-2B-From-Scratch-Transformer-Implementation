"""Load pretrained GPT-2 weights from HuggingFace into a YouAIModel.

This makes YouAI *work out of the box*: instead of training a model from
scratch, you can load real GPT-2 weights (any size) and immediately generate or
fine-tune. The mapping handles GPT-2's quirks — the fused ``c_attn`` QKV matrix
and the ``Conv1D`` (transposed) weight layout — and uses the ``gelu_new``
activation so outputs match HuggingFace to within numerical tolerance.
"""

from __future__ import annotations

from typing import Dict

import torch

from .config import YouAIConfig
from .model import YouAIModel
from .utils import get_logger

logger = get_logger()

# HuggingFace model name -> (hidden, layers, heads).  vocab/positions are read
# from the loaded config, so this is just a friendly allow-list.
GPT2_VARIANTS = {
    "gpt2": (768, 12, 12),
    "gpt2-medium": (1024, 24, 16),
    "gpt2-large": (1280, 36, 20),
    "gpt2-xl": (1600, 48, 25),
    "distilgpt2": (768, 6, 12),
}


def gpt2_config(hf_config) -> YouAIConfig:
    """Build a YouAIConfig that mirrors a HuggingFace GPT-2 config exactly."""
    return YouAIConfig(
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.n_positions,
        hidden_size=hf_config.n_embd,
        num_hidden_layers=hf_config.n_layer,
        num_attention_heads=hf_config.n_head,
        intermediate_size=hf_config.n_embd * 4,
        activation="gelu_new",
        norm_type="layernorm",
        position_embedding_type="learned",
        tie_word_embeddings=True,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        layer_norm_eps=hf_config.layer_norm_epsilon,
        # GPT-2 uses <|endoftext|> (50256) for bos/eos; there is no pad token.
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        pad_token_id=hf_config.eos_token_id,
    )


def _convert_state_dict(hf_state: Dict[str, torch.Tensor], config: YouAIConfig) -> Dict[str, torch.Tensor]:
    """Translate a HuggingFace GPT-2 state dict into YouAIModel parameter names."""
    out: Dict[str, torch.Tensor] = {}
    h = config.hidden_size

    out["token_embeddings.weight"] = hf_state["transformer.wte.weight"]
    out["position_embeddings.weight"] = hf_state["transformer.wpe.weight"]
    out["norm_f.weight"] = hf_state["transformer.ln_f.weight"]
    out["norm_f.bias"] = hf_state["transformer.ln_f.bias"]

    for i in range(config.num_hidden_layers):
        p = f"transformer.h.{i}."
        q = f"layers.{i}."

        out[q + "norm1.weight"] = hf_state[p + "ln_1.weight"]
        out[q + "norm1.bias"] = hf_state[p + "ln_1.bias"]
        out[q + "norm2.weight"] = hf_state[p + "ln_2.weight"]
        out[q + "norm2.bias"] = hf_state[p + "ln_2.bias"]

        # c_attn fuses Q,K,V and is a Conv1D (weight stored [in, out]) -> transpose
        # to nn.Linear layout [out, in] and split into three.
        c_attn_w = hf_state[p + "attn.c_attn.weight"].t()      # [3h, h]
        c_attn_b = hf_state[p + "attn.c_attn.bias"]            # [3h]
        qw, kw, vw = c_attn_w.split(h, dim=0)
        qb, kb, vb = c_attn_b.split(h, dim=0)
        out[q + "attention.q_proj.weight"] = qw
        out[q + "attention.q_proj.bias"] = qb
        out[q + "attention.k_proj.weight"] = kw
        out[q + "attention.k_proj.bias"] = kb
        out[q + "attention.v_proj.weight"] = vw
        out[q + "attention.v_proj.bias"] = vb

        out[q + "attention.o_proj.weight"] = hf_state[p + "attn.c_proj.weight"].t()
        out[q + "attention.o_proj.bias"] = hf_state[p + "attn.c_proj.bias"]

        out[q + "feed_forward.dense_in.weight"] = hf_state[p + "mlp.c_fc.weight"].t()
        out[q + "feed_forward.dense_in.bias"] = hf_state[p + "mlp.c_fc.bias"]
        out[q + "feed_forward.dense_out.weight"] = hf_state[p + "mlp.c_proj.weight"].t()
        out[q + "feed_forward.dense_out.bias"] = hf_state[p + "mlp.c_proj.bias"]

    return out


def from_pretrained_gpt2(model_name: str = "gpt2", device: str = "cpu") -> YouAIModel:
    """Load pretrained GPT-2 weights into a :class:`YouAIModel`.

    Args:
        model_name: ``gpt2``, ``gpt2-medium``, ``gpt2-large``, ``gpt2-xl`` or
            ``distilgpt2`` (any HuggingFace GPT-2 checkpoint actually works).
        device: Device to place the model on.

    Returns:
        A ready-to-use YouAIModel with GPT-2's weights.

    Example::

        model = youai.from_pretrained_gpt2("gpt2")
        print(youai.YouAIInference.from_model(model).generate("Hello, I am")[0])
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install transformers: pip install transformers") from exc

    logger.info("Loading pretrained weights: %s", model_name)
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    config = gpt2_config(hf_model.config)
    model = YouAIModel(config)

    converted = _convert_state_dict(hf_model.state_dict(), config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    # The only "missing" key should be the tied lm_head (it shares wte).
    missing = [m for m in missing if not m.startswith("lm_head")]
    if missing:
        raise RuntimeError(f"Unmapped parameters when loading {model_name}: {missing}")

    model.to(device).eval()
    logger.info("Loaded %s (%s params).", model_name, f"{model.num_parameters/1e6:.0f}M")
    return model
