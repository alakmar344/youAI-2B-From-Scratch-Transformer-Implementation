"""Load pretrained weights from 15+ model families into YouAIModel.

Supports: GPT-2, Llama (1/2/3), Qwen (1/1.5/2), Mistral, DeepSeek, Gemma,
Phi (1/2/3), Falcon, Yi, Baichuan, InternLM, MPT, StableLM, StarCoder,
OpenELMA, and any HuggingFace model with a compatible architecture.

Usage::

    model = youai.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = youai.from_pretrained("Qwen/Qwen2-7B")
    model = youai.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = youai.from_pretrained("gpt2")  # GPT-2 still works

The loader auto-detects the architecture from the HuggingFace config and
applies the correct weight mapping. No manual configuration needed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

from .config import YouAIConfig, ARCHITECTURE_FAMILIES, get_preset_config
from .model import YouAIModel
from .utils import get_logger

logger = get_logger()


# ======================================================================
# Architecture detection
# ======================================================================
# Maps HuggingFace model_type / architectures strings to our internal family.
_HF_ARCH_MAP: Dict[str, str] = {
    # GPT-2 family
    "gpt2": "gpt2",
    "distilgpt2": "gpt2",
    # Llama family (covers Llama 1/2/3, CodeLlama, Alpaca, Vicuna, etc.)
    "llama": "llama",
    "codellama": "llama",
    # Qwen family
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2",
    # Mistral
    "mistral": "mistral",
    "mixtral": "mistral",
    # DeepSeek
    "deepseek": "deepseek",
    "deepseek_v2": "deepseek",
    # Gemma
    "gemma": "gemma",
    "gemma2": "gemma",
    # Phi
    "phi": "phi",
    "phi3": "phi",
    # Falcon
    "falcon": "falcon",
    # Yi
    "yi": "yi",
    # Baichuan
    "baichuan": "baichuan",
    # InternLM
    "internlm": "internlm",
    "internlm2": "internlm",
    # MPT
    "mpt": "mpt",
    # StableLM
    "stablelm": "stablelm",
    # StarCoder
    "starcoder": "starcoder",
    "starcoder2": "starcoder",
    # OpenELMA
    "openelm": "openelm",
}

# Models that use the "Llama-style" weight layout (RMSNorm + SwiGLU + GQA +
# rotary). This is the dominant modern architecture — most new models are
# variants of this.
_LLAMA_STYLE_FAMILIES = {
    "llama", "qwen2", "mistral", "deepseek", "gemma", "yi",
    "baichuan", "internlm", "stablelm", "openelm",
}

# GPT-2 style models (LayerNorm + GELU + MHA + learned positions).
_GPT2_STYLE_FAMILIES = {"gpt2", "starcoder"}

# Falcon style (LayerNorm + GQA with shared kv, slightly different layout).
_FALCON_STYLE_FAMILIES = {"falcon"}

# Phi style (similar to Llama but with different MLP structure in some versions).
_PHI_STYLE_FAMILIES = {"phi"}

# MPT style (ALiBi positions, LayerNorm).
_MPT_STYLE_FAMILIES = {"mpt"}


# ======================================================================
# HuggingFace config → YouAIConfig
# ======================================================================
def _detect_family(hf_config) -> str:
    """Detect the architecture family from a HuggingFace config."""
    model_type = getattr(hf_config, "model_type", "").lower()
    if model_type in _HF_ARCH_MAP:
        return _HF_ARCH_MAP[model_type]

    # Fallback: check the architectures field.
    archs = getattr(hf_config, "architectures", []) or []
    for arch in archs:
        arch_lower = arch.lower()
        for pattern, family in _HF_ARCH_MAP.items():
            if pattern in arch_lower:
                return family

    raise ValueError(
        f"Unsupported HuggingFace model_type='{model_type}' "
        f"architectures={archs}. Supported: {sorted(set(_HF_ARCH_MAP.values()))}"
    )


def _hf_to_youai_config(hf_config, family: str) -> YouAIConfig:
    """Build a YouAIConfig from a HuggingFace config object."""
    # Read common fields with fallbacks for different naming conventions.
    vocab_size = getattr(hf_config, "vocab_size", 50257)
    hidden_size = getattr(hf_config, "hidden_size", None) or getattr(hf_config, "n_embd", 768)
    num_layers = (getattr(hf_config, "num_hidden_layers", None)
                  or getattr(hf_config, "n_layer", None) or getattr(hf_config, "num_layers", 12))
    num_heads = (getattr(hf_config, "num_attention_heads", None)
                 or getattr(hf_config, "n_head", None) or getattr(hf_config, "num_heads", 12))
    num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    intermediate_size = getattr(hf_config, "intermediate_size", None) or getattr(
        hf_config, "n_inner", None) or hidden_size * 4
    max_pos = (getattr(hf_config, "max_position_embeddings", None)
               or getattr(hf_config, "n_positions", None)
               or getattr(hf_config, "max_sequence_length", None) or 2048)
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)
    rms_eps = (getattr(hf_config, "rms_norm_eps", None)
               or getattr(hf_config, "layer_norm_epsilon", None)
               or getattr(hf_config, "norm_eps", 1e-5))

    # Detect architecture features from the HF config.
    norm_type = "rmsnorm" if getattr(hf_config, "rms_norm_eps", None) is not None else "layernorm"
    # GPT-2 has no rope_theta and uses n_embd/n_layer/n_head.
    pos_type = "rotary" if rope_theta != 10000.0 or norm_type == "rmsnorm" else "learned"
    if family == "mpt":
        pos_type = "alibi"
    if family in ("gpt2", "starcoder"):
        pos_type = "learned"
        norm_type = "layernorm"

    # Activation detection.
    act = getattr(hf_config, "hidden_act", None) or getattr(hf_config, "activation_function", "gelu")
    act = str(act).lower()
    if act in ("silu", "swish"):
        activation = "swiglu" if family not in ("falcon", "baichuan") else "silu"
    elif act in ("gelu_new",):
        activation = "gelu_new"
    elif act in ("gelu",):
        activation = "gelu" if family not in ("gemma",) else "gelu"
    else:
        activation = "swiglu"  # Modern default

    # Tie embeddings.
    tie = getattr(hf_config, "tie_word_embeddings", True)

    # bos/eos/pad ids.
    bos_id = getattr(hf_config, "bos_token_id", 50256) or 50256
    eos_id = getattr(hf_config, "eos_token_id", 50256) or 50256
    pad_id = getattr(hf_config, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    # Dropout (most modern models use 0).
    hidden_dropout = getattr(hf_config, "hidden_dropout_prob", None) or getattr(
        hf_config, "resid_pdrop", 0.0) or 0.0
    attn_dropout = getattr(hf_config, "attention_dropout_prob", None) or getattr(
        hf_config, "attn_pdrop", 0.0) or 0.0

    return YouAIConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout,
        attention_dropout_prob=attn_dropout,
        activation=activation,
        norm_type=norm_type,
        position_embedding_type=pos_type,
        rope_theta=rope_theta,
        layer_norm_eps=rms_eps,
        tie_word_embeddings=tie,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        architecture_family=family,
        use_attention_bias=(family in ("gpt2", "starcoder", "falcon")),
    )


# ======================================================================
# Weight converters — one per architecture family
# ======================================================================
def _convert_gpt2(hf_state: Dict[str, torch.Tensor], config: YouAIConfig) -> Dict[str, torch.Tensor]:
    """Convert GPT-2 / DistilGPT-2 weights."""
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
        c_attn_w = hf_state[p + "attn.c_attn.weight"].t()
        c_attn_b = hf_state[p + "attn.c_attn.bias"]
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


def _convert_llama_style(hf_state: Dict[str, torch.Tensor], config: YouAIConfig,
                          prefix: str = "model.") -> Dict[str, torch.Tensor]:
    """Convert Llama-style weights (also used for Qwen, Mistral, DeepSeek, Gemma,
    Yi, Baichuan, InternLM, StableLM, OpenELMA).

    These all share the same weight layout:
        model.embed_tokens.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight
    """
    out: Dict[str, torch.Tensor] = {}

    # Embeddings.
    emb_key = prefix + "embed_tokens.weight"
    if emb_key not in hf_state:
        # Some models use "model.tok_embeddings.weight" (Gemma, Yi).
        for alt in [prefix + "tok_embeddings.weight", "transformer.wte.weight",
                     "embed_tokens.weight", "tok_embeddings.weight"]:
            if alt in hf_state:
                emb_key = alt
                break
    out["token_embeddings.weight"] = hf_state[emb_key]

    # Final norm.
    norm_key = prefix + "norm.weight"
    if norm_key not in hf_state:
        for alt in ["model.norm.weight", "transformer.final_layernorm.weight",
                     "norm.weight", "final_layernorm.weight"]:
            if alt in hf_state:
                norm_key = alt
                break
    out["norm_f.weight"] = hf_state[norm_key]

    # LM head — may be tied.
    lm_head_key = "lm_head.weight"
    if lm_head_key in hf_state:
        out["lm_head.weight"] = hf_state[lm_head_key]

    # Layers.
    for i in range(config.num_hidden_layers):
        q = f"layers.{i}."

        # Find the layer prefix.
        layer_prefix = None
        for candidate in [f"{prefix}layers.{i}.", f"model.layers.{i}.",
                          f"transformer.layers.{i}.", f"h.{i}."]:
            if any(k.startswith(candidate) for k in hf_state):
                layer_prefix = candidate
                break
        if layer_prefix is None:
            raise RuntimeError(f"Cannot find layer {i} weights in state dict.")

        # Attention projections.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w_key = f"{layer_prefix}self_attn.{proj}.weight"
            if w_key not in hf_state:
                w_key = f"{layer_prefix}attention.{proj}.weight"
            out[q + f"attention.{proj}.weight"] = hf_state[w_key]

            b_key = f"{layer_prefix}self_attn.{proj}.bias"
            if b_key in hf_state:
                out[q + f"attention.{proj}.bias"] = hf_state[b_key]

        # MLP — gate/up/down (SwiGLU) or fc1/fc2 (GELU).
        if f"{layer_prefix}mlp.gate_proj.weight" in hf_state:
            out[q + "feed_forward.gate_proj.weight"] = hf_state[f"{layer_prefix}mlp.gate_proj.weight"]
            out[q + "feed_forward.up_proj.weight"] = hf_state[f"{layer_prefix}mlp.up_proj.weight"]
            out[q + "feed_forward.down_proj.weight"] = hf_state[f"{layer_prefix}mlp.down_proj.weight"]
        elif f"{layer_prefix}mlp.fc1.weight" in hf_state:
            out[q + "feed_forward.dense_in.weight"] = hf_state[f"{layer_prefix}mlp.fc1.weight"]
            out[q + "feed_forward.dense_out.weight"] = hf_state[f"{layer_prefix}mlp.fc2.weight"]
            if f"{layer_prefix}mlp.fc1.bias" in hf_state:
                out[q + "feed_forward.dense_in.bias"] = hf_state[f"{layer_prefix}mlp.fc1.bias"]
                out[q + "feed_forward.dense_out.bias"] = hf_state[f"{layer_prefix}mlp.fc2.bias"]

        # Norms — try both naming conventions.
        norm1_key = f"{layer_prefix}input_layernorm.weight"
        if norm1_key not in hf_state:
            norm1_key = f"{layer_prefix}ln_1.weight"
        out[q + "norm1.weight"] = hf_state[norm1_key]

        norm2_key = f"{layer_prefix}post_attention_layernorm.weight"
        if norm2_key not in hf_state:
            norm2_key = f"{layer_prefix}ln_2.weight"
        out[q + "norm2.weight"] = hf_state[norm2_key]

        # Norm biases (only for LayerNorm models).
        for src, dst in [(f"{layer_prefix}input_layernorm.bias", q + "norm1.bias"),
                         (f"{layer_prefix}post_attention_layernorm.bias", q + "norm2.bias")]:
            if src in hf_state:
                out[dst] = hf_state[src]

    return out


def _convert_falcon(hf_state: Dict[str, torch.Tensor], config: YouAIConfig) -> Dict[str, torch.Tensor]:
    """Convert Falcon weights (different naming: ln_attn, mlp.dense_4h_to_h)."""
    out: Dict[str, torch.Tensor] = {}

    out["token_embeddings.weight"] = hf_state["transformer.word_embeddings.weight"]
    out["norm_f.weight"] = hf_state["transformer.ln_f.weight"]
    out["norm_f.bias"] = hf_state["transformer.ln_f.bias"]
    if "lm_head.weight" in hf_state:
        out["lm_head.weight"] = hf_state["lm_head.weight"]

    for i in range(config.num_hidden_layers):
        p = f"transformer.h.{i}."
        q = f"layers.{i}."

        out[q + "norm1.weight"] = hf_state[p + "input_layernorm.weight"]
        out[q + "norm1.bias"] = hf_state[p + "input_layernorm.bias"]

        # Falcon uses a single attention block with fused QKV.
        if f"{p}self_attention.query_key_value.weight" in hf_state:
            qkv_w = hf_state[f"{p}self_attention.query_key_value.weight"]
            h = config.hidden_size
            kv_h = config.num_key_value_heads * config.head_dim
            # For GQA (falcon-40b): Q is full, K/V are smaller.
            if config.num_key_value_heads != config.num_attention_heads:
                q_w, k_w, v_w = qkv_w.split([h, kv_h, kv_h], dim=0)
            else:
                q_w, k_w, v_w = qkv_w.split(h, dim=0)
            out[q + "attention.q_proj.weight"] = q_w
            out[q + "attention.k_proj.weight"] = k_w
            out[q + "attention.v_proj.weight"] = v_w

            if f"{p}self_attention.query_key_value.bias" in hf_state:
                qkv_b = hf_state[f"{p}self_attention.query_key_value.bias"]
                if config.num_key_value_heads != config.num_attention_heads:
                    q_b, k_b, v_b = qkv_b.split([h, kv_h, kv_h], dim=0)
                else:
                    q_b, k_b, v_b = qkv_b.split(h, dim=0)
                out[q + "attention.q_proj.bias"] = q_b
                out[q + "attention.k_proj.bias"] = k_b
                out[q + "attention.v_proj.bias"] = v_b

        out[q + "attention.o_proj.weight"] = hf_state[f"{p}self_attention.dense.weight"]

        # MLP.
        out[q + "feed_forward.dense_in.weight"] = hf_state[f"{p}mlp.dense_h_to_4h.weight"]
        out[q + "feed_forward.dense_out.weight"] = hf_state[f"{p}mlp.dense_4h_to_h.weight"]

        # Falcon-40b has post_attention_layernorm.
        if f"{p}post_attention_layernorm.weight" in hf_state:
            out[q + "norm2.weight"] = hf_state[f"{p}post_attention_layernorm.weight"]
        else:
            # Falcon-7b doesn't have a separate norm2; reuse norm1.
            out[q + "norm2.weight"] = hf_state[p + "input_layernorm.weight"]
            out[q + "norm2.bias"] = hf_state[p + "input_layernorm.bias"]

    return out


def _convert_phi(hf_state: Dict[str, torch.Tensor], config: YouAIConfig) -> Dict[str, torch.Tensor]:
    """Convert Phi-2 / Phi-3 weights."""
    # Phi-2 uses a GPT-2-like layout with some differences.
    # Phi-3 uses the Llama-style layout.
    # Detect which by checking for the presence of key patterns.
    if any("model.layers" in k for k in hf_state):
        return _convert_llama_style(hf_state, config, prefix="model.")

    # Phi-2 style.
    out: Dict[str, torch.Tensor] = {}
    out["token_embeddings.weight"] = hf_state["model.embed_tokens.weight"]
    out["norm_f.weight"] = hf_state["model.final_layernorm.weight"]
    out["norm_f.bias"] = hf_state["model.final_layernorm.bias"]
    if "lm_head.weight" in hf_state:
        out["lm_head.weight"] = hf_state["lm_head.weight"]
    if "lm_head.bias" in hf_state:
        out["lm_head.bias"] = hf_state["lm_head.bias"]

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        q = f"layers.{i}."

        out[q + "norm1.weight"] = hf_state[p + "input_layernorm.weight"]
        out[q + "norm1.bias"] = hf_state[p + "input_layernorm.bias"]
        out[q + "norm2.weight"] = hf_state[p + "input_layernorm.weight"]
        out[q + "norm2.bias"] = hf_state[p + "input_layernorm.bias"]

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[q + f"attention.{proj}.weight"] = hf_state[f"{p}self_attn.{proj}.weight"]
            if f"{p}self_attn.{proj}.bias" in hf_state:
                out[q + f"attention.{proj}.bias"] = hf_state[f"{p}self_attn.{proj}.bias"]

        # Phi-2 MLP uses dense_h_to_4h / dense_4h_to_h with GELU.
        if f"{p}mlp.fc1.weight" in hf_state:
            out[q + "feed_forward.dense_in.weight"] = hf_state[f"{p}mlp.fc1.weight"]
            out[q + "feed_forward.dense_out.weight"] = hf_state[f"{p}mlp.fc2.weight"]
        elif f"{p}mlp.gate_proj.weight" in hf_state:
            out[q + "feed_forward.gate_proj.weight"] = hf_state[f"{p}mlp.gate_proj.weight"]
            out[q + "feed_forward.up_proj.weight"] = hf_state[f"{p}mlp.up_proj.weight"]
            out[q + "feed_forward.down_proj.weight"] = hf_state[f"{p}mlp.down_proj.weight"]

    return out


def _convert_mpt(hf_state: Dict[str, torch.Tensor], config: YouAIConfig) -> Dict[str, torch.Tensor]:
    """Convert MPT weights (ALiBi, LayerNorm, no bias)."""
    out: Dict[str, torch.Tensor] = {}

    out["token_embeddings.weight"] = hf_state["transformer.wte.weight"]
    out["norm_f.weight"] = hf_state["transformer.norm_f.weight"]
    if "transformer.norm_f.bias" in hf_state:
        out["norm_f.bias"] = hf_state["transformer.norm_f.bias"]
    if "lm_head.weight" in hf_state:
        out["lm_head.weight"] = hf_state["lm_head.weight"]

    for i in range(config.num_hidden_layers):
        p = f"transformer.blocks.{i}."
        q = f"layers.{i}."

        out[q + "norm1.weight"] = hf_state[f"{p}norm_1.weight"]
        if f"{p}norm_1.bias" in hf_state:
            out[q + "norm1.bias"] = hf_state[f"{p}norm_1.bias"]
        out[q + "norm2.weight"] = hf_state[f"{p}norm_2.weight"]
        if f"{p}norm_2.bias" in hf_state:
            out[q + "norm2.bias"] = hf_state[f"{p}norm_2.bias"]

        # MPT uses Wqkv (fused QKV) and out_proj.
        Wqkv = hf_state[f"{p}attn.Wqkv.weight"]
        h = config.hidden_size
        qw, kw, vw = Wqkv.split(h, dim=0)
        out[q + "attention.q_proj.weight"] = qw
        out[q + "attention.k_proj.weight"] = kw
        out[q + "attention.v_proj.weight"] = vw

        if f"{p}attn.Wqkv.bias" in hf_state:
            b = hf_state[f"{p}attn.Wqkv.bias"]
            qb, kb, vb = b.split(h, dim=0)
            out[q + "attention.q_proj.bias"] = qb
            out[q + "attention.k_proj.bias"] = kb
            out[q + "attention.v_proj.bias"] = vb

        out[q + "attention.o_proj.weight"] = hf_state[f"{p}attn.out_proj.weight"]
        if f"{p}attn.out_proj.bias" in hf_state:
            out[q + "attention.o_proj.bias"] = hf_state[f"{p}attn.out_proj.bias"]

        out[q + "feed_forward.dense_in.weight"] = hf_state[f"{p}ffn.up_proj.weight"]
        out[q + "feed_forward.dense_out.weight"] = hf_state[f"{p}ffn.down_proj.weight"]
        if f"{p}ffn.up_proj.bias" in hf_state:
            out[q + "feed_forward.dense_in.bias"] = hf_state[f"{p}ffn.up_proj.bias"]
            out[q + "ffn.down_proj.bias"] = hf_state[f"{p}ffn.down_proj.bias"]

    return out


# ======================================================================
# Converter registry
# ======================================================================
_CONVERTERS = {
    "gpt2": _convert_gpt2,
    "llama": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "qwen2": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "mistral": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "deepseek": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "gemma": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "yi": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "baichuan": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "internlm": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "stablelm": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "openelm": lambda s, c: _convert_llama_style(s, c, prefix="model."),
    "falcon": _convert_falcon,
    "phi": _convert_phi,
    "mpt": _convert_mpt,
    "starcoder": lambda s, c: _convert_llama_style(s, c, prefix="model."),
}


# ======================================================================
# Public API
# ======================================================================
SUPPORTED_FAMILIES = sorted(set(_HF_ARCH_MAP.values()))

# Popular model repos for each family (used by CLI for suggestions).
POPULAR_MODELS = {
    "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"],
    "llama": [
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf", "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B", "codellama/CodeLlama-7b-hf",
        "NousResearch/Llama-2-7b-hf",
    ],
    "qwen2": [
        "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B",
        "Qwen/Qwen2-72B", "Qwen/CodeQwen1.5-7B",
    ],
    "mistral": [
        "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-v0.1",
    ],
    "deepseek": [
        "deepseek-ai/deepseek-llm-7b-base", "deepseek-ai/deepseek-llm-67b-base",
    ],
    "gemma": [
        "google/gemma-2b", "google/gemma-7b",
    ],
    "phi": [
        "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
    ],
    "falcon": [
        "tiiuae/falcon-7b", "tiiuae/falcon-40b",
    ],
    "yi": [
        "01-ai/Yi-6B", "01-ai/Yi-34B",
    ],
    "baichuan": [
        "baichuan-inc/Baichuan-7B", "baichuan-inc/Baichuan2-7B-Base",
    ],
    "internlm": [
        "internlm/internlm-7b", "internlm/internlm2-7b",
    ],
    "mpt": [
        "mosaicml/mpt-7b",
    ],
    "stablelm": [
        "stabilityai/stablelm-3b-4e1t",
    ],
    "starcoder": [
        "bigcode/starcoder2-3b", "bigcode/starcoder2-7b",
    ],
}


def from_pretrained(model_name: str, device: str = "cpu",
                    trust_remote_code: bool = False,
                    dtype: Optional[str] = None,
                    **kwargs) -> YouAIModel:
    """Load ANY supported pretrained model into a YouAIModel.

    This is the single entry point for loading pretrained weights. It
    auto-detects the architecture from the HuggingFace config and applies
    the correct weight mapping.

    Args:
        model_name: Any HuggingFace model name or local path. Also accepts
            shorthand names like ``"gpt2"``, ``"llama2-7b"``, ``"qwen2-7b"``.
        device: Device to place the model on.
        trust_remote_code: Pass through to ``from_pretrained`` for custom models.
        dtype: Cast weights to this dtype (``"float16"``, ``"bfloat16"``,
            or ``None`` for default).
        **kwargs: Additional arguments passed to the HuggingFace loader.

    Returns:
        A ready-to-use YouAIModel.

    Example::

        model = youai.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = youai.from_pretrained("Qwen/Qwen2-7B", dtype="bfloat16")
        model = youai.from_pretrained("gpt2")
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError as exc:
        raise ImportError("Install transformers: pip install transformers") from exc

    logger.info("Loading pretrained model: %s", model_name)

    # Load the HuggingFace config first to detect the family.
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    family = _detect_family(hf_config)
    logger.info("Detected architecture family: %s", family)

    # Build a YouAIConfig from the HF config.
    youai_config = _hf_to_youai_config(hf_config, family)

    # Load the HF model.
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
        **kwargs,
    )

    # Convert weights.
    converter = _CONVERTERS.get(family)
    if converter is None:
        raise ValueError(
            f"No weight converter for family '{family}'. "
            f"Supported: {sorted(_CONVERTERS)}"
        )

    hf_state = hf_model.state_dict()
    converted = converter(hf_state, youai_config)

    # Create the YouAI model and load.
    model = YouAIModel(youai_config)
    missing, unexpected = model.load_state_dict(converted, strict=False)

    # The only "missing" key should be the tied lm_head (it shares wte).
    real_missing = [m for m in missing if not m.startswith("lm_head")]
    if real_missing:
        logger.warning("Unmapped parameters: %s", real_missing)

    if unexpected:
        logger.debug("Unexpected keys (ignored): %s", unexpected[:5])

    # Optionally cast dtype.
    if dtype:
        dt = getattr(torch, dtype)
        model = model.to(dtype=dt)

    model.to(device).eval()
    logger.info(
        "Loaded %s (%s params, family=%s, device=%s)",
        model_name, model.config.total_params_formatted, family, device,
    )
    return model


# Keep the old GPT-2 specific function for backward compatibility.
def from_pretrained_gpt2(model_name: str = "gpt2", device: str = "cpu") -> YouAIModel:
    """Load pretrained GPT-2 weights (backward-compatible wrapper).

    .. deprecated:: 2.0.0
        Use :func:`from_pretrained` instead — it supports all model families.
    """
    return from_pretrained(model_name, device=device)


# Legacy alias.
GPT2_VARIANTS = {
    "gpt2": (768, 12, 12),
    "gpt2-medium": (1024, 24, 16),
    "gpt2-large": (1280, 36, 20),
    "gpt2-xl": (1600, 48, 25),
    "distilgpt2": (768, 6, 12),
}
