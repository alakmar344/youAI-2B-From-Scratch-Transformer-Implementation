"""YouAI model architecture.

A clean, modern decoder-only transformer that supports:

* Rotary (RoPE), learned or ALiBi positional information.
* RMSNorm or LayerNorm, pre-norm residual blocks.
* GELU / SiLU / ReLU MLPs and gated SwiGLU / GEGLU MLPs.
* Grouped-query attention (GQA) with a fused, memory-efficient attention
  kernel (PyTorch scaled-dot-product-attention) when available.
* A key/value cache for fast autoregressive generation.
* Weight tying and gradient checkpointing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import YouAIConfig
from .generation import GenerationConfig, sample_next_token


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_norm(config: YouAIConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


# ----------------------------------------------------------------------
# Rotary positional embeddings
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Precomputes and applies rotary position embeddings."""

    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_positions = max_positions
        self._build_cache(max_positions)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0, device=None, dtype=None):
        needed = seq_len + offset
        if needed > self.cos_cached.size(0):
            self._build_cache(needed)
        cos = self.cos_cached[offset: offset + seq_len]
        sin = self.sin_cached[offset: offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [batch, heads, seq, head_dim]; cos/sin: [seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


# ----------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head / grouped-query causal self-attention with an optional KV cache."""

    def __init__(self, config: YouAIConfig, rotary: Optional[RotaryEmbedding]):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rotary = rotary
        self.use_alibi = config.position_embedding_type == "alibi"

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout_p = config.attention_dropout_prob
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # SDPA is available in torch >= 2.0.
        self.use_flash = config.use_flash_attention and hasattr(
            F, "scaled_dot_product_attention"
        )
        if self.use_alibi:
            self.register_buffer(
                "alibi_slopes", self._alibi_slopes(self.num_heads), persistent=False
            )

    @staticmethod
    def _alibi_slopes(num_heads: int) -> torch.Tensor:
        def slopes_pow2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = slopes_pow2(num_heads)
        else:
            closest = 2 ** math.floor(math.log2(num_heads))
            slopes = slopes_pow2(closest)
            extra = slopes_pow2(2 * closest)[0::2][: num_heads - closest]
            slopes = slopes + extra
        return torch.tensor(slopes, dtype=torch.float32)

    def _alibi_bias(self, seq_len: int, offset: int, device, dtype) -> torch.Tensor:
        q_pos = torch.arange(offset, offset + seq_len, device=device)
        k_pos = torch.arange(offset + seq_len, device=device)
        rel = k_pos[None, :] - q_pos[:, None]  # [q, k]
        bias = rel.to(dtype)[None, :, :] * self.alibi_slopes.to(device, dtype)[:, None, None]
        return bias  # [heads, q, k]

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, seq, _ = x.shape
        offset = past_kv[0].size(2) if past_kv is not None else 0

        q = self.q_proj(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            cos, sin = self.rotary(seq, offset=offset, device=x.device, dtype=q.dtype)
            q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        # Expand KV heads to match query heads for grouped-query attention.
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        total_k = k.size(2)
        # A plain forward (no KV cache) is a full causal pass; an incremental
        # decode step (past_kv present) attends to all cached keys. Keying off
        # ``past_kv`` keeps this a Python bool so torch.jit.trace works.
        is_causal = attention_bias is None and not self.use_alibi and past_kv is None

        bias = attention_bias
        if self.use_alibi:
            alibi = self._alibi_bias(seq, offset, x.device, q.dtype).unsqueeze(0)
            bias = alibi if bias is None else bias + alibi

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else bias,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                causal = torch.triu(
                    torch.full((seq, total_k), float("-inf"), device=x.device, dtype=scores.dtype),
                    diagonal=1 + offset,
                )
                scores = scores + causal
            elif bias is not None:
                scores = scores + bias
            probs = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                probs = F.dropout(probs, p=self.dropout_p)
            out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(b, seq, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, present


# ----------------------------------------------------------------------
# Feed-forward
# ----------------------------------------------------------------------
_ACT_FNS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


class FeedForward(nn.Module):
    """Position-wise feed-forward network with optional gating (SwiGLU/GEGLU)."""

    def __init__(self, config: YouAIConfig):
        super().__init__()
        self.gated = config.activation in ("swiglu", "geglu")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.gated:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.act = F.silu if config.activation == "swiglu" else F.gelu
        else:
            self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
            self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
            self.act = _ACT_FNS[config.activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            h = self.act(self.gate_proj(x)) * self.up_proj(x)
            return self.dropout(self.down_proj(h))
        h = self.act(self.dense_in(x))
        return self.dropout(self.dense_out(h))


# ----------------------------------------------------------------------
# Transformer block
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: YouAIConfig, rotary: Optional[RotaryEmbedding]):
        super().__init__()
        self.norm1 = build_norm(config)
        self.attention = Attention(config, rotary)
        self.norm2 = build_norm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, attention_bias, past_kv=None, use_cache=False):
        attn_out, present = self.attention(self.norm1(x), attention_bias, past_kv, use_cache)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, present


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class YouAIModel(nn.Module):
    """A decoder-only transformer language model.

    Args:
        config: A :class:`YouAIConfig` (or preset name string).
    """

    def __init__(self, config: YouAIConfig):
        super().__init__()
        if isinstance(config, str):
            from .config import get_preset_config

            config = get_preset_config(config)
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "learned"
            else None
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        rotary = (
            RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
            if config.position_embedding_type == "rotary"
            else None
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, rotary) for _ in range(config.num_hidden_layers)]
        )
        self.norm_f = build_norm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 / GPT-NeoX trick).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight") or name.endswith("dense_out.weight"):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.num_hidden_layers))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embeddings

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        """Run a forward pass.

        Args:
            input_ids: ``[batch, seq]`` token ids.
            attention_mask: Optional ``[batch, seq]`` mask (1 = keep, 0 = pad).
            labels: Optional ``[batch, seq]`` targets for computing loss. Positions
                equal to ``-100`` or to the pad token are ignored.
            past_key_values: Cached keys/values from a previous step.
            use_cache: Return updated key/value cache for fast generation.

        Returns:
            Dict with ``loss`` (or ``None``), ``logits`` and ``past_key_values``.
        """
        b, seq = input_ids.shape
        offset = past_key_values[0][0].size(2) if past_key_values is not None else 0

        if offset + seq > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {offset + seq} exceeds the model's "
                f"max_position_embeddings ({self.config.max_position_embeddings})."
            )

        hidden = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            pos_ids = torch.arange(offset, offset + seq, device=input_ids.device)
            hidden = hidden + self.position_embeddings(pos_ids)[None, :, :]
        hidden = self.embed_dropout(hidden)

        # Build an additive attention bias from the padding mask if provided.
        attention_bias = None
        if attention_mask is not None:
            total = offset + seq
            if attention_mask.size(1) < total:
                pad = torch.ones(b, total - attention_mask.size(1), device=input_ids.device)
                attention_mask = torch.cat([pad, attention_mask], dim=1)
            key_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
            causal = torch.triu(
                torch.full((seq, total), float("-inf"), device=input_ids.device, dtype=hidden.dtype),
                diagonal=1 + offset,
            )
            attention_bias = key_mask + causal[None, None, :, :]

        presents: List[Tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                hidden, present = checkpoint(
                    layer, hidden, attention_bias, None, False, use_reentrant=False
                )
            else:
                hidden, present = layer(hidden, attention_bias, past, use_cache)
            if use_cache:
                presents.append(present)

        hidden = self.norm_f(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Standard next-token objective. Positions labelled -100 (padding,
            # set by the dataset) are ignored — the model never auto-masks by
            # pad id because pad and eos often share a token id.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": presents}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressively generate tokens.

        Accepts either a :class:`GenerationConfig` or keyword arguments
        (``max_new_tokens``/``max_length``, ``temperature``, ``top_k``,
        ``top_p``, ``repetition_penalty``, ``do_sample``, ``eos_token_id`` ...).
        """
        was_training = self.training
        self.eval()

        if generation_config is None:
            if "max_length" in kwargs and "max_new_tokens" not in kwargs:
                kwargs["max_new_tokens"] = kwargs.pop("max_length")
            kwargs.setdefault("eos_token_id", self.config.eos_token_id)
            valid = GenerationConfig.__dataclass_fields__
            generation_config = GenerationConfig(**{k: v for k, v in kwargs.items() if k in valid})

        cfg = generation_config
        generated = input_ids
        past = None
        max_ctx = self.config.max_position_embeddings
        cur = input_ids

        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for step in range(cfg.max_new_tokens):
            # The KV cache cannot grow beyond the context window; stop cleanly
            # instead of raising. Without a cache we slide the window instead.
            if use_cache and generated.size(1) >= max_ctx:
                break
            # Truncate context to the model's window when not using a cache.
            model_input = cur if (use_cache and past is not None) else generated[:, -max_ctx:]
            out = self.forward(model_input, past_key_values=past, use_cache=use_cache)
            past = out["past_key_values"] if use_cache else None
            next_logits = out["logits"][:, -1, :]

            next_token = sample_next_token(next_logits, generated, cfg, step)
            # Once a sequence has produced EOS, keep emitting EOS (pad) for it.
            if cfg.eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_token, cfg.eos_token_id), next_token
                )

            generated = torch.cat([generated, next_token], dim=1)
            cur = next_token

            if cfg.eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == cfg.eos_token_id)
                if bool(finished.all()):
                    break

        if was_training:
            self.train()
        return generated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str) -> None:
        """Save weights and config to a directory (HF-style)."""
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()},
                   os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, map_location="cpu") -> "YouAIModel":
        """Load a model previously saved with :meth:`save_pretrained` (or a
        training checkpoint)."""
        import json
        import os

        with open(os.path.join(checkpoint_path, "config.json")) as f:
            config = YouAIConfig.from_dict(json.load(f))
        model = cls(config)
        state = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=map_location
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        return model
