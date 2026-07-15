"""Sampling utilities shared by the model, inference and streaming APIs.

Keeping the logit-processing logic in one place guarantees that batch
generation, single-prompt generation and token streaming all behave identically
and are all correct (the previous implementation had a subtle top-p bug that
removed the wrong tokens for batched inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Options controlling text generation.

    Args:
        max_new_tokens: Number of tokens to generate.
        temperature: Softmax temperature. ``0`` (or ``do_sample=False``) selects
            greedy decoding.
        top_k: Keep only the ``k`` most probable tokens (0 disables).
        top_p: Nucleus sampling threshold (1.0 disables).
        repetition_penalty: Penalise already-generated tokens (1.0 disables).
        do_sample: Sample from the distribution instead of taking the argmax.
        min_new_tokens: Suppress EOS until at least this many tokens are produced.
        eos_token_id: Stop when this token is generated.
    """

    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = None


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Apply the CTRL-style repetition penalty in-place-safe fashion.

    Args:
        logits: ``[batch, vocab]`` next-token logits.
        generated: ``[batch, seq]`` tokens produced so far.
        penalty: Values > 1 discourage repetition.
    """
    if penalty == 1.0:
        return logits
    for i in range(logits.size(0)):
        unique = torch.unique(generated[i])
        selected = logits[i, unique]
        # Divide positive logits, multiply negative ones (standard formulation).
        logits[i, unique] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def top_k_top_p_filter(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = float("-inf")
) -> torch.Tensor:
    """Filter a ``[batch, vocab]`` logits tensor with top-k and/or nucleus (top-p).

    This uses ``scatter`` so it is correct for arbitrary batch sizes.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, filter_value)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        # Shift so that we always keep at least the most probable token.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(remove, filter_value)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    generated: torch.Tensor,
    config: GenerationConfig,
    step: int,
) -> torch.Tensor:
    """Turn next-token logits into the next token id ``[batch, 1]``."""
    logits = logits.clone()

    if config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)

    # Prevent EOS before ``min_new_tokens`` tokens have been produced.
    if config.eos_token_id is not None and step < config.min_new_tokens:
        logits[:, config.eos_token_id] = float("-inf")

    greedy = not config.do_sample or config.temperature <= 0
    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(config.temperature, 1e-6)
    logits = top_k_top_p_filter(logits, config.top_k, config.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def normalize_stop_sequences(stop: Optional[List[str] | str]) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)
