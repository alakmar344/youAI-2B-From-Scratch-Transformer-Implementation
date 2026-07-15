"""Tokenizer helpers.

A thin, cached wrapper around the GPT-2 byte-pair tokenizer that guarantees a
pad token is set and exposes the special-token ids the rest of the library
relies on.  Centralising this avoids repeatedly re-loading the tokenizer and
keeps special-token handling consistent everywhere.
"""

from __future__ import annotations

from functools import lru_cache

from .utils import get_logger

logger = get_logger()


@lru_cache(maxsize=4)
def get_tokenizer(name: str = "gpt2"):
    """Return a cached HuggingFace tokenizer with a pad token guaranteed.

    Args:
        name: Any tokenizer name understood by ``AutoTokenizer``.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The 'transformers' package is required for tokenization. "
            "Install it with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def special_token_ids(tokenizer) -> dict:
    """Extract bos/eos/pad token ids from a tokenizer with sensible fallbacks."""
    eos = tokenizer.eos_token_id
    return {
        "bos_token_id": tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos,
        "eos_token_id": eos,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
    }
