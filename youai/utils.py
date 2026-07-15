"""Shared utilities: logging, reproducibility and device management."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
_LOGGER_NAME = "youai"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return the package logger, configuring a sane default handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        level = os.environ.get("YOUAI_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
    return logger


logger = get_logger()


def set_log_level(level: str) -> None:
    """Set the package-wide log level (e.g. ``"DEBUG"``, ``"WARNING"``)."""
    get_logger().setLevel(getattr(logging, level.upper(), logging.INFO))


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: The random seed.
        deterministic: If ``True``, force deterministic cuDNN algorithms
            (slower, but bit-for-bit reproducible on the same hardware).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is optional
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
def resolve_device(device: Optional[str] = "auto") -> torch.device:
    """Resolve a device string into a concrete, available ``torch.device``.

    ``"auto"`` (or ``None``) selects CUDA, then Apple MPS, then CPU.  A request
    for an unavailable accelerator degrades gracefully to CPU with a warning.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if dev.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return dev


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def format_count(n: int) -> str:
    """Human-readable large-number formatting (1.2M, 3.4B, ...)."""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def human_bytes(num_bytes: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def model_size_bytes(model: torch.nn.Module) -> int:
    """Total size in bytes of a model's parameters and buffers."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes + buffer_bytes
