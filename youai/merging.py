"""Model merging utilities for combining multiple models or adapters.

Supports:

* **LoRA merge** — fold LoRA adapters into base weights (fast inference).
* **Linear merge** — weighted average of two models' parameters.
* **SLERP** — spherical linear interpolation (better than linear for models).
* **DARE** — Drop And REscale (randomly drop delta params, rescale remaining).
* **TIES** — Trim, Elect Sign, Merge (prune small deltas, resolve sign conflicts).
* **Model soups** — average multiple fine-tuned checkpoints.

Usage::

    from youai.merging import merge_models, merge_lora_into_base

    # Merge LoRA adapter into base model
    merged = merge_lora_into_base(model)

    # Linear merge of two models
    merged = merge_models(model_a, model_b, alpha=0.7)  # 70% A, 30% B

    # SLERP merge
    merged = merge_models(model_a, model_b, method="slerp", alpha=0.5)

    # DARE merge (with random pruning)
    merged = merge_models(model_a, model_b, method="dare", alpha=0.5, density=0.2)

    # TIES merge
    merged = merge_models(model_a, model_b, method="ties", alpha=0.5, density=0.2)
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .utils import get_logger

logger = get_logger()


# ======================================================================
# Core merge functions
# ======================================================================
def linear_merge(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Weighted average: (1 - alpha) * A + alpha * B.

    Args:
        state_a: First model's state dict.
        state_b: Second model's state dict.
        alpha: Weight for model B (0.0 = all A, 1.0 = all B).

    Returns:
        Merged state dict.
    """
    merged = {}
    for key in state_a:
        if key in state_b:
            merged[key] = (1 - alpha) * state_a[key].float() + alpha * state_b[key].float()
        else:
            merged[key] = state_a[key]
    return merged


def slerp_merge(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Spherical linear interpolation between two models.

    SLERP preserves the norm of the weight vectors, which tends to give
    better results than linear interpolation for language models.

    Args:
        state_a: First model's state dict.
        state_b: Second model's state dict.
        alpha: Interpolation factor (0.0 = A, 1.0 = B).

    Returns:
        Merged state dict.
    """
    merged = {}
    for key in state_a:
        if key not in state_b:
            merged[key] = state_a[key]
            continue

        a = state_a[key].float().flatten()
        b = state_b[key].float().flatten()

        # Normalise.
        a_norm = torch.norm(a)
        b_norm = torch.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            merged[key] = (1 - alpha) * state_a[key] + alpha * state_b[key]
            continue

        a_unit = a / a_norm
        b_unit = b / b_norm

        # Cosine similarity.
        dot = torch.clamp(torch.dot(a_unit, b_unit), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta.abs() < 1e-6:
            # Vectors are nearly parallel; use linear interpolation.
            merged[key] = (1 - alpha) * state_a[key] + alpha * state_b[key]
            continue

        # SLERP formula.
        coeff_a = torch.sin((1 - alpha) * theta) / torch.sin(theta)
        coeff_b = torch.sin(alpha * theta) / torch.sin(theta)

        # Interpolate norms too.
        merged_norm = (1 - alpha) * a_norm + alpha * b_norm
        merged_flat = coeff_a * a_unit + coeff_b * b_unit
        merged_flat = merged_flat * (merged_norm / torch.norm(merged_flat))

        merged[key] = merged_flat.reshape(state_a[key].shape)

    return merged


def dare_merge(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    density: float = 0.2,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """DARE (Drop And REscale) merge.

    Randomly drops a fraction of the delta parameters (B - A) and rescales
    the remaining ones to preserve the expected value. This reduces
    interference between models.

    Args:
        state_a: Base model state dict.
        state_b: Fine-tuned model state dict.
        alpha: Weight for the pruned delta.
        density: Fraction of delta parameters to keep (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        Merged state dict.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    merged = {}
    for key in state_a:
        if key not in state_b:
            merged[key] = state_a[key]
            continue

        a = state_a[key].float()
        b = state_b[key].float()
        delta = b - a

        # Random mask: keep `density` fraction.
        mask = torch.bernoulli(torch.full_like(delta, density), generator=gen)
        # Rescale to preserve expected value.
        pruned_delta = delta * mask / density

        merged[key] = a + alpha * pruned_delta

    return merged


def ties_merge(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    density: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """TIES (Trim, Elect Sign, Merge) merge.

    1. Trim: remove small-magnitude delta parameters.
    2. Elect Sign: for each parameter, pick the sign that has more support.
    3. Merge: average the sign-aligned deltas.

    Args:
        state_a: Base model state dict.
        state_b: Fine-tuned model state dict.
        alpha: Weight for the merged delta.
        density: Fraction of delta parameters to keep after trimming.

    Returns:
        Merged state dict.
    """
    merged = {}
    for key in state_a:
        if key not in state_b:
            merged[key] = state_a[key]
            continue

        a = state_a[key].float()
        b = state_b[key].float()
        delta = b - a

        # Trim: keep top-k by magnitude.
        flat_delta = delta.abs().flatten()
        k = max(1, int(flat_delta.numel() * density))
        threshold = flat_delta.topk(k).values[-1]
        mask = delta.abs() >= threshold

        # Elect sign: majority sign.
        pos_count = ((delta > 0) & mask).sum().float()
        neg_count = ((delta < 0) & mask).sum().float()
        elected_sign = torch.where(pos_count >= neg_count, 1.0, -1.0)

        # Align signs and merge.
        sign_aligned = torch.where(mask, delta * elected_sign, torch.zeros_like(delta))
        merged_delta = sign_aligned * elected_sign  # restore original signs where aligned

        merged[key] = a + alpha * merged_delta

    return merged


# ======================================================================
# High-level merge API
# ======================================================================
def merge_models(
    model_a: nn.Module,
    model_b: nn.Module,
    method: str = "linear",
    alpha: float = 0.5,
    density: float = 0.2,
    output_path: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """Merge two models using the specified method.

    Args:
        model_a: First model (typically the base).
        model_b: Second model (typically the fine-tuned).
        method: ``"linear"``, ``"slerp"``, ``"dare"``, or ``"ties"``.
        alpha: Weight for model B (0.0 = all A, 1.0 = all B).
        density: Pruning density for DARE/TIES (0.0-1.0).
        output_path: Optional path to save the merged model.
        **kwargs: Additional method-specific arguments.

    Returns:
        A new model with merged weights.

    Example::

        merged = merge_models(base, finetuned, method="slerp", alpha=0.5)
        merged = merge_models(base, finetuned, method="dare", alpha=0.5, density=0.3)
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    merge_fn = {
        "linear": linear_merge,
        "slerp": slerp_merge,
        "dare": dare_merge,
        "ties": ties_merge,
    }.get(method)

    if merge_fn is None:
        raise ValueError(f"Unknown merge method '{method}'. Choose: linear, slerp, dare, ties")

    kwargs.update({"alpha": alpha})
    if method in ("dare", "ties"):
        kwargs["density"] = density

    merged_state = merge_fn(state_a, state_b, **kwargs)

    # Create a new model with merged weights.
    merged = copy.deepcopy(model_a)
    merged.load_state_dict(merged_state, strict=False)

    if output_path:
        merged.save_pretrained(output_path)
        logger.info("Saved merged model to %s", output_path)

    logger.info("Merged models using %s (alpha=%.2f, density=%.2f)", method, alpha, density)
    return merged


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """Fold LoRA adapters into the base weights for fast inference.

    This is equivalent to ``youai.merge_lora(model)`` but also returns
    the model for convenience.

    Args:
        model: A model with LoRA adapters applied.

    Returns:
        The same model with adapters merged into base weights.
    """
    from .lora import merge_lora
    return merge_lora(model)


def model_soup(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
    method: str = "uniform",
) -> nn.Module:
    """Average multiple models' parameters (model soup).

    Args:
        models: List of models to average.
        weights: Optional per-model weights (default: uniform).
        method: ``"uniform"`` (simple average) or ``"linear"`` (weighted).

    Returns:
        A new model with averaged weights.

    Example::

        # Average 3 checkpoints equally
        soup = model_soup([ckpt1, ckpt2, ckpt3])

        # Weighted average
        soup = model_soup([ckpt1, ckpt2], weights=[0.7, 0.3])
    """
    if not models:
        raise ValueError("Need at least one model.")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    if len(weights) != len(models):
        raise ValueError(f"Got {len(models)} models but {len(weights)} weights.")

    # Normalize weights.
    total = sum(weights)
    weights = [w / total for w in weights]

    merged = {}
    for key in models[0].state_dict():
        tensors = []
        for model, w in zip(models, weights):
            if key in model.state_dict():
                tensors.append(w * model.state_dict()[key].float())
        if tensors:
            merged[key] = sum(tensors)

    result = copy.deepcopy(models[0])
    result.load_state_dict(merged, strict=False)
    logger.info("Model soup: averaged %d models", len(models))
    return result


# ======================================================================
# Merge strategy info
# ======================================================================
MERGE_METHODS = {
    "linear": {
        "description": "Simple weighted average of parameters",
        "best_for": "Models with similar architectures and training",
        "params": ["alpha"],
    },
    "slerp": {
        "description": "Spherical linear interpolation (preserves norms)",
        "best_for": "Merging models with different training regimes",
        "params": ["alpha"],
    },
    "dare": {
        "description": "Drop And REscale (randomly prune delta, rescale)",
        "best_for": "Merging many models, reducing interference",
        "params": ["alpha", "density"],
    },
    "ties": {
        "description": "Trim, Elect Sign, Merge (prune, align signs, merge)",
        "best_for": "Merging models with conflicting parameter updates",
        "params": ["alpha", "density"],
    },
}


def list_merge_methods() -> None:
    """Print all available merge methods."""
    print("\nAvailable merge methods")
    print("=" * 60)
    for name, info in MERGE_METHODS.items():
        params = ", ".join(info["params"])
        print(f"  {name:<10} {info['description']}")
        print(f"  {'':10} best for: {info['best_for']}")
        print(f"  {'':10} params: {params}")
    print("=" * 60)
