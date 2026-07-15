"""LoRA / QLoRA parameter-efficient fine-tuning.

LoRA (Low-Rank Adaptation) freezes the base model and trains only tiny rank-``r``
adapter matrices injected into the linear layers. This lets you fine-tune a real
model (e.g. GPT-2 loaded via :func:`youai.from_pretrained_gpt2`) on a laptop,
training <1% of the parameters and saving adapters that are a few megabytes.

QLoRA additionally keeps the frozen base weights in 4-bit precision using
``bitsandbytes`` (CUDA only); on CPU / without bitsandbytes it degrades to
standard LoRA with a clear message.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger

logger = get_logger()

# Attention projections are the standard, effective default target set.
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with a trainable low-rank adapter.

    The forward pass is ``base(x) + scaling * B(A(dropout(x)))`` where ``A`` and
    ``B`` are low-rank; ``B`` is zero-initialised so the adapter starts as a
    no-op and the model initially reproduces the base output exactly.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the base weights; only the adapter trains.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self.scaling * lora_out

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def merge(self) -> nn.Linear:
        """Fold the adapter into a plain ``nn.Linear`` (for fast inference)."""
        merged = nn.Linear(self.in_features, self.out_features,
                            bias=self.base.bias is not None)
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight.data = self.base.weight.data + delta
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        return merged


def _iter_target_parents(model: nn.Module, target_modules: Sequence[str]):
    """Yield (parent_module, child_name, child) for linear layers to adapt."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_modules:
                yield module, child_name, child


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    freeze_base: bool = True,
) -> nn.Module:
    """Inject LoRA adapters into ``model`` in place and return it.

    Args:
        r: LoRA rank (higher = more capacity, more parameters).
        alpha: LoRA scaling numerator (effective scale is ``alpha / r``).
        dropout: Dropout applied to the adapter input.
        target_modules: Names of linear sub-modules to adapt.
        freeze_base: Freeze every non-adapter parameter.
    """
    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    replaced = 0
    for parent, child_name, child in list(_iter_target_parents(model, target_modules)):
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    if replaced == 0:
        raise ValueError(
            f"No target modules {tuple(target_modules)} found. "
            "Check the model architecture or pass different target_modules."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied to %d layers | trainable %s / %s params (%.3f%%)",
        replaced, f"{trainable/1e6:.2f}M", f"{total/1e6:.2f}M", 100 * trainable / total,
    )
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace all :class:`LoRALinear` modules with merged plain linears."""
    for parent, child_name, child in list(_iter_all_lora(model)):
        setattr(parent, child_name, child.merge())
    return model


def _iter_all_lora(model: nn.Module):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                yield module, child_name, child


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only the LoRA adapter parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model: nn.Module, output_dir: str, meta: Optional[dict] = None) -> str:
    """Save only the adapter weights (a few MB) plus metadata."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(lora_state_dict(model), os.path.join(output_dir, "adapter_model.bin"))
    info = {"peft_type": "LORA"}
    if meta:
        info.update(meta)
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Saved LoRA adapter to %s", output_dir)
    return output_dir


def load_lora(model: nn.Module, adapter_dir: str, strict: bool = True) -> nn.Module:
    """Load adapter weights into a model that already has LoRA applied."""
    path = os.path.join(adapter_dir, "adapter_model.bin")
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    logger.info("Loaded LoRA adapter from %s", adapter_dir)
    return model


def apply_qlora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> nn.Module:
    """QLoRA: 4-bit frozen base + LoRA adapters.

    True 4-bit quantization requires ``bitsandbytes`` and a CUDA GPU. When those
    are unavailable this transparently falls back to standard LoRA so code keeps
    working (the adapters are identical; only the base memory footprint differs).
    """
    has_bnb = False
    try:
        import bitsandbytes  # noqa: F401

        has_bnb = torch.cuda.is_available()
    except ImportError:
        has_bnb = False

    if not has_bnb:
        logger.warning(
            "QLoRA 4-bit base needs bitsandbytes + CUDA; falling back to standard "
            "LoRA (fp32 base). Adapters and training are unaffected."
        )
    else:  # pragma: no cover - requires CUDA
        logger.info("QLoRA: quantizing base weights to 4-bit (bitsandbytes).")
        _quantize_base_4bit(model, target_modules)

    return apply_lora(model, r=r, alpha=alpha, dropout=dropout, target_modules=target_modules)


def _quantize_base_4bit(model: nn.Module, target_modules: Sequence[str]):  # pragma: no cover
    """Replace target linears' base weights with bitsandbytes 4-bit linears."""
    import bitsandbytes as bnb

    for parent, child_name, child in list(_iter_target_parents(model, target_modules)):
        qlinear = bnb.nn.Linear4bit(
            child.in_features, child.out_features, bias=child.bias is not None,
            compute_dtype=torch.float16,
        )
        qlinear.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False)
        if child.bias is not None:
            qlinear.bias = child.bias
        setattr(parent, child_name, qlinear)


def trainable_parameter_summary(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total,
            "trainable_percent": round(100 * trainable / total, 4)}
