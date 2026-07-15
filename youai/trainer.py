"""Training loop for YouAI models.

A single, well-featured trainer that supports:

* Automatic mixed precision (fp16 / bf16) with correct gradient scaling.
* Gradient accumulation and gradient clipping.
* Warmup + cosine / linear / constant learning-rate schedules (one scheduler,
  stepped every optimizer step — no manual LR hacks).
* Periodic evaluation, best-checkpoint tracking and early stopping.
* Full checkpoint save / resume (model + optimizer + scheduler + scaler + step).
* Optional Weights & Biases logging and user callbacks.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import get_logger, resolve_device, format_count

logger = get_logger()


@dataclass
class TrainingConfig:
    """Hyper-parameters controlling a training run."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    warmup_steps: Optional[int] = None
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    mixed_precision: Optional[str] = None  # "fp16" | "bf16" | None
    gradient_checkpointing: bool = False
    max_steps: Optional[int] = None
    eval_steps: int = 200
    save_steps: int = 500
    log_steps: int = 10
    early_stopping_patience: Optional[int] = None
    save_total_limit: Optional[int] = 3
    output_dir: str = "./checkpoints"
    seed: int = 42
    use_wandb: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _build_param_groups(model: nn.Module, weight_decay: float):
    """Apply weight decay only to 2-D+ weights (not biases / norms / embeddings)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Trainer:
    """Train a :class:`YouAIModel`.

    Args:
        model: The model to train.
        train_dataloader: Training data.
        val_dataloader: Optional validation data.
        config: A :class:`TrainingConfig`; individual fields may also be passed
            as keyword arguments for convenience.
        device: Device string (``"auto"`` by default).
        callbacks: Optional list of callables invoked as ``cb(trainer, metrics)``
            after each logging step.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "auto",
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ):
        self.config = config or TrainingConfig()
        for key, value in kwargs.items():  # allow Trainer(model, ..., learning_rate=1e-4)
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []

        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        self.optimizer = torch.optim.AdamW(
            _build_param_groups(model, self.config.weight_decay),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        self.total_steps = self._compute_total_steps()
        self.warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps is not None
            else int(self.total_steps * self.config.warmup_ratio)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)

        # Mixed precision setup (new torch.amp API, no deprecation warnings).
        self.amp_dtype = None
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.amp_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
                logger.warning("bf16 not supported on this GPU; training in fp32.")
            else:
                self.amp_dtype = torch.bfloat16
        self.use_amp = self.amp_dtype is not None
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype == torch.float16
        )
        if self.use_amp:
            logger.info("Mixed precision: %s", self.config.mixed_precision)

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self._saved_checkpoints: List[str] = []

        if self.config.use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------
    def _compute_total_steps(self) -> int:
        steps_per_epoch = max(1, len(self.train_dataloader) // self.config.gradient_accumulation_steps)
        total = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps:
            total = min(total, self.config.max_steps)
        return max(1, total)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        min_ratio = self.config.min_lr_ratio
        if self.config.lr_scheduler == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if self.config.lr_scheduler == "linear":
            return min_ratio + (1 - min_ratio) * (1 - progress)
        return 1.0  # constant

    def _init_wandb(self):
        try:
            import wandb

            wandb.init(project="youai", config=self.config.to_dict())
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging.")
            self.config.use_wandb = False
            self._wandb = None

    def _log(self, metrics: Dict[str, float]):
        if self.config.use_wandb and getattr(self, "_wandb", None):
            self._wandb.log(metrics, step=self.global_step)
        for cb in self.callbacks:
            cb(self, metrics)

    # ------------------------------------------------------------------
    def _forward_loss(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out["loss"]

    @torch.no_grad()
    def evaluate(self) -> Optional[float]:
        """Return mean validation loss (or ``None`` if there is no val set)."""
        if not self.val_dataloader:
            return None
        self.model.eval()
        total, count = 0.0, 0
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            loss = self._forward_loss(batch)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(1, count)

    def train(self) -> Dict[str, float]:
        """Run the full training loop and return final metrics."""
        cfg = self.config
        self._print_banner()
        self.model.train()
        accum = cfg.gradient_accumulation_steps
        running_loss, t0 = 0.0, time.time()
        stop = False

        for epoch in range(self.start_epoch, cfg.num_epochs):
            progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(progress):
                loss = self._forward_loss(batch) / accum
                self.scaler.scale(loss).backward()
                running_loss += loss.item() * accum

                if (i + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % cfg.log_steps == 0:
                        avg = running_loss / (cfg.log_steps * accum)
                        lr = self.scheduler.get_last_lr()[0]
                        ppl = math.exp(min(avg, 20))
                        progress.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}")
                        self._log({"train/loss": avg, "train/lr": lr, "train/perplexity": ppl})
                        running_loss = 0.0

                    if self.val_dataloader and self.global_step % cfg.eval_steps == 0:
                        if self._do_eval_and_maybe_stop():
                            stop = True
                            break

                    if self.global_step % cfg.save_steps == 0:
                        self.save_checkpoint(f"step-{self.global_step}")

                    if cfg.max_steps and self.global_step >= cfg.max_steps:
                        stop = True
                        break

            if not stop:
                self.save_checkpoint(f"epoch-{epoch + 1}")
            if stop:
                break

        self.save_checkpoint("final")
        elapsed = time.time() - t0
        final_val = self.evaluate()
        logger.info("Training finished in %.1fs (%d steps).", elapsed, self.global_step)
        return {
            "global_step": self.global_step,
            "train_time_seconds": elapsed,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
            "final_val_loss": final_val,
        }

    def _do_eval_and_maybe_stop(self) -> bool:
        val_loss = self.evaluate()
        if val_loss is None:
            return False
        ppl = math.exp(min(val_loss, 20))
        logger.info("step %d | val_loss %.4f | val_ppl %.2f", self.global_step, val_loss, ppl)
        self._log({"val/loss": val_loss, "val/perplexity": ppl})
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint("best")
        else:
            self.patience_counter += 1
            if (self.config.early_stopping_patience
                    and self.patience_counter >= self.config.early_stopping_patience):
                logger.info("Early stopping at step %d.", self.global_step)
                return True
        return False

    # ------------------------------------------------------------------
    def _print_banner(self):
        logger.info("=" * 56)
        logger.info("YouAI training")
        logger.info("device=%s | params=%s | precision=%s",
                    self.device, format_count(sum(p.numel() for p in self.model.parameters())),
                    self.config.mixed_precision or "fp32")
        logger.info("epochs=%d | total_steps=%d | warmup=%d | accum=%d",
                    self.config.num_epochs, self.total_steps, self.warmup_steps,
                    self.config.gradient_accumulation_steps)
        logger.info("=" * 56)

    def save_checkpoint(self, name: str):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            os.path.join(path, "pytorch_model.bin"),
        )
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        with open(os.path.join(path, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Enforce save_total_limit for transient (step-*) checkpoints.
        if name.startswith("step-") and self.config.save_total_limit:
            self._saved_checkpoints.append(path)
            while len(self._saved_checkpoints) > self.config.save_total_limit:
                old = self._saved_checkpoints.pop(0)
                self._safe_rmtree(old)
        logger.debug("Saved checkpoint: %s", path)

    @staticmethod
    def _safe_rmtree(path: str):
        import shutil

        try:
            shutil.rmtree(path)
        except OSError:
            pass

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """Load model (and, if resuming, optimizer/scheduler/scaler/step)."""
        ckpt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if resume_training:
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info("Resumed from step %d.", self.global_step)


def estimate_training_time(model, dataset_size: int, batch_size: int = 8,
                           num_epochs: int = 3, device: str = "cuda") -> Dict[str, object]:
    """Rough estimate of training time, memory and cloud cost."""
    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4
    param_gb = num_params * bytes_per_param / 1024 ** 3
    # params + grads + Adam(m,v) ≈ 4x parameter memory, plus activations headroom.
    total_gb = param_gb * 4 * 1.3

    steps_per_epoch = max(1, dataset_size // batch_size)
    total_steps = steps_per_epoch * num_epochs
    ms_per_step = max(10.0, 50.0 * (num_params / 125e6))
    if device != "cuda":
        ms_per_step *= 12
    total_hours = (total_steps * ms_per_step) / 1000 / 3600

    return {
        "parameters": num_params,
        "parameters_formatted": format_count(num_params),
        "gpu_memory_gb": round(total_gb, 2),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "estimated_time_hours": round(total_hours, 2),
        "estimated_time_days": round(total_hours / 24, 2),
        "estimated_cost_usd": round(total_hours * 1.10, 2),  # ~A100 spot price
    }
