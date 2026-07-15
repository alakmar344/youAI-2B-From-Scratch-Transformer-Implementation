"""Backward-compatibility shim for the advanced-training API.

The functionality that used to live here now lives in :mod:`youai.trainer`
(a single, consolidated, bug-fixed trainer).  These aliases keep older code and
documentation working.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import Trainer, TrainingConfig, estimate_training_time  # noqa: F401
from .utils import get_logger, resolve_device

logger = get_logger()

# ``MixedPrecisionTrainer`` was the historical name for the trainer.
MixedPrecisionTrainer = Trainer


class LearningRateFinder:
    """Estimate a good learning rate with the exponential LR-range test."""

    def __init__(self, model: nn.Module, train_dataloader: DataLoader,
                 device: str = "auto", min_lr: float = 1e-7, max_lr: float = 1.0,
                 num_steps: int = 100):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps

    def find(self) -> float:
        logger.info("Running learning-rate finder...")
        initial_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.min_lr)
        gamma = (self.max_lr / self.min_lr) ** (1 / max(1, self.num_steps))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        losses, lrs, best = [], [], float("inf")
        self.model.train()
        data_iter = iter(self.train_dataloader)
        for _ in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            optimizer.zero_grad()
            loss = self.model(input_ids=input_ids, labels=labels)["loss"]
            val = loss.item()
            best = min(best, val)
            if val > best * 4:
                break
            loss.backward()
            optimizer.step()
            losses.append(val)
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        self.model.load_state_dict(initial_state)
        if len(losses) < 2:
            return self.min_lr * 10
        gradients = [(losses[i] - losses[i - 1]) / (lrs[i] - lrs[i - 1] + 1e-12)
                     for i in range(1, len(losses))]
        suggested = lrs[gradients.index(min(gradients))]
        logger.info("Suggested learning rate: %.2e", suggested)
        return suggested


__all__ = [
    "MixedPrecisionTrainer",
    "Trainer",
    "TrainingConfig",
    "LearningRateFinder",
    "estimate_training_time",
]
