"""Alignment training: DPO, ORPO, and SimPO.

These methods fine-tune a language model using preference data (chosen vs
rejected responses) without needing a separate reward model. They are the
standard way to align a model with human preferences after supervised
fine-tuning.

Supported algorithms:

* **DPO** (Direct Preference Optimization) — the original; uses a reference
  model to regularise the policy.
* **ORPO** (Odds Ratio Preference Optimization) — no reference model needed;
  adds a preference-aware term to the standard NLL loss.
* **SimPO** (Simple Preference Optimization) — uses average log-probability
  as the reward, no reference model, length-normalised.

Usage::

    from youai.alignment import DPOTrainer, DPOConfig, PreferenceDataset

    # Load your fine-tuned model as the policy
    policy = youai.from_pretrained("gpt2")

    # DPO needs a frozen reference model (auto-created if not provided)
    dataset = PreferenceDataset("preference_data.json", tokenizer)

    config = DPOConfig(beta=0.1, num_epochs=3)
    trainer = DPOTrainer(policy, dataset, config=config)
    trainer.train()

The preference data format is a JSON list of objects with "prompt",
"chosen", and "rejected" keys:

    [
      {"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I don't know"},
      ...
    ]

Or a JSONL file (one JSON object per line).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .utils import get_logger, resolve_device, format_count

logger = get_logger()


# ======================================================================
# Preference dataset
# ======================================================================
class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) preference pairs.

    Accepts a JSON/JSONL file with objects containing "prompt", "chosen",
    and "rejected" keys. Each example is tokenized into:
      - chosen_ids: prompt + chosen response
      - rejected_ids: prompt + rejected response

    The loss is computed only on the response tokens (prompt is masked).
    """

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Try JSON array first, then JSONL.
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            data = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        for item in data:
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if chosen and rejected:
                self.examples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        if not self.examples:
            raise ValueError(f"No valid preference pairs found in {data_path}")

        logger.info("Loaded %d preference pairs from %s", len(self.examples), data_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Tokenize prompt + response pairs.
        chosen_text = f"{prompt}\n{chosen}" if prompt else chosen
        rejected_text = f"{prompt}\n{rejected}" if prompt else rejected

        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        # Build response masks: 1 for response tokens, 0 for prompt/pad.
        prompt_len = len(self.tokenizer.encode(prompt)) if prompt else 0
        chosen_mask = chosen_enc["attention_mask"].squeeze(0).clone()
        rejected_mask = rejected_enc["attention_mask"].squeeze(0).clone()
        # Zero out prompt positions.
        chosen_mask[:prompt_len] = 0
        rejected_mask[:prompt_len] = 0

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "chosen_response_mask": chosen_mask,
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "rejected_response_mask": rejected_mask,
        }


def create_preference_data(
    output_path: str,
    num_examples: int = 100,
) -> str:
    """Generate sample preference data for testing the alignment pipeline."""
    templates = [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "rejected": "I don't know what that is.",
        },
        {
            "prompt": "Explain quantum computing simply.",
            "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing certain calculations to be performed much faster than traditional computers.",
            "rejected": "It's too complicated to explain.",
        },
        {
            "prompt": "How does photosynthesis work?",
            "chosen": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen. Plants capture light energy using chlorophyll and use it to power the chemical reactions.",
            "rejected": "Plants just grow somehow.",
        },
        {
            "prompt": "What causes climate change?",
            "chosen": "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, which trap heat in the atmosphere and raise global temperatures.",
            "rejected": "The weather just changes sometimes.",
        },
        {
            "prompt": "How do vaccines work?",
            "chosen": "Vaccines train your immune system to recognise and fight specific pathogens by introducing a weakened or inactive form of the pathogen, prompting your body to produce antibodies.",
            "rejected": "I'm not sure, ask a doctor.",
        },
    ]

    import random
    data = []
    for _ in range(num_examples):
        t = random.choice(templates)
        data.append(t)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Created %d preference examples at %s", num_examples, output_path)
    return output_path


# ======================================================================
# Alignment configurations
# ======================================================================
@dataclass
class DPOConfig:
    """Configuration for DPO / ORPO / SimPO alignment training."""

    algorithm: str = "dpo"  # "dpo", "orpo", "simpo"
    beta: float = 0.1  # DPO temperature / regularisation strength
    label_smoothing: float = 0.0  # Label smoothing for DPO loss
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    mixed_precision: Optional[str] = None  # "fp16", "bf16"
    max_length: int = 1024
    output_dir: str = "./alignment_output"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42
    # SimPO-specific
    simpo_gamma: float = 1.0  # Target reward margin for SimPO

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================
# Loss functions
# ======================================================================
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: Log-probs of chosen responses under the policy.
        policy_rejected_logps: Log-probs of rejected responses under the policy.
        reference_chosen_logps: Log-probs of chosen responses under the reference.
        reference_rejected_logps: Log-probs of rejected responses under the reference.
        beta: Temperature parameter (higher = more conservative).
        label_smoothing: Label smoothing (0.0 = standard DPO).

    Returns:
        Tuple of (loss, metrics_dict).
    """
    # Log-ratios.
    logits = beta * (
        (policy_chosen_logps - reference_chosen_logps)
        - (policy_rejected_logps - reference_rejected_logps)
    )

    # DPO loss with optional label smoothing.
    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)

    loss = loss.mean()

    # Metrics.
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

    metrics = {
        "loss": loss.item(),
        "reward_margin": reward_margin.item(),
        "accuracy": accuracy.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }
    return loss, metrics


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_nll: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the ORPO loss (no reference model needed).

    ORPO adds an odds-ratio preference term to the standard NLL loss:
    loss = NLL + beta * log(sigma(log_odds))

    Args:
        policy_chosen_logps: Log-probs of chosen responses.
        policy_rejected_logps: Log-probs of rejected responses.
        chosen_nll: Negative log-likelihood of chosen responses.
        beta: Strength of the preference term.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    # Log odds ratio.
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-torch.exp(policy_chosen_logps))
        - torch.log1p(-torch.exp(policy_rejected_logps))
    )
    or_loss = -F.logsigmoid(log_odds).mean()

    # Total loss = NLL + beta * OR.
    loss = chosen_nll.mean() + beta * or_loss

    with torch.no_grad():
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

    metrics = {
        "loss": loss.item(),
        "nll": chosen_nll.mean().item(),
        "or_loss": or_loss.item(),
        "accuracy": accuracy.item(),
    }
    return loss, metrics


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the SimPO loss (no reference model, length-normalised).

    SimPO uses average log-probability as the implicit reward, with a
    target reward margin gamma:
    loss = -log(beta * (avg_logp_chosen - avg_logp_rejected) - gamma)

    Args:
        policy_chosen_logps: Average log-probs of chosen responses.
        policy_rejected_logps: Average log-probs of rejected responses.
        beta: Temperature.
        gamma: Target reward margin.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    logits = beta * (policy_chosen_logps - policy_rejected_logps) - gamma
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "avg_chosen_logp": policy_chosen_logps.mean().item(),
        "avg_rejected_logp": policy_rejected_logps.mean().item(),
    }
    return loss, metrics


# ======================================================================
# Log-probability computation
# ======================================================================
@torch.no_grad()
def _compute_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute average log-probability of the response tokens.

    Args:
        model: The language model.
        input_ids: [batch, seq] token ids.
        attention_mask: [batch, seq] attention mask.
        response_mask: [batch, seq] 1 for response tokens, 0 for prompt/pad.

    Returns:
        [batch] average log-prob per response.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]  # [batch, seq, vocab]

    # Shift for next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()

    # Per-token log-probs.
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Average over response tokens.
    mask_sum = shift_mask.sum(dim=1).clamp(min=1)
    avg_logps = (token_logps * shift_mask).sum(dim=1) / mask_sum

    return avg_logps


# ======================================================================
# DPO Trainer
# ======================================================================
class DPOTrainer:
    """Train a model using DPO, ORPO, or SimPO alignment.

    Args:
        policy: The model to align (will be modified in-place).
        train_dataset: A :class:`PreferenceDataset`.
        reference: Optional frozen reference model. If ``None`` and
            algorithm is ``"dpo"``, a copy of the policy is created.
        config: A :class:`DPOConfig`.
        val_dataset: Optional validation dataset.
        device: Device string.
        callbacks: Optional list of callables.
    """

    def __init__(
        self,
        policy: nn.Module,
        train_dataset: PreferenceDataset,
        reference: Optional[nn.Module] = None,
        config: Optional[DPOConfig] = None,
        val_dataset: Optional[PreferenceDataset] = None,
        device: str = "auto",
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config or DPOConfig()
        self.device = resolve_device(device)
        self.policy = policy.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []

        # Reference model (frozen copy for DPO).
        if self.config.algorithm == "dpo":
            if reference is not None:
                self.reference = reference.to(self.device)
            else:
                logger.info("Creating frozen reference model for DPO.")
                import copy
                self.reference = copy.deepcopy(policy).to(self.device)
            for p in self.reference.parameters():
                p.requires_grad_(False)
            self.reference.eval()
        else:
            self.reference = None  # ORPO/SimPO don't need a reference.

        # DataLoader.
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0, pin_memory=pin, drop_last=True,
        )
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size,
                shuffle=False, num_workers=0, pin_memory=pin,
            )

        # Optimizer.
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler.
        total_steps = len(self.train_loader) * self.config.num_epochs
        total_steps = max(1, total_steps // self.config.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            if self.config.lr_scheduler == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            if self.config.lr_scheduler == "linear":
                return 1 - progress
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision.
        self.amp_dtype = None
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.amp_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
        self.use_amp = self.amp_dtype is not None

        self.global_step = 0
        self.best_accuracy = 0.0

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _compute_preference_loss(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the alignment loss for a batch."""
        cfg = self.config

        # Move batch to device.
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_resp = batch["chosen_response_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_resp = batch["rejected_response_mask"].to(self.device)

        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Policy log-probs.
            policy_chosen_logps = _compute_logprobs(
                self.policy, chosen_ids, chosen_mask, chosen_resp
            )
            policy_rejected_logps = _compute_logprobs(
                self.policy, rejected_ids, rejected_mask, rejected_resp
            )

            if cfg.algorithm == "dpo":
                # Reference log-probs.
                ref_chosen_logps = _compute_logprobs(
                    self.reference, chosen_ids, chosen_mask, chosen_resp
                )
                ref_rejected_logps = _compute_logprobs(
                    self.reference, rejected_ids, rejected_mask, rejected_resp
                )
                loss, metrics = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=cfg.beta, label_smoothing=cfg.label_smoothing,
                )
            elif cfg.algorithm == "orpo":
                # NLL for chosen responses.
                out = self.policy(input_ids=chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
                chosen_nll = out["loss"]
                loss, metrics = orpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    chosen_nll, beta=cfg.beta,
                )
            elif cfg.algorithm == "simpo":
                loss, metrics = simpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    beta=cfg.beta, gamma=cfg.simpo_gamma,
                )
            else:
                raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        return loss, metrics

    def train(self) -> Dict[str, float]:
        """Run alignment training and return final metrics."""
        cfg = self.config
        self.policy.train()
        if self.reference:
            self.reference.eval()

        accum = cfg.gradient_accumulation_steps
        running_metrics: Dict[str, float] = {}
        t0 = time.time()

        logger.info("=" * 56)
        logger.info("YouAI alignment training (%s)", cfg.algorithm.upper())
        logger.info("device=%s | params=%s | beta=%.2f",
                    self.device, format_count(sum(p.numel() for p in self.policy.parameters())),
                    cfg.beta)
        logger.info("epochs=%d | batch=%d | accum=%d | lr=%.2e",
                    cfg.num_epochs, cfg.batch_size, accum, cfg.learning_rate)
        logger.info("=" * 56)

        for epoch in range(cfg.num_epochs):
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(progress):
                loss, metrics = self._compute_preference_loss(batch)
                loss = loss / accum
                loss.backward()

                # Accumulate metrics.
                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v

                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % cfg.logging_steps == 0:
                        avg = {k: v / cfg.logging_steps for k, v in running_metrics.items()}
                        lr = self.scheduler.get_last_lr()[0]
                        progress.set_postfix(
                            loss=f"{avg.get('loss', 0):.4f}",
                            acc=f"{avg.get('accuracy', 0):.2f}",
                            lr=f"{lr:.2e}",
                        )
                        running_metrics = {}

                    if self.global_step % cfg.save_steps == 0:
                        self._save(f"step-{self.global_step}")

        # Final save.
        self._save("final")
        elapsed = time.time() - t0
        logger.info("Alignment finished in %.1fs (%d steps).", elapsed, self.global_step)

        return {
            "global_step": self.global_step,
            "train_time_seconds": elapsed,
            "algorithm": cfg.algorithm,
        }

    @torch.no_grad()
    def evaluate(self) -> Optional[float]:
        """Evaluate on the validation set and return mean accuracy."""
        if not self.val_loader:
            return None
        self.policy.eval()
        total_acc, count = 0.0, 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            _, metrics = self._compute_preference_loss(batch)
            total_acc += metrics.get("accuracy", 0)
            count += 1

        self.policy.train()
        return total_acc / max(1, count)

    def _save(self, name: str):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        with open(os.path.join(path, "alignment_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.debug("Saved alignment checkpoint: %s", path)


# ======================================================================
# High-level API
# ======================================================================
def align(
    model: nn.Module,
    data_path: str,
    algorithm: str = "dpo",
    beta: float = 0.1,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-7,
    output_dir: str = "./alignment_output",
    device: str = "auto",
    **kwargs,
) -> dict:
    """Align a model using DPO, ORPO, or SimPO.

    Args:
        model: The model to align (typically after SFT).
        data_path: Path to preference data (JSON/JSONL with prompt/chosen/rejected).
        algorithm: ``"dpo"``, ``"orpo"``, or ``"simpo"``.
        beta: Temperature / regularisation strength.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        output_dir: Where to save checkpoints.
        device: Device string.
        **kwargs: Additional :class:`DPOConfig` fields.

    Returns:
        Training metrics dict.

    Example::

        import youai

        model = youai.from_pretrained("gpt2")
        youai.train(model, "sft_data.txt", epochs=1, lora=True)  # SFT first
        metrics = youai.align(model, "preference_data.json", algorithm="dpo")
    """
    from .tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    dataset = PreferenceDataset(data_path, tokenizer=tokenizer)
    config = DPOConfig(
        algorithm=algorithm, beta=beta, num_epochs=num_epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        output_dir=output_dir, **kwargs,
    )
    trainer = DPOTrainer(model, dataset, config=config, device=device)
    return trainer.train()
