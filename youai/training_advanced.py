"""YouAI Advanced Training Features

This module provides advanced training capabilities including:
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Training resumption from checkpoints
- Early stopping
- Learning rate finder
"""

import os
import json
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Advanced training configuration.
    
    Args:
        mixed_precision: Enable mixed precision ('fp16', 'bf16', or None)
        gradient_checkpointing: Enable gradient checkpointing (saves memory)
        early_stopping_patience: Stop if val loss doesn't improve for N evals
        max_steps: Stop after N steps (overrides epochs if set)
        log_steps: Log metrics every N steps
        weight_decay: Weight decay for AdamW optimizer
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        adam_epsilon: Adam epsilon parameter
        max_grad_norm: Maximum gradient norm for clipping
        lr_scheduler: Learning rate scheduler type ('cosine', 'linear', 'constant')
        warmup_ratio: Ratio of total steps for warmup
    """
    mixed_precision: Optional[str] = None  # 'fp16', 'bf16', or None
    gradient_checkpointing: bool = False
    early_stopping_patience: Optional[int] = None
    max_steps: Optional[int] = None
    log_steps: int = 10
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1


class MixedPrecisionTrainer:
    """Training with automatic mixed precision for faster training on modern GPUs.
    
    Uses PyTorch's native AMP (Automatic Mixed Precision) for:
    - Up to 2x faster training on Volta/Turing/Ampere GPUs
    - Reduced memory usage (can use larger batch sizes)
    - Same model quality as full precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        learning_rate: float = 3e-4,
        num_epochs: int = 3,
        output_dir: str = "./checkpoints",
        device: str = "cuda",
    ):
        self.config = config or TrainingConfig()
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        # Setup mixed precision scaler
        self.scaler = None
        self.use_amp = False
        if self.config.mixed_precision == 'fp16' and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            print("Mixed precision training enabled (FP16)")
        elif self.config.mixed_precision == 'bf16' and self.device.type == 'cuda':
            if torch.cuda.is_bf16_supported():
                self.use_amp = True
                print("Mixed precision training enabled (BF16)")
            else:
                print("BF16 not supported on this GPU, using FP32")
        
        # Setup gradient checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            else:
                print("Warning: Model does not support gradient checkpointing")
        
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)
        
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps - warmup_steps
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=total_steps
            )
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def _get_device(self, device: str) -> torch.device:
        """Auto-detect best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    
    def _train_step(self, batch: dict) -> float:
        """Single training step with optional mixed precision."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        # Update learning rate
        if self.global_step < self.warmup_steps:
            lr_scale = self.global_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * param_group['initial_lr']
        else:
            self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> Optional[float]:
        """Evaluate on validation set."""
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, labels=labels)
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, 'pytorch_model.bin'))
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed training from step {self.global_step}")
    
    def train(self):
        """Run training loop with all advanced features."""
        print("\n" + "=" * 60)
        print("YouAI Advanced Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.config.mixed_precision or 'FP32'}")
        print(f"Gradient Checkpointing: {self.config.gradient_checkpointing}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Warmup Steps: {self.warmup_steps}")
        print("=" * 60 + "\n")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress):
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    print(f"\nReached max steps ({self.config.max_steps})")
                    break
                
                loss = self._train_step(batch)
                self.global_step += 1
                epoch_loss += loss
                
                # Update progress bar
                avg_loss = epoch_loss / (step + 1)
                perplexity = math.exp(min(avg_loss, 10))
                progress.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Evaluation
                if self.val_dataloader and self.global_step % 100 == 0:
                    val_loss = self.evaluate()
                    
                    if val_loss is not None:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best')
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                        
                        # Early stopping
                        if (self.config.early_stopping_patience and 
                            self.patience_counter >= self.config.early_stopping_patience):
                            print(f"\nEarly stopping at step {self.global_step}")
                            break
                
                # Save periodic checkpoint
                if self.global_step % 500 == 0:
                    self.save_checkpoint(f'step-{self.global_step}')
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch+1} completed. Loss: {avg_epoch_loss:.4f}")
            self.save_checkpoint(f'epoch-{epoch+1}')
        
        self.save_checkpoint('final')
        print("\nTraining completed!")


class LearningRateFinder:
    """Find the optimal learning rate for your model and data.
    
    Usage:
        finder = LearningRateFinder(model, train_dataloader)
        suggested_lr = finder.find()
        print(f"Suggested learning rate: {suggested_lr}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        device: str = "cuda",
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_steps: int = 100,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.device = device
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
    
    def find(self) -> float:
        """Run learning rate finder and return suggested LR.
        
        Returns:
            Suggested learning rate (where loss decreases fastest)
        """
        print("Running learning rate finder...")
        
        # Save initial model state
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.min_lr)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(self.max_lr / self.min_lr) ** (1 / self.num_steps)
        )
        
        losses = []
        lrs = []
        best_loss = float('inf')
        
        self.model.train()
        data_iter = iter(self.train_dataloader)
        
        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            # Stop if loss explodes
            if loss.item() > best_loss * 4:
                break
            
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # Restore initial model state
        self.model.load_state_dict(initial_state)
        
        # Find LR with steepest loss decrease
        if len(losses) < 2:
            return self.min_lr * 10
        
        # Calculate loss gradient
        gradients = []
        for i in range(1, len(losses)):
            gradients.append((losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1]))
        
        # Find where gradient is most negative (steepest decrease)
        best_idx = gradients.index(min(gradients))
        suggested_lr = lrs[best_idx]
        
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr


def estimate_training_time(
    model: nn.Module,
    dataset_size: int,
    batch_size: int = 8,
    num_epochs: int = 3,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Estimate training time and resource requirements.
    
    Args:
        model: The model to train
        dataset_size: Number of training examples
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Training device
        
    Returns:
        Dictionary with estimates for time, memory, and cost
    """
    num_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    
    # Estimate optimizer memory (Adam has 2x param overhead)
    optimizer_memory_gb = param_memory_gb * 2
    
    # Estimate gradient memory
    gradient_memory_gb = param_memory_gb
    
    # Total GPU memory needed
    total_memory_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb
    
    # Estimate steps
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    # Estimate time per step (rough)
    if device == "cuda":
        # Assume ~50ms per step for 125M model, scale linearly with params
        ms_per_step = 50 * (num_params / 125_000_000)
        if ms_per_step < 10:
            ms_per_step = 10
    else:
        # CPU is ~10x slower
        ms_per_step = 500 * (num_params / 125_000_000)
    
    total_seconds = (total_steps * ms_per_step) / 1000
    
    # Estimate cloud cost (Lambda Labs A100 pricing)
    gpu_cost_per_hour = 1.10
    total_hours = total_seconds / 3600
    estimated_cost = total_hours * gpu_cost_per_hour
    
    return {
        "parameters": num_params,
        "parameters_formatted": f"{num_params/1e6:.1f}M" if num_params < 1e9 else f"{num_params/1e9:.2f}B",
        "gpu_memory_gb": round(total_memory_gb, 2),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "estimated_time_hours": round(total_hours, 2),
        "estimated_time_days": round(total_hours / 24, 1),
        "estimated_cost_usd": round(estimated_cost, 2),
        "recommended_batch_size": max(1, int(40 / total_memory_gb) * batch_size),
    }
