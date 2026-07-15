"""YouAI Training Module"""

import os
import json
import math
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from .model import YouAIModel
from .config import YouAIConfig


class Trainer:
    """Training loop for YouAI models.
    
    Args:
        model: YouAIModel instance
        train_dataloader: Training data DataLoader
        val_dataloader: Validation data DataLoader (optional)
        learning_rate: Learning rate (default: 3e-4)
        weight_decay: Weight decay (default: 0.01)
        num_epochs: Number of training epochs (default: 3)
        warmup_steps: Number of warmup steps (default: 1000)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        save_steps: Save checkpoint every N steps (default: 1000)
        eval_steps: Evaluate every N steps (default: 500)
        output_dir: Directory for checkpoints (default: './checkpoints')
        use_wandb: Enable Weights & Biases logging (default: False)
        device: Device to use ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: YouAIModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_steps: int = 1000,
        eval_steps: int = 500,
        output_dir: str = './checkpoints',
        use_wandb: bool = False,
        device: str = 'cuda',
    ):
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )
        
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project="youai-training", config={
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                    "batch_size": train_dataloader.batch_size,
                })
            except ImportError:
                print("Warning: wandb not installed. Disabling logging.")
                self.use_wandb = False
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                if self.global_step < self.warmup_steps:
                    lr = (self.global_step / self.warmup_steps) * self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                avg_loss = total_loss / (step + 1)
                perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train_loss': loss.item() * self.gradient_accumulation_steps,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'perplexity': perplexity,
                        'epoch': epoch,
                        'step': self.global_step
                    })
                
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')
                
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    self.evaluate()
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self) -> Optional[float]:
        """Evaluate on validation set."""
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        print(f"\nValidation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}\n")
        
        if self.use_wandb:
            import wandb
            wandb.log({
                'val_loss': avg_loss,
                'val_perplexity': perplexity,
                'step': self.global_step
            })
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, os.path.join(checkpoint_path, 'pytorch_model.bin'))
        
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Run full training loop."""
        print("=" * 60)
        print("Starting training!")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Steps per epoch: {len(self.train_dataloader)}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Total training steps: {len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps}")
        print("=" * 60 + "\n")
        
        for epoch in range(self.num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}\n")
            self.save_checkpoint(f'epoch-{epoch+1}')
        
        print("\nTraining completed!")
        self.save_checkpoint('final')
