"""
Training Script for YouAI 2B Parameter Model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer
import json
import os
from tqdm import tqdm
import wandb
from model_architecture import create_youai_2b
import math


class TextDataset(Dataset):
    """Dataset for training on text data"""
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if line:
                    self.examples.append(line)
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class Trainer:
    """Training loop for YouAI"""
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=3e-4,
        weight_decay=0.01,
        num_epochs=3,
        warmup_steps=1000,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=500,
        output_dir='./checkpoints',
        use_wandb=False,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )
        
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project="youai-training", config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
            })
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Warmup or regular scheduler
                if self.global_step < self.warmup_steps:
                    lr = (self.global_step / self.warmup_steps) * self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                total_loss += loss.item() * self.gradient_accumulation_steps
                avg_loss = total_loss / (step + 1)
                perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item() * self.gradient_accumulation_steps,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'perplexity': perplexity,
                        'epoch': epoch,
                        'step': self.global_step
                    })
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')
                
                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    self.evaluate()
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self):
        """Evaluate on validation set"""
        if not self.val_dataloader:
            return
        
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
            wandb.log({
                'val_loss': avg_loss,
                'val_perplexity': perplexity,
                'step': self.global_step
            })
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, os.path.join(checkpoint_path, 'pytorch_model.bin'))
        
        # Save config
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(self.model.config.__dict__, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Full training loop"""
        print("=" * 60)
        print("Starting training!")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Steps per epoch: {len(self.train_dataloader)}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Total training steps: {len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps}")
        print("=" * 60 + "\n")
        
        for epoch in range(self.num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}\n")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch-{epoch+1}')
        
        print("\nTraining completed!")
        self.save_checkpoint('final')


def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'train_file': 'train_data.txt',  # Your training data
        'val_file': 'val_data.txt',      # Your validation data (optional)
        'batch_size': 8,                  # Reduce if OOM
        'gradient_accumulation_steps': 4, # Effective batch size = 8 * 4 = 32
        'learning_rate': 3e-4,
        'num_epochs': 3,
        'max_length': 512,
        'warmup_steps': 1000,
        'save_steps': 1000,
        'eval_steps': 500,
        'output_dir': './youai_checkpoints',
        'use_wandb': False,  # Set to True if you want to use Weights & Biases logging
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("\n⚠️  WARNING: Training on CPU will be EXTREMELY slow!")
        print("Consider using Google Colab with GPU or a cloud GPU instance.\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = TextDataset(
        CONFIG['train_file'],
        tokenizer,
        max_length=CONFIG['max_length']
    )
    
    val_dataset = None
    if os.path.exists(CONFIG['val_file']):
        val_dataset = TextDataset(
            CONFIG['val_file'],
            tokenizer,
            max_length=CONFIG['max_length']
        )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if device == 'cuda' else False
        )
    
    # Create model
    print("\nInitializing model...")
    model, config = create_youai_2b()
    
    # Calculate memory requirements
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    print(f"\nModel memory: ~{param_memory:.2f} GB")
    print(f"Estimated training memory (with optimizer): ~{param_memory * 4:.2f} GB")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        save_steps=CONFIG['save_steps'],
        eval_steps=CONFIG['eval_steps'],
        output_dir=CONFIG['output_dir'],
        use_wandb=CONFIG['use_wandb'],
        device=device
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
