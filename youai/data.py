"""YouAI Data Preparation Utilities"""

import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm


class TextDataset(Dataset):
    """Dataset for language model training.
    
    Args:
        file_path: Path to text file (one example per line)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(self, file_path: str, tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(line)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_sample_dataset(
    output_dir: str = './data',
    num_examples: int = 10000,
) -> Tuple[str, str]:
    """Create a sample dataset for testing.
    
    Args:
        output_dir: Directory to save data files
        num_examples: Number of training examples to generate
        
    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    templates = [
        "The importance of {topic} cannot be overstated in modern {field}.",
        "Recent advances in {topic} have revolutionized the way we think about {concept}.",
        "Scientists have discovered that {topic} plays a crucial role in {process}.",
        "The relationship between {topic} and {concept} has been studied extensively.",
        "Understanding {topic} is essential for anyone working in {field}.",
        "Experts agree that {topic} will continue to shape the future of {field}.",
        "{topic} has emerged as a key factor in understanding {concept}.",
        "The impact of {topic} on {field} cannot be ignored by researchers.",
        "New research suggests that {topic} may be more important than previously thought.",
        "Many scholars have devoted their careers to studying {topic} and its effects.",
    ]
    
    topics = [
        "artificial intelligence", "climate change", "quantum computing", "biotechnology",
        "renewable energy", "space exploration", "nanotechnology", "genetics", "robotics",
        "neuroscience", "cryptography", "materials science", "ecology", "psychology"
    ]
    
    fields = [
        "science", "technology", "medicine", "education", "industry", "research",
        "engineering", "business", "society", "environment"
    ]
    
    concepts = [
        "innovation", "sustainability", "efficiency", "development", "progress",
        "understanding", "discovery", "implementation", "evolution", "transformation"
    ]
    
    processes = [
        "learning", "adaptation", "growth", "change", "communication",
        "interaction", "development", "evolution", "transformation", "optimization"
    ]
    
    train_file = os.path.join(output_dir, 'train_data.txt')
    val_file = os.path.join(output_dir, 'val_data.txt')
    
    train_examples = []
    for _ in tqdm(range(num_examples), desc="Generating training data"):
        template = random.choice(templates)
        sentence = template.format(
            topic=random.choice(topics),
            field=random.choice(fields),
            concept=random.choice(concepts),
            process=random.choice(processes)
        )
        train_examples.append(sentence)
    
    val_examples = []
    for _ in tqdm(range(num_examples // 10), desc="Generating validation data"):
        template = random.choice(templates)
        sentence = template.format(
            topic=random.choice(topics),
            field=random.choice(fields),
            concept=random.choice(concepts),
            process=random.choice(processes)
        )
        val_examples.append(sentence)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(example + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(example + '\n')
    
    return train_file, val_file


def prepare_custom_text(
    text_files: List[str],
    output_dir: str = './data',
    train_split: float = 0.9,
) -> Tuple[str, str]:
    """Prepare training data from custom text files.
    
    Args:
        text_files: List of paths to text files
        output_dir: Directory to save prepared data
        train_split: Proportion of data for training
        
    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_lines = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_lines.extend([line.strip() for line in lines if line.strip()])
    
    random.shuffle(all_lines)
    
    split_idx = int(len(all_lines) * train_split)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]
    
    train_file = os.path.join(output_dir, 'train_data.txt')
    val_file = os.path.join(output_dir, 'val_data.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    return train_file, val_file


def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    tokenizer: Optional[GPT2Tokenizer] = None,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 2,
    device: str = 'cuda',
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for training.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file (optional)
        tokenizer: GPT2Tokenizer instance (creates default if None)
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        device: Device type for pin_memory
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = TextDataset(train_file, tokenizer, max_length=max_length)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == 'cuda'
    )
    
    val_dataloader = None
    if val_file and os.path.exists(val_file):
        val_dataset = TextDataset(val_file, tokenizer, max_length=max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == 'cuda'
        )
    
    return train_dataloader, val_dataloader
