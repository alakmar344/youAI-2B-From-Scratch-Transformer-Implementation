"""YouAI Data Preparation Utilities"""

import os
import random
from typing import List, Optional, Tuple, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm


# Available HuggingFace datasets
DATASET_PRESETS = {
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "description": "Simple short stories - great for quick training and testing",
        "size": "~2GB",
        "split": "train",
    },
    "openwebtext": {
        "name": "openwebtext",
        "description": "Web text from Reddit links - good for general knowledge",
        "size": "~40GB",
        "split": "train",
    },
    "wikipedia": {
        "name": "wikitext",
        "description": "Wikipedia articles (wikitext-103) - encyclopedia knowledge",
        "size": "~500MB",
        "subconfig": "wikitext-103-v1",
        "split": "train",
    },
}

DatasetPreset = Literal["tinystories", "openwebtext", "wikipedia"]


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


def list_datasets() -> None:
    """Print available HuggingFace dataset presets."""
    print("\nAvailable HuggingFace Datasets:")
    print("=" * 50)
    for key, info in DATASET_PRESETS.items():
        print(f"\n  {key}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
    print("\n" + "=" * 50)
    print("Usage: train_file, val_file = youai.download_dataset('tinystories')")
    print()


def download_dataset(
    dataset: DatasetPreset,
    output_dir: str = "./data",
    num_examples: Optional[int] = None,
    train_split: float = 0.9,
) -> Tuple[str, str]:
    """Download and prepare a HuggingFace dataset for training.
    
    Args:
        dataset: Dataset preset name ('tinystories', 'openwebtext', 'wikipedia')
        output_dir: Directory to save data files
        num_examples: Limit number of examples (None = use all)
        train_split: Proportion of data for training
        
    Returns:
        Tuple of (train_file_path, val_file_path)
        
    Examples:
        # Download TinyStories (fast, ~2GB) - good for testing
        train_file, val_file = youai.download_dataset("tinystories")
        
        # Download with limited examples
        train_file, val_file = youai.download_dataset("tinystories", num_examples=50000)
        
        # Download Wikipedia
        train_file, val_file = youai.download_dataset("wikipedia")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )
    
    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_PRESETS.keys())}")
    
    preset = DATASET_PRESETS[dataset]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset} dataset...")
    print(f"Description: {preset['description']}")
    print(f"Size: {preset['size']}")
    
    # Load dataset
    if "subconfig" in preset:
        hf_dataset = load_dataset(preset["name"], preset["subconfig"], split=preset["split"])
    else:
        hf_dataset = load_dataset(preset["name"], split=preset["split"])
    
    # Limit examples if specified
    if num_examples and num_examples < len(hf_dataset):
        hf_dataset = hf_dataset.select(range(num_examples))
        print(f"Using {num_examples} examples")
    
    # Extract text
    print("Processing text...")
    texts = []
    for example in tqdm(hf_dataset, desc="Extracting text"):
        text = example.get("text", "").strip()
        if text and len(text) > 10:  # Skip very short texts
            texts.append(text)
    
    # Shuffle and split
    random.shuffle(texts)
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Save to files
    train_file = os.path.join(output_dir, f"{dataset}_train.txt")
    val_file = os.path.join(output_dir, f"{dataset}_val.txt")
    
    print(f"Saving {len(train_texts)} training examples...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\n')
    
    print(f"Saving {len(val_texts)} validation examples...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\n')
    
    print(f"\nDataset ready!")
    print(f"  Train: {train_file} ({len(train_texts)} examples)")
    print(f"  Val:   {val_file} ({len(val_texts)} examples)")
    
    return train_file, val_file
