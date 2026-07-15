"""YouAI - Train Your Own Language Model From Scratch

A simple, callable library for training and using custom language models.

Quick Start:
    import youai
    
    # Create a model
    model = youai.create_model("125m")
    
    # Prepare sample data
    train_file, val_file = youai.create_sample_data(num_examples=1000)
    
    # Or download a HuggingFace dataset
    train_file, val_file = youai.download_dataset("tinystories")
    
    # Train the model
    youai.train(model, train_file=train_file, epochs=1)
    
    # Generate text
    result = youai.generate("Hello world", checkpoint_path="./checkpoints/final")
    print(result)
"""

from typing import List

from .config import YouAIConfig, get_preset_config
from .model import YouAIModel
from .trainer import Trainer
from .inference import YouAIInference
from .data import (
    TextDataset,
    create_sample_dataset,
    prepare_custom_text,
    create_dataloaders,
    download_dataset,
    list_datasets,
)

__version__ = "0.1.0"
__all__ = [
    "create_model",
    "train",
    "generate",
    "create_sample_data",
    "prepare_data",
    "download_dataset",
    "list_datasets",
    "load_model",
    "YouAIConfig",
    "YouAIModel",
    "YouAIInference",
    "Trainer",
]


def create_model(preset: str = "125m", **kwargs) -> YouAIModel:
    """Create a YouAI model with a preset configuration.
    
    Args:
        preset: Model size preset ('125m', '350m', '750m', '2b')
        **kwargs: Override config parameters (hidden_size, num_hidden_layers, etc.)
        
    Returns:
        Initialized YouAIModel instance
        
    Examples:
        model = youai.create_model("125m")
        model = youai.create_model("350m", hidden_dropout_prob=0.05)
    """
    config = get_preset_config(preset)
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return YouAIModel(config)


def create_sample_data(
    num_examples: int = 10000,
    output_dir: str = "./data",
) -> tuple:
    """Create sample training data for testing.
    
    Args:
        num_examples: Number of training examples to generate
        output_dir: Directory to save data files
        
    Returns:
        Tuple of (train_file_path, val_file_path)
        
    Examples:
        train_file, val_file = youai.create_sample_data(num_examples=1000)
    """
    return create_sample_dataset(output_dir=output_dir, num_examples=num_examples)


def prepare_data(
    text_files: list,
    output_dir: str = "./data",
    train_split: float = 0.9,
) -> tuple:
    """Prepare training data from text files.
    
    Args:
        text_files: List of paths to text files
        output_dir: Directory to save prepared data
        train_split: Proportion of data for training
        
    Returns:
        Tuple of (train_file_path, val_file_path)
        
    Examples:
        train_file, val_file = youai.prepare_data(["data1.txt", "data2.txt"])
    """
    return prepare_custom_text(
        text_files=text_files,
        output_dir=output_dir,
        train_split=train_split,
    )


def train(
    model: YouAIModel,
    train_file: str,
    val_file: str = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    output_dir: str = "./checkpoints",
    device: str = "cuda",
    **kwargs,
) -> None:
    """Train a YouAI model.
    
    Args:
        model: YouAIModel instance to train
        train_file: Path to training data file
        val_file: Path to validation data file (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        output_dir: Directory for checkpoints
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Additional Trainer parameters
        
    Examples:
        model = youai.create_model("125m")
        youai.train(model, train_file="data/train.txt", epochs=1)
    """
    train_dataloader, val_dataloader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        num_epochs=epochs,
        output_dir=output_dir,
        device=device,
        **kwargs,
    )
    
    trainer.train()


def generate(
    prompt: str,
    checkpoint_path: str = "./checkpoints/final",
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    device: str = "cuda",
) -> List[str]:
    """Generate text from a prompt using a trained model.
    
    Args:
        prompt: Input text prompt
        checkpoint_path: Path to model checkpoint directory
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        num_return_sequences: Number of sequences to generate
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        List of generated text strings
        
    Examples:
        results = youai.generate("The future of AI", checkpoint_path="./checkpoints/final")
        print(results[0])
    """
    inferencer = YouAIInference(checkpoint_path, device=device)
    return inferencer.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
    )


def load_model(checkpoint_path: str, device: str = "cuda") -> YouAIInference:
    """Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        YouAIInference instance ready for generation
        
    Examples:
        model = youai.load_model("./checkpoints/final")
        response = model.generate("Hello world")
        chat_response = model.chat("How are you?")
    """
    return YouAIInference(checkpoint_path, device=device)
