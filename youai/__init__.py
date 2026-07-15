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

Advanced Features:
    # Mixed precision training (2x faster on modern GPUs)
    youai.train(model, train_file, mixed_precision="fp16")
    
    # Streaming generation
    for token in youai.stream_generate(model, tokenizer, "Hello"):
        print(token, end="")
    
    # Export model
    youai.export_onnx(model, "model.onnx")
    
    # CLI usage
    # youai train --preset 125m --dataset tinystories --epochs 3
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

__version__ = "0.2.0"
__all__ = [
    # Core functions
    "create_model",
    "train",
    "generate",
    "create_sample_data",
    "prepare_data",
    "download_dataset",
    "list_datasets",
    "load_model",
    
    # Advanced training
    "train_advanced",
    "estimate_training",
    "find_learning_rate",
    
    # Export
    "export_onnx",
    "export_quantized",
    "benchmark_model",
    
    # Streaming
    "stream_generate",
    "ChatSession",
    
    # Classes
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
    mixed_precision: str = None,
    gradient_checkpointing: bool = False,
    resume_from: str = None,
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
        device: Device to use ('cuda', 'cpu', or 'auto')
        mixed_precision: Enable mixed precision ('fp16', 'bf16', or None)
        gradient_checkpointing: Enable gradient checkpointing (saves memory)
        resume_from: Resume training from checkpoint path
        **kwargs: Additional Trainer parameters
        
    Examples:
        model = youai.create_model("125m")
        youai.train(model, train_file="data/train.txt", epochs=1)
        
        # Advanced training with mixed precision
        youai.train(model, train_file, mixed_precision="fp16", gradient_checkpointing=True)
    """
    from .training_advanced import MixedPrecisionTrainer, TrainingConfig
    
    train_dataloader, val_dataloader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    
    config = TrainingConfig(
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    trainer = MixedPrecisionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        learning_rate=learning_rate,
        num_epochs=epochs,
        output_dir=output_dir,
        device=device,
    )
    
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    trainer.train()


def train_advanced(
    model: YouAIModel,
    train_dataloader,
    val_dataloader=None,
    config=None,
    **kwargs,
):
    """Advanced training with full control.
    
    Args:
        model: YouAIModel instance
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader (optional)
        config: TrainingConfig instance
        **kwargs: Additional trainer parameters
        
    Returns:
        MixedPrecisionTrainer instance
    """
    from .training_advanced import MixedPrecisionTrainer
    
    trainer = MixedPrecisionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        **kwargs,
    )
    return trainer


def estimate_training(
    model: YouAIModel,
    dataset_size: int,
    batch_size: int = 8,
    num_epochs: int = 3,
) -> dict:
    """Estimate training time and resource requirements.
    
    Args:
        model: YouAIModel instance
        dataset_size: Number of training examples
        batch_size: Batch size
        num_epochs: Number of epochs
        
    Returns:
        Dictionary with estimates
        
    Examples:
        model = youai.create_model("125m")
        estimates = youai.estimate_training(model, dataset_size=100000)
        print(f"Estimated time: {estimates['estimated_time_hours']} hours")
    """
    from .training_advanced import estimate_training_time
    return estimate_training_time(model, dataset_size, batch_size, num_epochs)


def find_learning_rate(
    model: YouAIModel,
    train_dataloader,
    **kwargs,
) -> float:
    """Find optimal learning rate.
    
    Args:
        model: YouAIModel instance
        train_dataloader: Training DataLoader
        
    Returns:
        Suggested learning rate
        
    Examples:
        lr = youai.find_learning_rate(model, train_loader)
        youai.train(model, train_file, learning_rate=lr)
    """
    from .training_advanced import LearningRateFinder
    finder = LearningRateFinder(model, train_dataloader, **kwargs)
    return finder.find()


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


def stream_generate(
    model_or_checkpoint,
    tokenizer=None,
    prompt: str = "",
    max_length: int = 100,
    temperature: float = 0.8,
    device: str = "auto",
):
    """Stream generated tokens one by one.
    
    Args:
        model_or_checkpoint: YouAIModel or path to checkpoint
        tokenizer: Tokenizer (required if model_or_checkpoint is a model)
        prompt: Input text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        
    Yields:
        Generated tokens one at a time
        
    Examples:
        # From checkpoint
        for token in youai.stream_generate("./checkpoints/final", prompt="Hello"):
            print(token, end="", flush=True)
        
        # From model
        for token in youai.stream_generate(model, tokenizer, "Hello"):
            print(token, end="", flush=True)
    """
    from .streaming import StreamingGenerator
    
    if isinstance(model_or_checkpoint, str):
        inferencer = YouAIInference(model_or_checkpoint, device=device)
        model = inferencer.model
        tokenizer = inferencer.tokenizer
    else:
        model = model_or_checkpoint
    
    generator = StreamingGenerator(model, tokenizer, device=device)
    yield from generator.stream(prompt, max_length=max_length, temperature=temperature)


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


def export_onnx(model: YouAIModel, output_path: str, **kwargs) -> str:
    """Export model to ONNX format.
    
    Args:
        model: YouAIModel instance
        output_path: Path to save ONNX model
        
    Returns:
        Path to exported model
        
    Examples:
        youai.export_onnx(model, "model.onnx")
    """
    from .export import ModelExporter
    exporter = ModelExporter(model)
    return exporter.export_onnx(output_path, **kwargs)


def export_quantized(model: YouAIModel, output_dir: str, dtype: str = "qint8") -> None:
    """Export quantized model for faster inference.
    
    Args:
        model: YouAIModel instance
        output_dir: Directory to save quantized model
        dtype: Quantization type ('qint8' or 'float16')
        
    Examples:
        youai.export_quantized(model, "quantized_model")
    """
    from .export import ModelQuantizer
    quantizer = ModelQuantizer(model)
    quantized = quantizer.dynamic_quantize(dtype=dtype)
    quantizer.save(quantized, output_dir)


def benchmark_model(model: YouAIModel, device: str = "cpu", **kwargs) -> dict:
    """Benchmark model inference speed.
    
    Args:
        model: YouAIModel instance
        device: Device to benchmark on
        
    Returns:
        Dictionary with benchmark results
        
    Examples:
        results = youai.benchmark_model(model)
        print(f"Tokens/sec: {results['tokens_per_second']}")
    """
    from .export import benchmark_model as _benchmark
    return _benchmark(model, device=device, **kwargs)
