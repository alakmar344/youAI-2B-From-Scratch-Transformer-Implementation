# YouAI Documentation

## Table of Contents

- [What is YouAI?](#what-is-youai)
- [Why YouAI?](#why-youai)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features Overview](#features-overview)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [FAQ](#faq)

---

## What is YouAI?

YouAI is a Python library for training and using custom language models from scratch. It provides a simple, high-level API that makes it easy to:

- **Train your own AI model** from scratch on any text data
- **Generate text** using trained models
- **Chat interactively** with your AI
- **Export models** for production deployment
- **Stream responses** in real-time

### Key Differentiators

| Feature | YouAI | HuggingFace | GPT-NeoX |
|---------|-------|-------------|----------|
| Lines of code | 5-10 | 50-100 | 200+ |
| Learning curve | Low | Medium | High |
| From-scratch training | Built-in | Manual | Complex |
| Model presets | Yes | No | No |
| Streaming | Built-in | Manual | Manual |
| Export tools | Built-in | External | External |
| CLI interface | Yes | No | No |

---

## Why YouAI?

### 1. **Simplicity First**

Train a model in 5 lines of code:

```python
import youai

model = youai.create_model("125m")
train_file, val_file = youai.download_dataset("tinystories")
youai.train(model, train_file=train_file, epochs=3)
result = youai.generate("Once upon a time", checkpoint_path="./checkpoints/final")
```

### 2. **Full Ownership**

- No API costs or rate limits
- No data leaves your machine
- Complete control over model behavior
- Deploy anywhere

### 3. **Production Ready**

- Mixed precision training (2x faster)
- Gradient checkpointing (50% less memory)
- ONNX export for cross-platform deployment
- Model quantization for smaller size
- Streaming for real-time applications

### 4. **Educational**

- Clean, readable source code
- Learn how LLMs actually work
- Understand training dynamics
- Experiment with architectures

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/youai/youai.git
cd youai
pip install -e .
```

### Dependencies

```bash
pip install torch transformers tqdm numpy tokenizers
```

### Optional Dependencies

```bash
# For HuggingFace datasets
pip install datasets

# For Weights & Biases logging
pip install wandb

# For ONNX export
pip install onnx onnxruntime

# For web interface
pip install flask flask-cors
```

---

## Quick Start

### 1. Train a Model

```python
import youai

# Create a small model for testing
model = youai.create_model("125m")

# Create sample data
train_file, val_file = youai.create_sample_data(num_examples=5000)

# Train
youai.train(
    model,
    train_file=train_file,
    epochs=3,
    batch_size=8,
    output_dir="./my_model"
)
```

### 2. Use a HuggingFace Dataset

```python
import youai

# Download TinyStories (fast, good for testing)
train_file, val_file = youai.download_dataset("tinystories")

# Or limit to 50k examples
train_file, val_file = youai.download_dataset("tinystories", num_examples=50000)

# Train
model = youai.create_model("125m")
youai.train(model, train_file=train_file, epochs=1)
```

### 3. Generate Text

```python
import youai

# Simple generation
results = youai.generate(
    "The future of AI is",
    checkpoint_path="./checkpoints/final",
    max_length=200,
    temperature=0.8
)
print(results[0])
```

### 4. Interactive Chat

```python
import youai

model = youai.load_model("./checkpoints/final")

# Chat
response = model.chat("Hello! What can you do?")
print(response)

# Chat with history
history = [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help?"}
]
response = model.chat("Tell me a joke", history=history)
```

---

## Features Overview

### Model Presets

| Preset | Parameters | Hidden Size | Layers | Best For |
|--------|-----------|-------------|--------|----------|
| `125m` | 125M | 768 | 12 | Learning, testing |
| `350m` | 350M | 1024 | 24 | Real applications |
| `750m` | 750M | 1536 | 24 | Strong performance |
| `2b` | 2B | 2048 | 24 | Production quality |

```python
model = youai.create_model("125m")  # Fast, for testing
model = youai.create_model("2b")    # Production quality
```

### Mixed Precision Training

Up to 2x faster training on modern GPUs:

```python
youai.train(
    model,
    train_file=train_file,
    mixed_precision="fp16",  # or "bf16" for newer GPUs
)
```

### Gradient Checkpointing

Trade compute for memory - train larger models:

```python
youai.train(
    model,
    train_file=train_file,
    gradient_checkpointing=True,
)
```

### Streaming Generation

Real-time token-by-token output:

```python
import youai

# Stream from checkpoint
for token in youai.stream_generate("./checkpoints/final", prompt="Hello"):
    print(token, end="", flush=True)

# Stream with model object
from youai.streaming import StreamingGenerator

generator = StreamingGenerator(model, tokenizer)
for token in generator.stream("Tell me a story"):
    print(token, end="", flush=True)
```

### Chat Sessions

Maintain conversation context:

```python
from youai.streaming import ChatSession

session = ChatSession(model, tokenizer, system_prompt="You are a helpful assistant.")

# Multi-turn conversation
response1 = session.chat("Hi!")
response2 = session.chat("What's my name?")  # Knows context

# Streaming chat
for token in session.chat_stream("Tell me more"):
    print(token, end="", flush=True)
```

### Model Export

#### ONNX Format

```python
import youai

# Export to ONNX
youai.export_onnx(model, "model.onnx")

# Use with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

#### Quantized Models

Smaller models, faster inference:

```python
# INT8 quantization (2-4x smaller)
youai.export_quantized(model, "quantized_model", dtype="qint8")

# FP16 quantization (2x smaller)
youai.export_quantized(model, "fp16_model", dtype="float16")
```

### Training Estimation

Before training, estimate time and cost:

```python
import youai

model = youai.create_model("125m")
estimates = youai.estimate_training(model, dataset_size=100000)

print(f"Parameters: {estimates['parameters_formatted']}")
print(f"GPU Memory: {estimates['gpu_memory_gb']} GB")
print(f"Time: {estimates['estimated_time_hours']} hours")
print(f"Cost: ${estimates['estimated_cost_usd']}")
```

### Learning Rate Finder

Find optimal learning rate automatically:

```python
import youai

model = youai.create_model("125m")
train_loader, _ = youai.create_dataloaders(train_file, batch_size=8)

# Find best learning rate
lr = youai.find_learning_rate(model, train_loader)
print(f"Suggested LR: {lr}")

# Use it for training
youai.train(model, train_file=train_file, learning_rate=lr)
```

### Resume Training

Continue from a checkpoint:

```python
youai.train(
    model,
    train_file=train_file,
    resume_from="./checkpoints/epoch-2",
    epochs=5,  # Will continue to epoch 5
)
```

### Early Stopping

Stop training when validation loss stops improving:

```python
from youai.training_advanced import TrainingConfig

config = TrainingConfig(
    early_stopping_patience=5,  # Stop after 5 evals without improvement
)

trainer = youai.train_advanced(
    model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config,
)
trainer.train()
```

### Model Benchmarking

Measure inference performance:

```python
import youai

results = youai.benchmark_model(model, device="cpu")
print(f"Inference: {results['avg_inference_ms']} ms")
print(f"Speed: {results['tokens_per_second']} tokens/sec")
```

---

## API Reference

### Core Functions

#### `youai.create_model(preset, **kwargs)`

Create a model with preset configuration.

**Parameters:**
- `preset` (str): Model size ('125m', '350m', '750m', '2b')
- `**kwargs`: Override config parameters

**Returns:** `YouAIModel`

#### `youai.train(model, train_file, **kwargs)`

Train a model.

**Parameters:**
- `model`: YouAIModel instance
- `train_file` (str): Path to training data
- `val_file` (str): Path to validation data (optional)
- `epochs` (int): Number of epochs (default: 3)
- `batch_size` (int): Batch size (default: 8)
- `learning_rate` (float): Learning rate (default: 3e-4)
- `mixed_precision` (str): 'fp16', 'bf16', or None
- `gradient_checkpointing` (bool): Enable gradient checkpointing
- `resume_from` (str): Checkpoint path to resume from

#### `youai.generate(prompt, checkpoint_path, **kwargs)`

Generate text from a prompt.

**Parameters:**
- `prompt` (str): Input text
- `checkpoint_path` (str): Path to model checkpoint
- `max_length` (int): Maximum tokens (default: 100)
- `temperature` (float): Sampling temperature (default: 0.8)
- `top_k` (int): Top-k sampling (default: 50)
- `top_p` (float): Nucleus sampling (default: 0.9)

**Returns:** List[str]

#### `youai.download_dataset(dataset, **kwargs)`

Download a HuggingFace dataset.

**Parameters:**
- `dataset` (str): Dataset name ('tinystories', 'openwebtext', 'wikipedia')
- `num_examples` (int): Limit number of examples
- `output_dir` (str): Output directory

**Returns:** Tuple[str, str] (train_file, val_file)

#### `youai.load_model(checkpoint_path, device)`

Load a trained model.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint
- `device` (str): Device ('cuda', 'cpu', 'auto')

**Returns:** `YouAIInference`

### Classes

#### `YouAIConfig`

Model configuration.

```python
config = youai.YouAIConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
```

#### `YouAIModel`

The model itself.

```python
model = youai.YouAIModel(config)
print(model.num_parameters)  # Parameter count
```

#### `YouAIInference`

Inference wrapper.

```python
inference = youai.YouAIInference("./checkpoints/final")
response = inference.generate("Hello")
chat_response = inference.chat("How are you?")
```

#### `Trainer`

Basic trainer.

```python
trainer = youai.Trainer(
    model=model,
    train_dataloader=train_loader,
    num_epochs=3,
)
trainer.train()
```

#### `MixedPrecisionTrainer`

Advanced trainer with mixed precision.

```python
from youai.training_advanced import MixedPrecisionTrainer, TrainingConfig

config = TrainingConfig(mixed_precision="fp16", gradient_checkpointing=True)
trainer = MixedPrecisionTrainer(model, train_loader, config=config)
trainer.train()
```

#### `StreamingGenerator`

Streaming text generation.

```python
from youai.streaming import StreamingGenerator

generator = StreamingGenerator(model, tokenizer)
for token in generator.stream("Hello"):
    print(token, end="", flush=True)
```

#### `ChatSession`

Multi-turn chat.

```python
from youai.streaming import ChatSession

session = ChatSession(model, tokenizer)
response = session.chat("Hello!")
```

---

## Advanced Usage

### Custom Model Architecture

```python
from youai import YouAIConfig, YouAIModel

# Custom configuration
config = YouAIConfig(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    vocab_size=32000,
)

model = YouAIModel(config)
print(f"Parameters: {model.num_parameters:,}")
```

### Advanced Training Configuration

```python
from youai.training_advanced import TrainingConfig, MixedPrecisionTrainer

config = TrainingConfig(
    mixed_precision="fp16",
    gradient_checkpointing=True,
    early_stopping_patience=5,
    max_steps=10000,
    lr_scheduler="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
)

trainer = MixedPrecisionTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config,
    learning_rate=3e-4,
    num_epochs=3,
    output_dir="./checkpoints",
)

trainer.train()
```

### Custom Data Preparation

```python
import youai

# From text files
train_file, val_file = youai.prepare_data(
    text_files=["data1.txt", "data2.txt", "data3.txt"],
    train_split=0.9,
)

# From HuggingFace dataset
train_file, val_file = youai.download_dataset(
    "tinystories",
    num_examples=100000,
    output_dir="./my_data",
)
```

### Batch Generation

```python
import youai

model = youai.load_model("./checkpoints/final")

prompts = ["Hello", "How are you?", "Tell me a joke"]
for prompt in prompts:
    response = model.generate(prompt, max_length=50)
    print(f"{prompt} -> {response[0]}")
```

### Web Interface

```python
# Run the web interface
# python web_interface.py --checkpoint ./checkpoints/final --port 5000
```

---

## CLI Reference

YouAI provides a command-line interface for common operations.

### Training

```bash
# Basic training
youai train --preset 125m --data train.txt --epochs 3

# With HuggingFace dataset
youai train --preset 125m --dataset tinystories --epochs 1

# With mixed precision
youai train --preset 125m --data train.txt --mixed-precision fp16

# Resume training
youai train --preset 125m --data train.txt --resume ./checkpoints/epoch-2

# Show estimates first
youai train --preset 125m --data train.txt --estimate
```

### Generation

```bash
# Generate text
youai generate --checkpoint ./checkpoints/final --prompt "Hello world"

# With parameters
youai generate \
    --checkpoint ./checkpoints/final \
    --prompt "The future of AI" \
    --max-length 200 \
    --temperature 0.9 \
    --num-sequences 3
```

### Chat

```bash
# Interactive chat
youai chat --checkpoint ./checkpoints/final
```

### Export

```bash
# Export to ONNX
youai export --checkpoint ./checkpoints/final --format onnx

# Export quantized model
youai export --checkpoint ./checkpoints/final --format quantized
```

### Benchmark

```bash
# Benchmark on CPU
youai benchmark --checkpoint ./checkpoints/final --device cpu

# Benchmark on GPU
youai benchmark --checkpoint ./checkpoints/final --device cuda --runs 200
```

### Info

```bash
# Show model info
youai info --preset 125m

# List available datasets
youai datasets
```

---

## Examples

### Example 1: Quick Training Script

```python
#!/usr/bin/env python3
"""Quick training example."""
import youai

# Setup
model = youai.create_model("125m")
train_file, val_file = youai.download_dataset("tinystories", num_examples=10000)

# Train
youai.train(
    model,
    train_file=train_file,
    val_file=val_file,
    epochs=1,
    batch_size=8,
    output_dir="./quick_model",
)

# Test
result = youai.generate("Once upon a time", checkpoint_path="./quick_model/final")
print(result[0])
```

### Example 2: Advanced Training with Monitoring

```python
#!/usr/bin/env python3
"""Advanced training with Weights & Biases."""
import youai
from youai.training_advanced import TrainingConfig, MixedPrecisionTrainer

# Create model
model = youai.create_model("350m")

# Prepare data
train_file, val_file = youai.download_dataset("tinystories")

# Create dataloaders
train_loader, val_loader = youai.create_dataloaders(
    train_file, val_file, batch_size=16
)

# Configure training
config = TrainingConfig(
    mixed_precision="fp16",
    gradient_checkpointing=True,
    early_stopping_patience=3,
)

# Train
trainer = MixedPrecisionTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config,
    learning_rate=3e-4,
    num_epochs=5,
    output_dir="./advanced_model",
)

trainer.train()
```

### Example 3: Streaming Chat Application

```python
#!/usr/bin/env python3
"""Streaming chat application."""
from youai.streaming import ChatSession
from youai import load_model

# Load model
inference = load_model("./checkpoints/final")
model = inference.model
tokenizer = inference.tokenizer

# Create chat session
session = ChatSession(
    model=model,
    tokenizer=tokenizer,
    system_prompt="You are a helpful and friendly AI assistant.",
)

print("Chat started! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    print("AI: ", end="", flush=True)
    for token in session.chat_stream(user_input):
        print(token, end="", flush=True)
    print("\n")
```

### Example 4: Model Export Pipeline

```python
#!/usr/bin/env python3
"""Export model for production."""
import youai

# Load trained model
inference = youai.load_model("./checkpoints/final")
model = inference.model

# Export ONNX
youai.export_onnx(model, "production/model.onnx")

# Export quantized
youai.export_quantized(model, "production/model_quantized", dtype="qint8")

# Benchmark
results = youai.benchmark_model(model, device="cpu")
print(f"Inference speed: {results['tokens_per_second']} tokens/sec")
```

---

## FAQ

### Q: How much does it cost to train?

**A:** Depends on model size and hardware:

| Model | GPU | Time | Cost |
|-------|-----|------|------|
| 125M | T4 (Colab) | 2-3 hours | Free |
| 125M | A100 | 30 min | ~$0.50 |
| 350M | A100 | 4-6 hours | ~$5-7 |
| 2B | A100 | 2-4 weeks | ~$800-1200 |

### Q: Can I train on my laptop?

**A:** Yes, for small models (125M). CPU training is slow but works for testing. For real training, use a GPU (Colab, Kaggle, or cloud).

### Q: What data format do I need?

**A:** Simple text file, one example per line:

```
This is the first training example.
This is the second example.
Each line is a separate document.
```

### Q: How do I resume training?

**A:** Pass the checkpoint path:

```python
youai.train(model, train_file=train_file, resume_from="./checkpoints/epoch-2")
```

### Q: Can I fine-tune existing models?

**A:** YouAI is designed for training from scratch. For fine-tuning, use HuggingFace Transformers directly.

### Q: How do I use my model in production?

**A:** Options:
1. Export to ONNX and use ONNX Runtime
2. Use the quantized model for smaller size
3. Use the web interface for simple deployment
4. Use the Python API directly

### Q: What's the best model size to start with?

**A:** Start with `125m` for testing. Move to `350m` for real applications. Use `2b` only if you have GPU resources.

### Q: How do I improve model quality?

**A:** 
1. Use more data (10GB+)
2. Train for more epochs
3. Use larger model
4. Use better data quality
5. Adjust learning rate

---

## Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Documentation:** This file

## License

MIT License
