# YouAI - Train Your Own Language Model From Scratch

A powerful yet simple library for training, using, and deploying custom language models.

## Why YouAI?

| Feature | YouAI | Others |
|---------|-------|--------|
| Lines of code | 5-10 | 100+ |
| Learning curve | Low | High |
| From-scratch training | Built-in | Manual setup |
| Model presets | Yes | No |
| Mixed precision | Built-in | Manual |
| Streaming | Built-in | Manual |
| CLI interface | Yes | No |
| Export tools | Built-in | External |

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start (5 lines)

```python
import youai

model = youai.create_model("125m")
train_file, val_file = youai.download_dataset("tinystories")
youai.train(model, train_file=train_file, epochs=1)
result = youai.generate("Hello world", checkpoint_path="./checkpoints/final")
```

## Features

### Model Presets

```python
model = youai.create_model("125m")   # 125M params - testing
model = youai.create_model("350m")   # 350M params - real apps
model = youai.create_model("750m")   # 750M params - strong
model = youai.create_model("2b")     # 2B params - production
```

### HuggingFace Datasets

```python
train_file, val_file = youai.download_dataset("tinystories")    # ~2GB, fast
train_file, val_file = youai.download_dataset("wikipedia")      # ~500MB
train_file, val_file = youai.download_dataset("openwebtext")    # ~40GB
```

### Advanced Training

```python
# Mixed precision (2x faster)
youai.train(model, train_file, mixed_precision="fp16")

# Gradient checkpointing (50% less memory)
youai.train(model, train_file, gradient_checkpointing=True)

# Resume from checkpoint
youai.train(model, train_file, resume_from="./checkpoints/epoch-2")
```

### Streaming Generation

```python
for token in youai.stream_generate("./checkpoints/final", prompt="Hello"):
    print(token, end="", flush=True)
```

### Chat Sessions

```python
from youai.streaming import ChatSession

session = ChatSession(model, tokenizer)
for token in session.chat_stream("Tell me a story"):
    print(token, end="", flush=True)
```

### Model Export

```python
youai.export_onnx(model, "model.onnx")           # ONNX format
youai.export_quantized(model, "quantized")        # INT8 quantization
```

### Training Estimation

```python
estimates = youai.estimate_training(model, dataset_size=100000)
print(f"Time: {estimates['estimated_time_hours']} hours")
print(f"Cost: ${estimates['estimated_cost_usd']}")
```

### CLI Interface

```bash
# Train
youai train --preset 125m --dataset tinystories --epochs 3

# Generate
youai generate --checkpoint ./checkpoints/final --prompt "Hello"

# Chat
youai chat --checkpoint ./checkpoints/final

# Export
youai export --checkpoint ./checkpoints/final --format onnx

# Benchmark
youai benchmark --checkpoint ./checkpoints/final
```

## What Makes YouAI Different?

### 1. Simplicity

```python
# YouAI (5 lines)
import youai
model = youai.create_model("125m")
train_file, _ = youai.download_dataset("tinystories")
youai.train(model, train_file=train_file)
result = youai.generate("Hello", checkpoint_path="./checkpoints/final")

# HuggingFace (50+ lines)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
# ... many more lines of setup
```

### 2. Full Ownership

- No API costs
- No rate limits
- No data leaves your machine
- Complete control

### 3. Production Ready

- Mixed precision training
- Gradient checkpointing
- ONNX export
- Model quantization
- Streaming support

### 4. Educational

- Clean source code
- Learn how LLMs work
- Understand training
- Experiment freely

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete documentation.

## Project Structure

```
youai/
├── __init__.py              # Main API
├── config.py                # Model configurations
├── model.py                 # Model architecture
├── trainer.py               # Basic trainer
├── training_advanced.py     # Advanced training features
├── inference.py             # Inference utilities
├── streaming.py             # Streaming generation
├── export.py                # Model export tools
├── data.py                  # Data preparation
├── cli.py                   # Command line interface
├── setup.py                 # Package setup
├── DOCUMENTATION.md         # Full documentation
└── README.md                # This file
```

## Use Cases

### Learning & Education
```python
model = youai.create_model("125m")
# Train on small dataset to understand how LLMs work
```

### Domain-Specific Models
```python
# Train on your own data
train_file, _ = youai.prepare_data(["medical_data.txt"])
model = youai.create_model("350m")
youai.train(model, train_file=train_file)
```

### Prototyping
```python
# Quick iteration on model ideas
model = youai.create_model("125m", hidden_size=512, num_hidden_layers=6)
```

### Production Deployment
```python
# Export for production
youai.export_onnx(model, "model.onnx")
youai.export_quantized(model, "model_int8")
```

## Hardware Requirements

| Model | Min GPU | Training Time | Cost |
|-------|---------|---------------|------|
| 125M | T4 | 2-3 hours | Free (Colab) |
| 350M | A100 | 4-6 hours | ~$5-7 |
| 750M | A100 | 1-2 days | ~$30-50 |
| 2B | A100 | 2-4 weeks | ~$800-1200 |

## License

MIT License

## Links

- [Documentation](DOCUMENTATION.md)
- [Examples](DOCUMENTATION.md#examples)
- [CLI Reference](DOCUMENTATION.md#cli-reference)
