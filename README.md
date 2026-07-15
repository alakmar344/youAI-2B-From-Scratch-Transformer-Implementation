# YouAI - Train Your Own Language Model From Scratch

A simple, callable library for training and using custom language models.

## Installation

```bash
# Install from source
git clone <repo-url>
cd youai
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start (Notebook / Python Script)

```python
import youai

# Step 1: Create a model (choose: '125m', '350m', '750m', '2b')
model = youai.create_model("125m")

# Step 2: Create sample data for testing
train_file, val_file = youai.create_sample_data(num_examples=1000)

# Step 3: Train the model
youai.train(model, train_file=train_file, epochs=1, batch_size=4)

# Step 4: Generate text
results = youai.generate("The future of AI is", checkpoint_path="./checkpoints/final")
print(results[0])
```

## Library API

### Create a Model

```python
import youai

# Use a preset size
model = youai.create_model("125m")   # 125M parameters
model = youai.create_model("350m")   # 350M parameters
model = youai.create_model("750m")   # 750M parameters
model = youai.create_model("2b")     # 2B parameters

# Customize configuration
model = youai.create_model("125m", hidden_dropout_prob=0.05, num_hidden_layers=8)
```

### Prepare Data

```python
# Create sample data for testing
train_file, val_file = youai.create_sample_data(num_examples=5000)

# Prepare from your own text files
train_file, val_file = youai.prepare_data([
    "path/to/data1.txt",
    "path/to/data2.txt",
])
```

### Train

```python
# Train with default settings
youai.train(model, train_file="data/train.txt")

# Train with custom settings
youai.train(
    model,
    train_file="data/train.txt",
    val_file="data/val.txt",
    epochs=3,
    batch_size=8,
    learning_rate=3e-4,
    max_length=512,
    output_dir="./my_checkpoints",
)
```

### Generate Text

```python
# Generate from a checkpoint
results = youai.generate(
    "Once upon a time",
    checkpoint_path="./checkpoints/final",
    max_length=200,
    temperature=0.8,
)

# Load model for multiple generations
model = youai.load_model("./checkpoints/final")
response1 = model.generate("Hello!")
response2 = model.generate("How are you?")

# Chat mode
reply = model.chat("What is machine learning?")
```

### Advanced: Use Classes Directly

```python
from youai import YouAIConfig, YouAIModel, Trainer, YouAIInference

# Custom configuration
config = YouAIConfig(
    hidden_size=1024,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=4096,
)

# Create model
model = YouAIModel(config)

# Create trainer manually
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    num_epochs=5,
    learning_rate=1e-4,
)

# Load checkpoint manually
inference = YouAIInference("./checkpoints/final")
```

## Preset Model Sizes

| Preset | Parameters | Hidden Size | Layers | Heads | Best For |
|--------|-----------|-------------|--------|-------|----------|
| `125m` | 125M | 768 | 12 | 12 | Learning, testing |
| `350m` | 350M | 1024 | 24 | 16 | Real applications |
| `750m` | 750M | 1536 | 24 | 16 | Strong performance |
| `2b` | 2B | 2048 | 24 | 16 | Production quality |

## Command Line Interface

The original scripts still work:

```bash
# Prepare data interactively
python prepare_data.py

# Train model
python train.py

# Run inference
python inference.py --checkpoint ./youai_checkpoints/final --mode chat

# Start web interface
python web_interface.py --checkpoint ./youai_checkpoints/final --port 5000
```

## Hardware Requirements

| Model | Min GPU | Training Time | Cost Estimate |
|-------|---------|---------------|---------------|
| 125M | T4 (16GB) | 2-3 days | $50-100 |
| 350M | A100 (40GB) | 5-7 days | $200-400 |
| 750M | A100 (40GB) | 10-14 days | $400-800 |
| 2B | A100 (80GB) | 2-4 weeks | $1000-4000 |

## Project Structure

```
youai/
├── __init__.py          # Main package interface
├── config.py            # Model configurations
├── model.py             # Model architecture
├── trainer.py           # Training loop
├── inference.py         # Inference utilities
├── data.py              # Data preparation utilities
├── setup.py             # Package installation
├── requirements.txt     # Dependencies
├── train.py             # Legacy training script
├── inference.py         # Legacy inference script
├── prepare_data.py      # Legacy data preparation script
└── web_interface.py     # Web UI
```

## License

MIT License
