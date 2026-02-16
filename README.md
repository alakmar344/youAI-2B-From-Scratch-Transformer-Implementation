# YouAI - Train Your Own 2B Parameter Model From Scratch

Complete guide to training your own 2 billion parameter AI model from scratch!

## ⚠️ REALITY CHECK - READ THIS FIRST

Training a 2B parameter model from scratch is **VERY EXPENSIVE AND TIME-CONSUMING**:

| Requirement | Specification |
|------------|---------------|
| **Cost** | $50,000 - $200,000 in cloud GPU costs |
| **Hardware** | 4-8x NVIDIA A100 (80GB) or H100 GPUs |
| **Time** | 2-6 weeks of continuous training |
| **Data** | 100-500 GB of quality text |
| **Storage** | 50+ GB for checkpoints |
| **Expertise** | Advanced ML knowledge required |

### Budget-Friendly Alternatives:

1. **Fine-tune existing models** (Cost: $10-$100)
   - Start with pre-trained model, customize on your data
   - Takes hours instead of weeks
   - 1000x cheaper

2. **Train smaller models** (Cost: $50-$500)
   - 125M-350M parameters
   - Finishes in days
   - Good for specific domains

3. **Use Google Colab** (Cost: $10-$50/month)
   - Access to free/cheap GPUs
   - Perfect for learning
   - Sufficient for small models

## 📁 Project Structure

```
youai-training/
├── model_architecture.py   # Model definition (2B params)
├── train.py                # Training script
├── inference.py            # Use trained model
├── prepare_data.py         # Data preparation
├── requirements.txt        # Dependencies
├── README.md              # This file
└── data/
    ├── train_data.txt     # Training data
    └── val_data.txt       # Validation data
```

## 🚀 Quick Start (Testing with Sample Data)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create Sample Dataset (For Testing)

```bash
python prepare_data.py
# Choose option 1: Create sample dataset
# Enter 10000 for quick testing
```

This creates a small dataset to test the training pipeline.

### Step 3: Start Training

```bash
python train.py
```

⚠️ **First run will be slow** as PyTorch compiles the model. On CPU, expect 1-2 hours per 100 steps. On GPU, 1-2 minutes per 100 steps.

### Step 4: Use Your Model

```bash
python inference.py --checkpoint ./youai_checkpoints/final --mode chat
```

## 💾 Real Data Preparation

For serious training, you need **100-500 GB** of quality text data:

### Option 1: Download Public Datasets

```python
from datasets import load_dataset

# The Pile (825 GB) - Best for general knowledge
dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)

# OpenWebText (40 GB) - Web content
dataset = load_dataset('openwebtext')

# Wikipedia (20 GB) - Encyclopedia
dataset = load_dataset('wikipedia', '20220301.en')

# C4 (305 GB) - Clean web crawl
dataset = load_dataset('c4', 'en')

# Save to text file
with open('train_data.txt', 'w', encoding='utf-8') as f:
    for example in dataset:
        f.write(example['text'] + '\n')
```

### Option 2: Use Your Own Data

```bash
python prepare_data.py
# Choose option 3: Prepare custom text files
# Enter your .txt file paths
```

## 🖥️ Hardware Requirements

### Minimum (CPU Only):
- **WILL NOT WORK** for full 2B training
- Use for testing with sample data only
- Expect extremely slow training

### Realistic Options:

#### Budget Option ($10-50/month):
- **Google Colab Pro** or **Kaggle**
- 1x T4 or P100 GPU (16GB)
- Train **125M-350M parameter models**
- Good for learning and experimentation

#### Serious Training ($500-2000/month):
- Cloud instance: AWS, GCP, Lambda Labs
- 1-2x A100 (40-80GB) or 4x V100 (32GB)
- Can train 2B model in 3-6 weeks
- Recommended: Lambda Labs (cheapest)

#### Professional Setup ($5000-10000/month):
- 4-8x A100 (80GB) or H100
- Train 2B model in 1-2 weeks
- Necessary for production models

## ⚙️ Training Configuration

Edit `train.py` CONFIG section:

```python
CONFIG = {
    'batch_size': 8,          # Reduce to 2-4 if OOM
    'gradient_accumulation_steps': 4,  # Increases effective batch size
    'learning_rate': 3e-4,    # Standard for transformers
    'num_epochs': 3,          # 3-5 epochs typical
    'max_length': 512,        # Sequence length (reduce if OOM)
}
```

### Memory Optimization:

If you get Out Of Memory (OOM) errors:

1. **Reduce batch_size**: `batch_size: 2` or `1`
2. **Reduce max_length**: `max_length: 256`
3. **Use gradient checkpointing**: Add to model
4. **Use mixed precision**: Enabled by default
5. **Reduce model size**: Edit `model_architecture.py`

### Smaller Model Sizes:

Edit `YouAIConfig` in `model_architecture.py`:

```python
# 350M parameters (trains in 3-7 days on 1x A100)
config = YouAIConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)

# 125M parameters (trains in 1-3 days on 1x T4)
config = YouAIConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
```

## 📊 Monitoring Training

### Option 1: Terminal Output
Training shows loss, perplexity, and learning rate in real-time.

### Option 2: Weights & Biases (Recommended)

```python
# In train.py, set:
'use_wandb': True

# Then login:
wandb login
```

Visit wandb.ai to see beautiful training curves!

## 💰 Cost Estimates

### Cloud GPU Costs:

| Provider | GPU | Cost/Hour | Days for 2B | Total Cost |
|----------|-----|-----------|-------------|------------|
| Lambda Labs | 1x A100 (40GB) | $1.10 | 30-45 days | $800-$1200 |
| Lambda Labs | 8x A100 (40GB) | $7.20 | 5-7 days | $850-$1200 |
| Google Cloud | 1x A100 (40GB) | $3.67 | 30-45 days | $2650-$4000 |
| AWS | 1x A100 (40GB) | $4.10 | 30-45 days | $3000-$4500 |

**Cheapest option**: Lambda Labs with spot instances

### Alternative Approaches:

| Approach | Cost | Time | Result Quality |
|----------|------|------|----------------|
| Sample training | $0 (CPU) | 1 hour | Testing only |
| Small model (125M) | $50-200 | 2-3 days | Good for specific tasks |
| Medium model (350M) | $200-500 | 5-7 days | Very capable |
| Full 2B from scratch | $800-$4000 | 2-6 weeks | State-of-art |
| **Fine-tune existing 2B** | $10-100 | 4-12 hours | **Best value!** |

## 🎯 Training Stages

### Stage 1: Setup (1-2 hours)
- Install dependencies
- Prepare data
- Configure training

### Stage 2: Initial Training (Days 1-7)
- Model learns basic patterns
- Loss drops rapidly
- Perplexity: 1000 → 100

### Stage 3: Deep Learning (Days 7-21)
- Model learns complex patterns
- Loss decreases slowly
- Perplexity: 100 → 30

### Stage 4: Refinement (Days 21-30)
- Fine-tuning and polish
- Minimal loss improvement
- Perplexity: 30 → 20

## 🔍 Using Your Trained Model

### Chat Mode (Interactive):
```bash
python inference.py --checkpoint ./youai_checkpoints/final --mode chat
```

### Generate Mode (Single prompt):
```bash
python inference.py \
  --checkpoint ./youai_checkpoints/final \
  --mode generate \
  --prompt "The future of artificial intelligence" \
  --max_length 200
```

### Python API:
```python
from inference import YouAIInference

youai = YouAIInference('./youai_checkpoints/final')
response = youai.generate("Hello, YouAI!", max_length=50)
print(response[0])
```

## 🐛 Troubleshooting

### Out of Memory (OOM):
```python
# Reduce batch size
'batch_size': 2  # or even 1

# Reduce sequence length
'max_length': 256

# Reduce model size (in model_architecture.py)
hidden_size=1024  # instead of 2048
num_hidden_layers=12  # instead of 24
```

### Training is Too Slow:
- **CPU**: Not viable for 2B. Use Google Colab free GPU
- **Old GPU**: Train smaller model (125M-350M params)
- **Limited budget**: Use gradient accumulation to increase effective batch size

### Loss Not Decreasing:
- Check data quality (no corruption)
- Lower learning rate: `3e-5` instead of `3e-4`
- Increase warmup steps: `2000` instead of `1000`
- Check for NaN losses (might need gradient clipping)

### Model Generates Gibberish:
- Train longer (might be undertrained)
- Use better/more diverse data
- Adjust generation parameters (temperature, top_p)

## 📚 Learning Path

### Beginner:
1. Start with sample dataset (10K examples)
2. Train on Google Colab free GPU
3. Train 125M model for 1-2 days
4. Learn the pipeline

### Intermediate:
1. Collect 10-50 GB of domain-specific data
2. Use Colab Pro or Kaggle
3. Train 350M model for 3-5 days
4. Fine-tune on your use case

### Advanced:
1. Collect 100+ GB diverse data
2. Rent cloud GPUs (Lambda Labs)
3. Train full 2B model for 2-4 weeks
4. Optimize for production

## 🎓 Key Concepts

### Hyperparameters:
- **Learning rate**: How fast model learns (3e-4 is standard)
- **Batch size**: Examples per update (larger = more stable)
- **Temperature**: Generation randomness (0.8 = balanced)
- **Top-k/Top-p**: Sampling strategies for generation

### Metrics:
- **Loss**: Lower is better (measures prediction error)
- **Perplexity**: Lower is better (exp(loss), more interpretable)
- **Tokens/second**: Training speed indicator

## 📖 Further Resources

- [Hugging Face Course](https://huggingface.co/course)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Lambda Labs GPU Pricing](https://lambdalabs.com/service/gpu-cloud)
- [The Pile Dataset](https://pile.eleuther.ai/)
- [Weights & Biases Docs](https://docs.wandb.ai/)

## ⚡ Quick Commands Reference

```bash
# Prepare data
python prepare_data.py

# Train model
python train.py

# Test inference
python inference.py --checkpoint ./youai_checkpoints/final --mode chat

# Generate single response
python inference.py --checkpoint ./youai_checkpoints/final \
  --mode generate --prompt "Your prompt here"

# Monitor GPU
nvidia-smi -l 1
```

## 🎉 What You Get

After training, you'll have:
- ✅ Your own 2B parameter language model
- ✅ Complete ownership (no API costs)
- ✅ Customized to your data
- ✅ Full control over generation
- ✅ Can run locally or serve as API
- ✅ Can continue training anytime

## ⚖️ License

This training code is MIT licensed. However:
- Your trained model belongs to you
- Check data source licenses
- Commercial use depends on training data licenses

## 🤝 Contributing

Found a bug? Have improvements? PRs welcome!

## 📬 Questions?

Training LLMs is complex. Common questions:

**Q: Should I really train from scratch?**
A: Probably not. Fine-tuning existing models is 1000x cheaper and often better. Only train from scratch if you have specialized data or requirements.

**Q: Can I train on my gaming PC?**
A: For testing only. Real training needs datacenter GPUs. Use cloud providers.

**Q: How long until I see good results?**
A: Small models (125M): 2-3 days. Medium (350M): 5-7 days. Full 2B: 2-4 weeks.

**Q: Can I stop and resume training?**
A: Yes! Checkpoints save every 1000 steps. Just resume from latest checkpoint.

---

**Remember**: Training from scratch is an educational experience and engineering challenge. For production use, consider fine-tuning existing models like Llama, Mistral, or Phi!

Good luck! 🚀
