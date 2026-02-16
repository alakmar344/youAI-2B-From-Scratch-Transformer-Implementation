# YouAI Quick Start Guide

Choose your path based on your resources and goals:

## 🎯 Path 1: Just Testing (FREE - 1 hour)

**Goal**: Understand the training pipeline

```bash
# 1. Install
pip install -r requirements.txt

# 2. Create tiny sample dataset
python prepare_data.py
# Choose option 1, enter 1000

# 3. Test training (will run a few steps)
python train.py
# Press Ctrl+C after 50 steps to stop

# 4. Test inference
python inference.py --checkpoint ./youai_checkpoints/checkpoint-50 --mode chat
```

**What you'll learn**: How training works, the pipeline, data formats

---

## 🎓 Path 2: Learning Project ($10-50 - 2-3 days)

**Goal**: Train a working small model

**Hardware**: Google Colab Pro ($10/month) with T4 GPU

```bash
# 1. Setup in Colab
!git clone <your-repo>
!pip install -r requirements.txt

# 2. Download real data (small)
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1')

# Save to file
with open('train_data.txt', 'w') as f:
    for item in dataset['train']:
        f.write(item['text'] + '\n')

# 3. Reduce model size to 125M parameters
# Edit model_architecture.py:
YouAIConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

# 4. Train
python train.py
# Let it run for 2-3 days

# 5. Test your model
python web_interface.py --checkpoint ./youai_checkpoints/final
```

**Result**: A working 125M parameter model trained by you!

---

## 🚀 Path 3: Serious Project ($200-1000 - 1-2 weeks)

**Goal**: Train a capable 350M-750M model

**Hardware**: Lambda Labs 1x A100 ($1.10/hour)

```bash
# 1. Setup on Lambda Labs instance
git clone <your-repo>
pip install -r requirements.txt

# 2. Download substantial dataset
from datasets import load_dataset
dataset = load_dataset('openwebtext')

# 3. Configure 350M model
# Edit model_architecture.py:
YouAIConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)

# 4. Configure training
# Edit train.py CONFIG:
{
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
    'num_epochs': 3,
    'max_length': 512,
}

# 5. Train (will run for ~7 days)
python train.py

# 6. Deploy
python web_interface.py --checkpoint ./youai_checkpoints/final --port 80
```

**Cost**: ~$185 for 7 days
**Result**: High-quality model suitable for real applications

---

## 💎 Path 4: Full 2B Model ($800-4000 - 2-6 weeks)

**Goal**: State-of-the-art performance

**Hardware**: Lambda Labs 4x A100 or 8x A100

```bash
# 1. Setup on multi-GPU instance
git clone <your-repo>
pip install -r requirements.txt

# 2. Download The Pile or similar (100GB+)
from datasets import load_dataset
dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)

# Save chunks
chunk_size = 100000
for i, chunk in enumerate(dataset.take(5000000)):
    with open(f'train_data_chunk_{i}.txt', 'a') as f:
        f.write(chunk['text'] + '\n')
    if i % chunk_size == 0:
        print(f"Processed {i} examples")

# 3. Use default 2B configuration (no changes needed)

# 4. Enable distributed training
# Edit train.py to use torch.distributed

# 5. Train
python train.py
# Let run for 2-6 weeks

# 6. Deploy and fine-tune
python web_interface.py --checkpoint ./youai_checkpoints/final
```

**Cost**: 
- Lambda 8x A100: $7.20/hour × 7 days = $1,210
- Lambda 4x A100: $3.60/hour × 14 days = $1,210
- AWS similar: $3,000-5,000

**Result**: Production-grade 2B parameter model

---

## 🎨 Path 5: Domain-Specific Model ($50-500 - 3-7 days)

**Goal**: Specialize in one topic (medicine, law, code, etc.)

**Best approach**: Fine-tune existing model OR train small model on domain data

```bash
# Option A: Fine-tune Phi-2 (MUCH FASTER)
from transformers import AutoModelForCausalLM, Trainer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
# Fine-tune on your domain data...

# Option B: Train 350M from scratch on domain data
# 1. Collect 10-50GB domain-specific text
# 2. Follow Path 3 above
# 3. Train for 3-5 days
```

**Cost**: $50-300 depending on size
**Result**: Expert model in your chosen domain

---

## 📊 Quick Reference: Model Sizes

| Parameters | Training Time | GPU Memory | Cost | Best For |
|-----------|---------------|------------|------|----------|
| 125M | 2-3 days | 8GB | $50-100 | Learning, testing |
| 350M | 5-7 days | 16GB | $200-400 | Real applications |
| 750M | 10-14 days | 24GB | $400-800 | Strong performance |
| 2B | 21-30 days | 40GB+ | $1000-4000 | Production quality |
| 7B+ | 60+ days | 80GB+ | $5000+ | State-of-the-art |

---

## 🛠️ Essential Commands

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor training (in another terminal)
tail -f nohup.out

# Check checkpoint size
du -sh youai_checkpoints/*

# Test checkpoint quickly
python inference.py --checkpoint ./youai_checkpoints/checkpoint-1000 \
  --mode generate --prompt "Hello" --max_length 50

# Resume training from checkpoint
# (Training script auto-resumes from latest checkpoint)

# Deploy web interface
python web_interface.py --checkpoint ./youai_checkpoints/final --port 5000
```

---

## 💡 Pro Tips

1. **Start small**: Train 125M first to test pipeline
2. **Monitor closely**: First 1000 steps are critical
3. **Save checkpoints**: Every 1000 steps (default)
4. **Use wandb**: Essential for monitoring long training runs
5. **Lambda Labs**: Cheapest cloud GPUs (use spot instances)
6. **Interrupt anytime**: Training resumes from last checkpoint
7. **Test early**: Generate text at step 500 to check quality

---

## 🚨 Red Flags (Stop Training If You See These)

- ❌ Loss = NaN (learning rate too high)
- ❌ Loss increases (learning rate too high)
- ❌ Loss doesn't decrease after 5000 steps (bad data or config)
- ❌ GPU utilization < 50% (bottleneck somewhere)
- ❌ Perplexity > 1000 after 10k steps (something's wrong)

---

## ✅ Good Signs (You're On Track)

- ✅ Loss steadily decreases
- ✅ Perplexity drops from 1000+ to <100
- ✅ GPU utilization 90-100%
- ✅ Generated text makes some sense after 1000 steps
- ✅ Loss oscillates slightly (normal with small batches)

---

## 🎯 Expected Milestones

**125M Model:**
- Step 500: Perplexity ~500, text is gibberish
- Step 2000: Perplexity ~200, some word patterns
- Step 5000: Perplexity ~80, basic sentences
- Step 10000: Perplexity ~40, coherent but simple
- Final: Perplexity ~30, decent quality

**2B Model:**
- Step 1000: Perplexity ~300
- Step 5000: Perplexity ~100
- Step 20000: Perplexity ~40
- Step 50000: Perplexity ~25
- Final: Perplexity ~20, high quality

---

## 🔄 When Things Go Wrong

**Out of Memory:**
```python
# Reduce batch size to 1
'batch_size': 1

# Reduce sequence length
'max_length': 256

# Use gradient checkpointing (add to model)
```

**Training Too Slow:**
```python
# Increase batch size if you have room
'batch_size': 16

# Reduce sequence length
'max_length': 256

# Use fewer validation steps
'eval_steps': 2000
```

**Loss Not Decreasing:**
```python
# Lower learning rate
'learning_rate': 1e-4

# More warmup steps
'warmup_steps': 2000

# Check data quality!
```

---

Remember: Training from scratch is a journey. Start small, learn the process, then scale up! 🚀
