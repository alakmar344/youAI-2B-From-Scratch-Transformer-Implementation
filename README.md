<div align="center">

# YouAI

**Load, fine-tune, and serve any open-source LLM — in a few lines of Python.**

A professional, batteries-included toolkit supporting **15 model families**,
**18 datasets**, **12 export formats**, LoRA/QLoRA fine-tuning, multi-GPU
training, and a production inference server with streaming and batching.

</div>

---

## What you can do in 5 lines

```python
import youai

# Load ANY pretrained model — Llama, Qwen, Mistral, DeepSeek, Gemma, Phi, Falcon...
model = youai.from_pretrained("meta-llama/Llama-2-7b-hf")
model = youai.from_pretrained("Qwen/Qwen2-7B")
model = youai.from_pretrained("mistralai/Mistral-7B-v0.1")
model = youai.from_pretrained("gpt2")  # still works too

# Fine-tune with LoRA — trains <0.5% of params, adapter is a few MB
youai.train(model, "my_data.txt", epochs=1, lora=True)

# Serve it as an HTTP API with streaming + batching
youai.serve(pretrained="gpt2", port=8000)
```

## Supported model families (15)

| Family | Models | Architecture |
|--------|--------|--------------|
| **Llama** | Llama-2 7/13/70B, Llama-3 8/70B, CodeLlama | RoPE + RMSNorm + SwiGLU + GQA |
| **Qwen** | Qwen2 0.5B/1.5B/7B/72B, CodeQwen | RoPE + RMSNorm + SwiGLU + GQA |
| **Mistral** | Mistral-7B, Mixtral-8x7B | RoPE + RMSNorm + SwiGLU + GQA |
| **DeepSeek** | DeepSeek 7B/67B | RoPE + RMSNorm + SwiGLU |
| **Gemma** | Gemma 2B/7B | RoPE + RMSNorm + GELU |
| **Phi** | Phi-2, Phi-3 Mini | RoPE + RMSNorm + SwiGLU |
| **Falcon** | Falcon 7B/40B | RoPE + LayerNorm + GQA |
| **Yi** | Yi 6B/34B | RoPE + RMSNorm + SwiGLU |
| **Baichuan** | Baichuan 7B, Baichuan2 7B | RoPE + RMSNorm + SiLU |
| **InternLM** | InternLM 7B, InternLM2 7B | RoPE + RMSNorm + SwiGLU |
| **MPT** | MPT 7B | ALiBi + LayerNorm |
| **StableLM** | StableLM 3B | RoPE + RMSNorm + SwiGLU |
| **StarCoder** | StarCoder2 3B/7B | Learned + LayerNorm + GELU |
| **GPT-2** | gpt2/medium/large/xl, distilgpt2 | Learned + LayerNorm + GELU |
| **OpenELMA** | OpenELM | RoPE + RMSNorm + SwiGLU |

```python
# Auto-detect from any HuggingFace model name
model = youai.from_pretrained("NousResearch/Llama-2-7b-hf")
model = youai.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
model = youai.from_pretrained("google/gemma-2b")
```

## Size presets (13)

| Preset | Params | | Preset | Params |
|--------|-------:|-|--------|-------:|
| `nano` | ~7.5M | | `3b` | ~4.2B |
| `micro` | ~19M | | `7b` | ~7.0B |
| `tiny` | ~42M | | `13b` | ~13B |
| `125m` | ~152M | | `34b` | ~34B |
| `350m` | ~454M | | `70b` | ~70B |
| `750m` | ~983M | | | |
| `1.3b` | ~1.7B | | | |
| `2b` | ~3.5B | | | |

## Datasets (18 presets)

```python
youai.list_datasets()  # show all

# Pre-training: tinystories, wikipedia, openwebtext, c4, slimpajama,
#               redpajama, the_pile, oscar, bookcorpus, mc4
# Code:         code, codesearchnet, stackexchange
# Fine-tuning:  alpaca, dolly, sharegpt
# Science:      pubmed, arxiv

train_file, val_file = youai.download_dataset("tinystories", num_examples=50000)
train_file, val_file = youai.download_dataset("alpaca")
```

## Export formats (12)

```python
youai.list_export_formats()  # show all

youai.export_model(model, "onnx", "./model.onnx")
youai.export_model(model, "safetensors", "./model.safetensors")
youai.export_model(model, "huggingface", "./hf_model/")
youai.export_model(model, "vllm", "./vllm_model/")
youai.export_model(model, "gguf", "./meta.json")
youai.export_model(model, "int8", "./int8_model")
youai.export_model(model, "fp16", "./fp16_model")
youai.export_model(model, "torchscript", "./model.pt")
```

## Why YouAI?

| | YouAI | nanoGPT | HF Transformers | vLLM |
|---|:---:|:---:|:---:|:---:|
| Lines to load any model | **1** | ✗ | 5-10 | ✗ |
| Lines to LoRA fine-tune | **3** | ✗ | 50+ | ✗ |
| 15 model families | ✅ | ✗ (GPT-2) | ✅ | partial |
| 18 dataset presets | ✅ | ✗ | partial | ✗ |
| 12 export formats | ✅ | ✗ | partial | ✗ |
| Inference server | ✅ | ✗ | ✗ | ✅ |
| Multi-GPU training | ✅ | ✗ | ✅ | ✗ |
| Readable source | ✅ | ✅ | ✗ | ✗ |
| Test suite (126 tests) | ✅ | ✗ | ✅ | ✅ |

## Quick start

```python
import youai

youai.set_seed(42)

# 1. Create a modern transformer
model = youai.create_model("125m")

# 2. Get some data
train_file, val_file = youai.create_sample_data(2000)

# 3. Train
youai.train(model, train_file, val_file, epochs=1, mixed_precision="bf16")

# 4. Generate
print(youai.generate("The future of AI", checkpoint_path="./checkpoints/final")[0])

# 5. Chat
bot = youai.load_model("./checkpoints/final")
print(bot.chat("Hello!"))
```

## Command line

```bash
youai info --preset 125m                      # architecture + parameter count
youai datasets                                # list dataset presets
youai models                                  # list supported model families
youai models --family llama                   # list Llama model variants
youai formats                                 # list export formats
youai train --preset 125m --dataset tinystories --epochs 3
youai train --pretrained meta-llama/Llama-2-7b-hf --lora --dataset alpaca
youai generate --checkpoint ./checkpoints/final --prompt "Hello"
youai chat --checkpoint ./checkpoints/final
youai export --checkpoint ./checkpoints/final --format onnx
youai serve --checkpoint ./checkpoints/final --port 8000
```

## Installation

```bash
pip install -e .              # core (torch + transformers)
pip install -e ".[server]"    # + FastAPI inference server
pip install -e ".[accelerate]"# + multi-GPU training
pip install -e ".[all]"       # everything
```

## Testing

```bash
pip install -e ".[dev]"
pytest    # 126 tests, runs in seconds on CPU
```

## Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API reference and
[`CHANGELOG.md`](CHANGELOG.md) for the version history.

## License

MIT — see [`LICENSE`](LICENSE).
