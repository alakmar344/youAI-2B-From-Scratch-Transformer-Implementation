<div align="center">

# YouAI

**Load, fine-tune, align, merge, evaluate, and serve any open-source LLM — in a few lines of Python.**

A professional, batteries-included toolkit: **15 model families**, **18 datasets**,
**12 export formats**, DPO/ORPO alignment, LoRA/QLoRA fine-tuning, model merging
(DARE/TIES/SLERP), evaluation tools, multi-GPU training, and a production
inference server with streaming and batching.

**Home: [esamz.me](https://esamz.me)** — this toolkit grew out of building
[eSAMz](https://esamz.me) and its AI stack. It started as a 2B-parameter
transformer written from scratch, and grew into the full toolkit it is today
(v3.0.0). 20 modules in `youai/`, 14 test files in `tests/` — everything
listed above is code you can read, run, and test, not marketing copy.

</div>

---

## What you can do in 5 lines

```python
import youai

# Load ANY pretrained model — Llama, Qwen, Mistral, DeepSeek, Gemma, Phi, Falcon...
model = youai.from_pretrained("meta-llama/Llama-2-7b-hf")

# Fine-tune with LoRA
youai.train(model, "my_data.txt", epochs=1, lora=True)

# Align with DPO (human preference optimization)
youai.align(model, "preferences.json", algorithm="dpo")

# Merge two models (DARE, TIES, SLERP, linear)
merged = youai.merge_models(base_model, finetuned, method="dare", alpha=0.5)

# Evaluate
results = youai.run_benchmark(model, tokenizer, data_path="test.txt")

# Serve
youai.serve(pretrained="gpt2", port=8000)
```

## New in v3.0 — Alignment, Merging, Evaluation

### DPO/ORPO/SimPO alignment training
```python
# Direct Preference Optimization — align with human preferences
youai.align(model, "preference_data.json", algorithm="dpo", beta=0.1)
youai.align(model, "prefs.json", algorithm="orpo")  # no reference model
youai.align(model, "prefs.json", algorithm="simpo")  # length-normalised
```

### Model merging (4 methods)
```python
# Linear merge
merged = youai.merge_models(model_a, model_b, method="linear", alpha=0.7)

# SLERP (spherical interpolation — better for language models)
merged = youai.merge_models(model_a, model_b, method="slerp", alpha=0.5)

# DARE (randomly prune delta, rescale — good for multi-model merge)
merged = youai.merge_models(model_a, model_b, method="dare", density=0.2)

# TIES (trim, elect sign, merge — handles conflicting updates)
merged = youai.merge_models(model_a, model_b, method="ties", density=0.2)

# Model soup (average multiple checkpoints)
merged = youai.model_soup([ckpt1, ckpt2, ckpt3])
```

### Evaluation tools
```python
# Perplexity
results = youai.evaluate_perplexity(model, tokenizer, "test.txt")

# Generation quality (diversity, repetition, coherence)
results = youai.evaluate_generation(model, tokenizer, prompts=["What is AI?"])

# BLEU / ROUGE
bleu = youai.compute_bleu(references, hypotheses)
rouge = youai.compute_rouge_l(references, hypotheses)

# Comprehensive benchmark
results = youai.run_benchmark(model, tokenizer, data_path="test.txt")
```

### Model card generation
```python
youai.generate_model_card(model, "README.md", model_name="my-llama",
                          base_model="meta-llama/Llama-2-7b-hf",
                          metrics={"perplexity": 12.3})
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

## Why YouAI?

| | YouAI | nanoGPT | HF Transformers | vLLM |
|---|:---:|:---:|:---:|:---:|
| Load any model (15 families) | **1 line** | ✗ | 5-10 lines | ✗ |
| LoRA fine-tune | **3 lines** | ✗ | 50+ lines | ✗ |
| DPO/ORPO alignment | **1 line** | ✗ | separate lib | ✗ |
| Model merging (DARE/TIES/SLERP) | **1 line** | ✗ | separate lib | ✗ |
| Evaluation suite | **built-in** | ✗ | partial | ✗ |
| 18 dataset presets | ✅ | ✗ | partial | ✅ |
| 12 export formats | ✅ | ✗ | partial | ✅ |
| Inference server | ✅ | ✗ | ✗ | ✅ |
| Readable source | ✅ | ✅ | ✗ | ✗ |
| **175 tests** | ✅ | ✗ | ✅ | ✅ |

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
pytest    # 175 tests, runs in seconds on CPU
```

## Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API reference and
[`CHANGELOG.md`](CHANGELOG.md) for the version history.

## License

MIT — see [`LICENSE`](LICENSE).
