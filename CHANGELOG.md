# Changelog

## 3.0.0 — Alignment, merging, evaluation

### New features

#### Alignment training (DPO, ORPO, SimPO)
- `youai.align(model, "preferences.json", algorithm="dpo")` — align a model
  with human preferences using DPO, ORPO, or SimPO.
- `PreferenceDataset` class for loading preference data (JSON/JSONL with
  prompt/chosen/rejected).
- `DPOConfig` with all hyperparameters (beta, label_smoothing, lr scheduler,
  mixed precision, etc.).
- `create_preference_data()` to generate sample preference data for testing.
- `DPOTrainer` with full training loop, evaluation, checkpointing.

#### Model merging (4 methods)
- `youai.merge_models(model_a, model_b, method="linear|slerp|dare|ties")`.
- **Linear**: simple weighted average.
- **SLERP**: spherical linear interpolation (better for LLMs).
- **DARE**: Drop And REscale (randomly prune delta, rescale).
- **TIES**: Trim, Elect Sign, Merge (handles conflicting updates).
- `youai.model_soup([ckpt1, ckpt2, ckpt3])` — average multiple checkpoints.
- `youai.merge_lora_into_base(model)` — fold LoRA adapters into base weights.
- `youai.list_merge_methods()` — show all methods with descriptions.

#### Evaluation tools
- `youai.evaluate_perplexity(model, tokenizer, "test.txt")` — sliding-window
  perplexity evaluation.
- `youai.evaluate_generation(model, tokenizer, prompts)` — diversity (distinct-n),
  repetition rate, lexical diversity.
- `youai.compute_bleu(references, hypotheses)` — BLEU score (no external deps).
- `youai.compute_rouge_l(references, hypotheses)` — ROUGE-L F1 score.
- `youai.evaluate_multiple_choice(model, tokenizer, questions)` — MMLU-style
  evaluation using log-probability scoring.
- `youai.run_benchmark(model, tokenizer)` — comprehensive benchmark suite.

#### Performance
- `youai.compile_model(model, mode="default|reduce-overhead|max-autotune")` —
  torch.compile integration for free speedup.

#### Deployment
- `youai.generate_model_card(model, "README.md")` — HuggingFace-compatible
  model card generation with metrics, training config, and citations.
- `plot_training_curve()` — ASCII training curve visualization.

### Tests
- Test suite grew from 126 → **175 tests**, all passing.
- Added alignment loss function tests (DPO, ORPO, SimPO).
- Added merge method tests (linear, SLERP, DARE, TIES).
- Added evaluation tests (BLEU, ROUGE, perplexity, generation quality).
- Added advanced utility tests (compile, model card, training curve).

## 2.0.0 — Universal pretrained model loading

### Breaking changes
- `from_pretrained_gpt2()` is deprecated. Use `from_pretrained()` instead — it
  supports all 15 model families and auto-detects the architecture.
- `use_attention_bias` config field added (defaults to `False`). GPT-2 style
  presets set it to `True`. This only affects new models; existing checkpoints
  are unaffected.

### New features

#### 15 pretrained model families
- `youai.from_pretrained("meta-llama/Llama-2-7b-hf")` — load ANY supported
  model with a single function call. Auto-detects architecture from HuggingFace
  config. Supports: Llama (1/2/3), Qwen (1/1.5/2), Mistral, DeepSeek, Gemma,
  Phi (1/2/3), Falcon, Yi, Baichuan, InternLM, MPT, StableLM, StarCoder,
  OpenELMA, GPT-2.
- `youai.models` CLI command to list all supported families and popular models.

#### 13 size presets + 37 family presets
- New size presets: `tiny`, `3b`, `7b`, `13b`, `34b`, `70b`.
- 37 architecture family presets: `llama2-7b`, `qwen2-7b`, `mistral-7b`,
  `phi-2`, `gemma-2b`, `falcon-7b`, `deepseek-7b`, `yi-6b`, `baichuan-7b`,
  `internlm-7b`, `mpt-7b`, `stablelm-3b`, `starcoder-3b`, `gpt2-style`, etc.
- `youai.create_from_family("llama2-7b")` or `youai.get_preset_config("llama2-7b")`.
- `youai.list_architecture_families()` and `youai.list_family_presets()`.

#### 18 dataset presets
- New: `c4`, `slimpajama`, `redpajama`, `the_pile`, `oscar`, `bookcorpus`,
  `codesearchnet`, `stackexchange`, `alpaca`, `dolly`, `sharegpt`, `pubmed`,
  `arxiv`, `mc4`.
- `youai.list_datasets(category="code")` to filter by category.
- `youai.list_datasets_by_category()` to group datasets.

#### 12 export formats
- New: `safetensors`, `huggingface`, `vllm`, `gguf`, `coreml`, `openvino`,
  `ctranslate2`, `tflite`.
- `youai.list_export_formats()` to see all formats with descriptions.
- `youai.export_model(model, "vllm", "./vllm_model/")` for vLLM serving.
- `youai.export_model(model, "safetensors", "./model.safetensors")` for safe
  serialization.
- `youai.export_model(model, "huggingface", "./hf_model/")` for HuggingFace
  compatibility.
- `youai.formats` CLI command.

#### Other improvements
- `use_attention_bias` config option for correct bias handling per architecture.
- Multi-batch benchmarking: `benchmark_model(model, batch_sizes=[1, 4, 8])`.
- Server now accepts any pretrained model name (not just GPT-2).
- CLI `--pretrained` flag now accepts any HuggingFace model name.

### Bug fixes
- Fixed duplicate keyword arg in `_hf_to_youai_config`.
- Fixed `safetensors` export failing on tied weights (clone before save).
- Fixed param count formula to account for `use_attention_bias`.

### Tests
- Test suite grew from 61 → **126 tests**, all passing.
- Added offline architecture detection tests for all 15 families.
- Added weight conversion tests for Llama-style and GPT-2-style.
- Added export format tests for all 12 formats.
- Added dataset preset validation tests.

## 1.1.0 — LoRA, multi-GPU, inference server

- LoRA / QLoRA fine-tuning.
- Multi-GPU training via accelerate.
- FastAPI inference server with streaming, batching, OpenAI-compatible API.
- GPT-2 pretrained weight loading (token-for-token identical to HuggingFace).
- 61-test pytest suite.

## 1.0.0 — Initial release

- Modern transformer architecture (RoPE, RMSNorm, SwiGLU, GQA, flash attention).
- Training loop with mixed precision, gradient accumulation, warmup+cosine schedule.
- KV-cache generation, streaming, chat sessions.
- ONNX / TorchScript export, INT8 / FP16 quantization.
- CLI interface.
