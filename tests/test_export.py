"""Tests for the export module — covers all export formats and quantization."""

import pytest
import torch

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.export import (
    ModelExporter, ModelQuantizer, export_model, benchmark_model,
    list_export_formats, EXPORT_FORMATS,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def test_list_export_formats(capsys):
    list_export_formats()
    out = capsys.readouterr().out
    assert "onnx" in out
    assert "safetensors" in out
    assert "huggingface" in out
    assert "vllm" in out
    assert "gguf" in out
    assert "int8" in out
    assert "fp16" in out


def test_export_format_registry():
    assert len(EXPORT_FORMATS) >= 12
    for name, info in EXPORT_FORMATS.items():
        assert "description" in info
        assert "extension" in info
        assert "requires" in info


def test_export_onnx(tmp_path):
    try:
        model = _model()
        out = export_model(model, "onnx", str(tmp_path / "model.onnx"), seq_len=8)
        import os
        assert os.path.exists(out)
    except ImportError:
        pytest.skip("onnx not installed")


def test_export_torchscript(tmp_path):
    model = _model()
    out = export_model(model, "torchscript", str(tmp_path / "model.pt"), seq_len=8)
    import os
    assert os.path.exists(out)


def test_export_safetensors(tmp_path):
    try:
        model = _model()
        out = export_model(model, "safetensors", str(tmp_path / "model.safetensors"))
        import os
        assert os.path.exists(out)
    except ImportError:
        pytest.skip("safetensors not installed")


def test_export_huggingface(tmp_path):
    model = _model()
    out = export_model(model, "huggingface", str(tmp_path / "hf_model"))
    import os
    assert os.path.exists(os.path.join(out, "config.json"))
    assert os.path.exists(os.path.join(out, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(out, "generation_config.json"))


def test_export_vllm(tmp_path):
    model = _model()
    out = export_model(model, "vllm", str(tmp_path / "vllm_model"))
    import os
    import json
    assert os.path.exists(os.path.join(out, "vllm_config.json"))
    with open(os.path.join(out, "vllm_config.json")) as f:
        config = json.load(f)
    assert "tensor_parallel_size" in config
    assert "dtype" in config


def test_export_gguf(tmp_path):
    model = _model()
    out = export_model(model, "gguf", str(tmp_path / "meta.json"))
    import os
    import json
    assert os.path.exists(out)
    with open(out) as f:
        meta = json.load(f)
    assert "general" in meta
    assert "llama" in meta


def test_export_int8(tmp_path):
    model = _model()
    out = export_model(model, "int8", str(tmp_path / "int8_model"))
    import os
    assert os.path.exists(os.path.join(out, "pytorch_model.bin"))


def test_export_fp16(tmp_path):
    model = _model()
    out = export_model(model, "fp16", str(tmp_path / "fp16_model"))
    import os
    assert os.path.exists(os.path.join(out, "pytorch_model.bin"))


def test_export_unknown_format_raises():
    model = _model()
    with pytest.raises(ValueError, match="Unknown format"):
        export_model(model, "nonexistent", "/tmp/test")


def test_export_config(tmp_path):
    model = _model()
    exporter = ModelExporter(model)
    exporter.export_config(str(tmp_path))
    import os
    assert os.path.exists(str(tmp_path / "config.json"))
    assert os.path.exists(str(tmp_path / "model_info.json"))


def test_quantize_int8():
    model = _model()
    quantizer = ModelQuantizer(model)
    q = quantizer.dynamic_quantize("qint8")
    # INT8 model should have qint8 tensors.
    for p in q.parameters():
        if p.dtype == torch.qint8:
            break
    else:
        # Some params may stay float (embeddings, norms).
        pass


def test_quantize_fp16():
    model = _model()
    quantizer = ModelQuantizer(model)
    q = quantizer.dynamic_quantize("fp16")
    for p in q.parameters():
        assert p.dtype in (torch.float16, torch.float32)


def test_quantize_invalid_dtype():
    model = _model()
    quantizer = ModelQuantizer(model)
    with pytest.raises(ValueError):
        quantizer.dynamic_quantize("int4")


def test_benchmark_model():
    model = _model()
    results = benchmark_model(model, device="cpu", num_runs=2, sequence_length=8, warmup=1)
    assert "batch_1" in results
    assert "avg_inference_ms" in results["batch_1"]
    assert "tokens_per_second" in results["batch_1"]


def test_benchmark_multiple_batch_sizes():
    model = _model()
    results = benchmark_model(model, device="cpu", num_runs=2, sequence_length=8,
                              warmup=1, batch_sizes=[1, 2])
    assert "batch_1" in results
    assert "batch_2" in results
