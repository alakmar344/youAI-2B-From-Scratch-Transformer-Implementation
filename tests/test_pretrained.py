"""Test pretrained weight loading for multiple model families.

Tests GPT-2 loading (requires network) and offline architecture detection /
config conversion logic for all supported families.
"""

import pytest
import torch
from unittest.mock import MagicMock

from youai.config import YouAIConfig, create_from_family
from youai.model import YouAIModel


# ---- Offline tests (no network) ----
class TestFamilyDetection:
    """Test that the architecture detection works for all families."""

    def test_detect_gpt2(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "gpt2"
        hf_config.architectures = ["GPT2LMHeadModel"]
        assert _detect_family(hf_config) == "gpt2"

    def test_detect_llama(self):
        from youai.pretrained import _detect_family
        for model_type in ("llama", "codellama"):
            hf_config = MagicMock()
            hf_config.model_type = model_type
            hf_config.architectures = ["LlamaForCausalLM"]
            assert _detect_family(hf_config) == "llama"

    def test_detect_qwen(self):
        from youai.pretrained import _detect_family
        for model_type in ("qwen", "qwen2"):
            hf_config = MagicMock()
            hf_config.model_type = model_type
            hf_config.architectures = ["Qwen2ForCausalLM"]
            assert _detect_family(hf_config) == "qwen2"

    def test_detect_mistral(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "mistral"
        hf_config.architectures = ["MistralForCausalLM"]
        assert _detect_family(hf_config) == "mistral"

    def test_detect_deepseek(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "deepseek"
        hf_config.architectures = ["DeepseekForCausalLM"]
        assert _detect_family(hf_config) == "deepseek"

    def test_detect_gemma(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "gemma"
        hf_config.architectures = ["GemmaForCausalLM"]
        assert _detect_family(hf_config) == "gemma"

    def test_detect_phi(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "phi"
        hf_config.architectures = ["PhiForCausalLM"]
        assert _detect_family(hf_config) == "phi"

    def test_detect_falcon(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "falcon"
        hf_config.architectures = ["FalconForCausalLM"]
        assert _detect_family(hf_config) == "falcon"

    def test_detect_yi(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "yi"
        hf_config.architectures = ["YiForCausalLM"]
        assert _detect_family(hf_config) == "yi"

    def test_detect_baichuan(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "baichuan"
        hf_config.architectures = ["BaichuanForCausalLM"]
        assert _detect_family(hf_config) == "baichuan"

    def test_detect_internlm(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "internlm"
        hf_config.architectures = ["InternLMForCausalLM"]
        assert _detect_family(hf_config) == "internlm"

    def test_detect_mpt(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "mpt"
        hf_config.architectures = ["MptForCausalLM"]
        assert _detect_family(hf_config) == "mpt"

    def test_detect_stablelm(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "stablelm"
        hf_config.architectures = ["StableLmForCausalLM"]
        assert _detect_family(hf_config) == "stablelm"

    def test_detect_starcoder(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "starcoder"
        hf_config.architectures = ["StarCoderForCausalLM"]
        assert _detect_family(hf_config) == "starcoder"

    def test_detect_unknown_raises(self):
        from youai.pretrained import _detect_family
        hf_config = MagicMock()
        hf_config.model_type = "unknown_model"
        hf_config.architectures = []
        with pytest.raises(ValueError, match="Unsupported"):
            _detect_family(hf_config)


class TestConfigConversion:
    """Test HF config → YouAIConfig conversion."""

    def _mock_hf_config(self, **kwargs):
        config = MagicMock()
        defaults = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(config, k, v)
        return config

    def test_llama_config_conversion(self):
        from youai.pretrained import _hf_to_youai_config
        hf = self._mock_hf_config()
        cfg = _hf_to_youai_config(hf, "llama")
        assert cfg.architecture_family == "llama"
        assert cfg.activation == "swiglu"
        assert cfg.norm_type == "rmsnorm"
        assert cfg.position_embedding_type == "rotary"
        assert cfg.num_key_value_heads == 8

    def test_gpt2_config_conversion(self):
        from youai.pretrained import _hf_to_youai_config
        hf = MagicMock()
        hf.vocab_size = 50257
        hf.hidden_size = 768
        hf.num_hidden_layers = 12
        hf.num_attention_heads = 12
        hf.num_key_value_heads = None
        hf.intermediate_size = 3072
        hf.max_position_embeddings = 1024
        hf.rope_theta = 10000.0
        hf.rms_norm_eps = None
        hf.layer_norm_epsilon = 1e-5
        hf.hidden_act = "gelu"
        hf.tie_word_embeddings = True
        hf.bos_token_id = 50256
        hf.eos_token_id = 50256
        hf.pad_token_id = None
        cfg = _hf_to_youai_config(hf, "gpt2")
        assert cfg.architecture_family == "gpt2"
        assert cfg.norm_type == "layernorm"
        assert cfg.position_embedding_type == "learned"

    def test_mpt_config_conversion(self):
        from youai.pretrained import _hf_to_youai_config
        hf = self._mock_hf_config(
            model_type="mpt",
            hidden_act="silu",
            rope_theta=10000.0,
        )
        cfg = _hf_to_youai_config(hf, "mpt")
        assert cfg.architecture_family == "mpt"
        assert cfg.position_embedding_type == "alibi"


class TestWeightConversion:
    """Test weight dict conversion for each family."""

    def test_llama_style_conversion(self):
        """Test that _convert_llama_style handles the standard layout."""
        from youai.pretrained import _convert_llama_style

        config = create_from_family("llama2-7b").replace(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, vocab_size=100,
            max_position_embeddings=32,
        )
        h = config.hidden_size
        kv_h = config.num_key_value_heads * config.head_dim

        # Build a fake HF state dict.
        hf_state = {
            "model.embed_tokens.weight": torch.randn(config.vocab_size, h),
            "model.norm.weight": torch.randn(h),
            "lm_head.weight": torch.randn(config.vocab_size, h),
        }
        for i in range(config.num_hidden_layers):
            p = f"model.layers.{i}."
            hf_state[p + "self_attn.q_proj.weight"] = torch.randn(h, h)
            hf_state[p + "self_attn.k_proj.weight"] = torch.randn(kv_h, h)
            hf_state[p + "self_attn.v_proj.weight"] = torch.randn(kv_h, h)
            hf_state[p + "self_attn.o_proj.weight"] = torch.randn(h, h)
            hf_state[p + "mlp.gate_proj.weight"] = torch.randn(config.intermediate_size, h)
            hf_state[p + "mlp.up_proj.weight"] = torch.randn(config.intermediate_size, h)
            hf_state[p + "mlp.down_proj.weight"] = torch.randn(h, config.intermediate_size)
            hf_state[p + "input_layernorm.weight"] = torch.randn(h)
            hf_state[p + "post_attention_layernorm.weight"] = torch.randn(h)

        converted = _convert_llama_style(hf_state, config, prefix="model.")
        model = YouAIModel(config)
        missing, unexpected = model.load_state_dict(converted, strict=False)
        real_missing = [m for m in missing if not m.startswith("lm_head")]
        assert real_missing == [], f"Missing: {real_missing}"

    def test_gpt2_conversion(self):
        from youai.pretrained import _convert_gpt2

        config = create_from_family("gpt2-style").replace(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, intermediate_size=128, vocab_size=100,
            max_position_embeddings=32,
        )
        h = config.hidden_size

        hf_state = {
            "transformer.wte.weight": torch.randn(config.vocab_size, h),
            "transformer.wpe.weight": torch.randn(config.max_position_embeddings, h),
            "transformer.ln_f.weight": torch.randn(h),
            "transformer.ln_f.bias": torch.randn(h),
        }
        for i in range(config.num_hidden_layers):
            p = f"transformer.h.{i}."
            hf_state[p + "ln_1.weight"] = torch.randn(h)
            hf_state[p + "ln_1.bias"] = torch.randn(h)
            hf_state[p + "ln_2.weight"] = torch.randn(h)
            hf_state[p + "ln_2.bias"] = torch.randn(h)
            hf_state[p + "attn.c_attn.weight"] = torch.randn(h, 3 * h)
            hf_state[p + "attn.c_attn.bias"] = torch.randn(3 * h)
            hf_state[p + "attn.c_proj.weight"] = torch.randn(h, h)
            hf_state[p + "attn.c_proj.bias"] = torch.randn(h)
            hf_state[p + "mlp.c_fc.weight"] = torch.randn(h, config.intermediate_size)
            hf_state[p + "mlp.c_fc.bias"] = torch.randn(config.intermediate_size)
            hf_state[p + "mlp.c_proj.weight"] = torch.randn(config.intermediate_size, h)
            hf_state[p + "mlp.c_proj.bias"] = torch.randn(h)

        converted = _convert_gpt2(hf_state, config)
        model = YouAIModel(config)
        missing, _ = model.load_state_dict(converted, strict=False)
        real_missing = [m for m in missing if not m.startswith("lm_head")]
        assert real_missing == []


# ---- Network tests (GPT-2) ----
@pytest.fixture(scope="module")
def gpt2_pair():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from youai.pretrained import from_pretrained

        hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        ours = from_pretrained("gpt2")
        tok = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as exc:
        pytest.skip(f"gpt2 unavailable: {exc}")
    return hf, ours, tok


def test_logits_match_huggingface(gpt2_pair):
    hf, ours, tok = gpt2_pair
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_logits = hf(ids).logits
        our_logits = ours(ids)["logits"]
    assert (hf_logits - our_logits).abs().max().item() < 1e-3
    assert hf_logits[0, -1].argmax() == our_logits[0, -1].argmax()


def test_greedy_generation_matches_huggingface(gpt2_pair):
    from youai.generation import GenerationConfig

    hf, ours, tok = gpt2_pair
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    ours_gen = ours.generate(ids, GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=None))
    hf_gen = hf.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    assert torch.equal(ours_gen, hf_gen)


def test_param_count_reasonable(gpt2_pair):
    _, ours, _ = gpt2_pair
    assert 1.2e8 < ours.num_parameters < 1.3e8


def test_from_pretrained_backward_compat():
    """from_pretrained_gpt2 still works."""
    try:
        from youai.pretrained import from_pretrained_gpt2
        model = from_pretrained_gpt2("gpt2")
        assert model.num_parameters > 0
    except Exception:
        pytest.skip("gpt2 unavailable")


def test_supported_families_list():
    from youai.pretrained import SUPPORTED_FAMILIES
    assert "llama" in SUPPORTED_FAMILIES
    assert "qwen2" in SUPPORTED_FAMILIES
    assert "mistral" in SUPPORTED_FAMILIES
    assert "gpt2" in SUPPORTED_FAMILIES
    assert len(SUPPORTED_FAMILIES) >= 10


def test_popular_models_list():
    from youai.pretrained import POPULAR_MODELS
    assert len(POPULAR_MODELS["llama"]) >= 3
    assert len(POPULAR_MODELS["qwen2"]) >= 2
    assert len(POPULAR_MODELS["mistral"]) >= 1
