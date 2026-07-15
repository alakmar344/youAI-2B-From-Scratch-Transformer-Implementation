import pytest

from youai.config import (
    YouAIConfig, get_preset_config, list_presets,
    list_architecture_families, list_family_presets, create_from_family,
    ARCHITECTURE_FAMILIES,
)


def test_preset_names():
    presets = list_presets()
    assert "125m" in presets and "nano" in presets
    assert "7b" in presets and "70b" in presets


def test_all_size_presets_valid():
    for name in list_presets():
        cfg = get_preset_config(name)
        assert cfg.hidden_size > 0
        assert cfg.num_hidden_layers > 0
        assert cfg.total_params > 0


def test_param_count_matches_model(tiny_config):
    from youai.model import YouAIModel

    model = YouAIModel(tiny_config)
    assert model.num_parameters == tiny_config.total_params


def test_roundtrip_serialisation(tiny_config):
    restored = YouAIConfig.from_dict(tiny_config.to_dict())
    assert restored.to_dict() == tiny_config.to_dict()


def test_from_dict_ignores_unknown_keys():
    data = get_preset_config("nano").to_dict()
    data["some_future_field"] = 123
    cfg = YouAIConfig.from_dict(data)  # must not raise
    assert cfg.hidden_size == 128


def test_validation_rejects_bad_head_division():
    with pytest.raises(ValueError):
        YouAIConfig(hidden_size=65, num_attention_heads=4)


def test_validation_rejects_unknown_activation():
    with pytest.raises(ValueError):
        YouAIConfig(activation="not-an-activation")


def test_classic_vs_modern_preset():
    modern = get_preset_config("125m", modern=True)
    classic = get_preset_config("125m", modern=False)
    assert modern.position_embedding_type == "rotary"
    assert classic.position_embedding_type == "learned"


# ---- Architecture family presets ----
def test_all_architecture_families_valid():
    for name, overrides in ARCHITECTURE_FAMILIES.items():
        cfg = create_from_family(name)
        assert cfg.hidden_size > 0
        assert cfg.total_params > 0
        assert cfg.architecture_family == overrides.get("architecture_family", "youai")


def test_list_architecture_families():
    families = list_architecture_families()
    assert "llama" in families
    assert "qwen2" in families
    assert "mistral" in families
    assert "gpt2" in families
    assert len(families) >= 10


def test_list_family_presets():
    presets = list_family_presets()
    assert "llama2-7b" in presets
    assert "qwen2-7b" in presets
    assert "mistral-7b" in presets
    assert len(presets) >= 15


def test_family_preset_has_correct_arch():
    llama = create_from_family("llama2-7b")
    assert llama.architecture_family == "llama"
    assert llama.activation == "swiglu"
    assert llama.norm_type == "rmsnorm"
    assert llama.position_embedding_type == "rotary"
    assert llama.num_attention_heads == 32
    assert llama.num_key_value_heads == 8  # GQA


def test_qwen2_preset():
    qwen = create_from_family("qwen2-7b")
    assert qwen.architecture_family == "qwen2"
    assert qwen.rope_theta == 1000000.0
    assert qwen.vocab_size == 152064


def test_mistral_preset():
    mistral = create_from_family("mistral-7b")
    assert mistral.architecture_family == "mistral"
    assert mistral.num_key_value_heads == 8


def test_phi_preset():
    phi = create_from_family("phi-2")
    assert phi.architecture_family == "phi"
    assert phi.hidden_size == 2560


def test_falcon_preset():
    falcon = create_from_family("falcon-7b")
    assert falcon.architecture_family == "falcon"
    assert falcon.norm_type == "layernorm"


def test_gemma_preset():
    gemma = create_from_family("gemma-2b")
    assert gemma.architecture_family == "gemma"
    assert gemma.activation == "gelu"
    assert gemma.tie_word_embeddings is True


def test_deepseek_preset():
    ds = create_from_family("deepseek-7b")
    assert ds.architecture_family == "deepseek"
    assert ds.vocab_size == 100015


def test_yi_preset():
    yi = create_from_family("yi-6b")
    assert yi.architecture_family == "yi"
    assert yi.rope_theta == 5000000.0


def test_baichuan_preset():
    bc = create_from_family("baichuan-7b")
    assert bc.architecture_family == "baichuan"
    assert bc.activation == "silu"


def test_internlm_preset():
    il = create_from_family("internlm-7b")
    assert il.architecture_family == "internlm"


def test_mpt_preset():
    mpt = create_from_family("mpt-7b")
    assert mpt.architecture_family == "mpt"
    assert mpt.position_embedding_type == "alibi"


def test_stablelm_preset():
    slm = create_from_family("stablelm-3b")
    assert slm.architecture_family == "stablelm"


def test_starcoder_preset():
    sc = create_from_family("starcoder-3b")
    assert sc.architecture_family == "starcoder"
    assert sc.position_embedding_type == "learned"


def test_get_preset_accepts_family():
    cfg = get_preset_config("llama2-7b")
    assert cfg.architecture_family == "llama"


def test_override_with_family():
    cfg = create_from_family("mistral-7b", max_position_embeddings=16384)
    assert cfg.max_position_embeddings == 16384
    assert cfg.architecture_family == "mistral"


def test_gpt2_style_preset():
    cfg = create_from_family("gpt2-style")
    assert cfg.architecture_family == "gpt2"
    assert cfg.activation == "gelu_new"
    assert cfg.position_embedding_type == "learned"
