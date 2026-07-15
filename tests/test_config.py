import pytest

from youai.config import YouAIConfig, get_preset_config, list_presets


def test_preset_names():
    presets = list_presets()
    assert "125m" in presets and "nano" in presets


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
