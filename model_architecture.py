"""Backward-compatibility shim.

The model now lives in the :mod:`youai` package. This module re-exports the
public classes and a couple of legacy factory helpers so older scripts that did
``from model_architecture import ...`` keep working.
"""

from youai.config import YouAIConfig, get_preset_config
from youai.model import YouAIModel


def create_youai_model(preset: str = "125m", **kwargs) -> YouAIModel:
    """Create a model from a preset (see :func:`youai.create_model`)."""
    return YouAIModel(get_preset_config(preset, **kwargs))


def create_youai_2b(**kwargs) -> YouAIModel:
    """Legacy helper: build the ~2B-parameter preset."""
    return YouAIModel(get_preset_config("2b", **kwargs))


__all__ = ["YouAIConfig", "YouAIModel", "create_youai_model", "create_youai_2b"]
