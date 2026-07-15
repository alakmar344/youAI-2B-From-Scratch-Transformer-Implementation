"""Backward-compatibility shim for ``from inference import YouAIInference``.

The implementation now lives in :mod:`youai.inference`.
"""

from youai.inference import YouAIInference

__all__ = ["YouAIInference"]


if __name__ == "__main__":
    from youai.cli import main

    main()
