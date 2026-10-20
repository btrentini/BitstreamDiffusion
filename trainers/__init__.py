"""Training package exports with lazy loading for evaluation-only utilities."""

from __future__ import annotations

__all__ = ["Trainer"]


def __getattr__(name: str):
    if name != "Trainer":
        raise AttributeError(name)
    from .trainer import Trainer

    return Trainer
