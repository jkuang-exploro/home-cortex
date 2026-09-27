"""Orchestration for repeatable Home Cortex model benchmarks.

Scoring stays in the existing benchmark runners. This package records provenance,
writes immutable run artifacts, and compares those artifacts. It does not interpret
language or execute household facts.
"""

__all__ = ["registry"]


def __getattr__(name: str):
    if name == "registry":
        from .registry import registry

        return registry
    raise AttributeError(name)
