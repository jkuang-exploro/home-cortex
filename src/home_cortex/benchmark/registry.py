"""In-process suite registry.

Adapters register themselves. The runner does not grow a conditional per suite.
"""

from __future__ import annotations

from typing import Protocol

from .types import RunContext, SuiteResult


class BenchmarkSuite(Protocol):
    name: str
    description: str
    requires_real_model: bool
    requires_gpu_host: bool

    def run(self, context: RunContext) -> SuiteResult:
        """Execute the suite. Implementations live outside this package."""


class SuiteRegistry:
    def __init__(self) -> None:
        self._suites: dict[str, BenchmarkSuite] = {}

    def register(self, suite: BenchmarkSuite) -> None:
        self._suites[suite.name] = suite

    def get(self, name: str) -> BenchmarkSuite:
        try:
            return self._suites[name]
        except KeyError as error:
            known = ", ".join(self.names()) or "(none registered)"
            raise KeyError(f"Unknown suite {name!r}. Registered suites: {known}") from error

    def names(self) -> list[str]:
        return list(self._suites)

    def suites(self) -> list[BenchmarkSuite]:
        return list(self._suites.values())


_REGISTRY = SuiteRegistry()


def registry() -> SuiteRegistry:
    return _REGISTRY


def reset_registry() -> SuiteRegistry:
    """Replace the process registry. Tests use this to install fake suites."""

    global _REGISTRY
    _REGISTRY = SuiteRegistry()
    return _REGISTRY
