"""Load suite adapters without a static import from the serving package.

The application package must not import ``scripts``. Benchmark commands are the
exception, and they resolve adapters only when the CLI starts: first through the
``home_cortex.benchmark_suites`` entry point, then by module name in a checkout
that has not been installed.
"""

from __future__ import annotations


def load_builtin_suites() -> None:
    loaded = False
    try:
        from importlib.metadata import entry_points

        group = entry_points(group="home_cortex.benchmark_suites")
    except Exception:
        group = ()
    for item in group:
        try:
            item.load()()
        except ModuleNotFoundError:
            continue
        loaded = True
    if loaded:
        return
    import importlib

    module = importlib.import_module("scripts.benchmarks.hc_suites")
    module.register()
