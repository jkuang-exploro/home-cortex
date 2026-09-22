"""Installed ``hc-bench`` command.

The authoritative scorers remain the existing benchmark modules. This entry point
registers adapters, then hands arguments to ``home_cortex.benchmark``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from home_cortex.benchmark.cli import main as cli_main
    from scripts.benchmarks.hc_suites import register

    register()
    sys.exit(cli_main(argv))


if __name__ == "__main__":
    main()
