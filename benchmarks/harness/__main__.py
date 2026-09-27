"""``python -m benchmarks.harness``"""

from __future__ import annotations

import sys

from benchmarks.harness.cli import main
from benchmarks.harness.plugins import load_builtin_suites


def _main() -> None:
    load_builtin_suites()
    sys.exit(main())


if __name__ == "__main__":
    _main()
