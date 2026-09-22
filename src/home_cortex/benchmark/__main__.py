"""``python -m home_cortex.benchmark``"""

from __future__ import annotations

import sys

from home_cortex.benchmark.cli import main
from home_cortex.benchmark.plugins import load_builtin_suites


def _main() -> None:
    load_builtin_suites()
    sys.exit(main())


if __name__ == "__main__":
    _main()
