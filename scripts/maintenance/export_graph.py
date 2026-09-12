"""Export the configured database using the runtime's canonical export library."""
import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from home_cortex.config import get_settings
from home_cortex.db import Database
from home_cortex.export import ExportResult, export_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the connected SurrealDB household graph to canonical "
            "nodes/ and edges/ JSON files."
        )
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory that will receive nodes/ and edges/. Not assumed to be data/.",
    )
    arguments = parser.parse_args()
    result = asyncio.run(_export_connected_database(arguments.target_dir))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


async def _export_connected_database(target_dir: Path) -> ExportResult:
    settings = get_settings()
    database = Database(settings)
    await database.connect()
    try:
        return await export_directory(database, target_dir)
    finally:
        await database.close()


if __name__ == "__main__":
    main()
