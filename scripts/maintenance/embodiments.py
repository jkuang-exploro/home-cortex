"""Operator-only embodiment configuration against the configured SurrealDB."""
import argparse
import asyncio
import json
from pathlib import Path

from home_cortex.config import get_settings
from home_cortex.mutation.embodiments import EmbodimentWritingService
from home_cortex.persistence.db import Database
from home_cortex.spatial.embodiment import embodiment_as_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "update"):
        commands.add_parser(command).add_argument("record_json", type=Path)
    for command in ("get", "delete", "unassign"):
        commands.add_parser(command).add_argument("embodiment_id")
    assign = commands.add_parser("assign")
    assign.add_argument("embodiment_id")
    assign.add_argument("agent_id")
    commands.add_parser("list")
    print(json.dumps(asyncio.run(_run(parser.parse_args())), ensure_ascii=False, indent=2))


async def _run(args: argparse.Namespace):
    database = Database(get_settings())
    await database.connect()
    try:
        service = EmbodimentWritingService(database)
        if args.command in {"create", "update", "delete", "assign", "unassign"}:
            await service.ensure_constraints()
        if args.command in {"create", "update"}:
            value = json.loads(args.record_json.read_text(encoding="utf-8"))
            body = await getattr(service, args.command)(value)
            return embodiment_as_mapping(body)
        if args.command == "list":
            return [embodiment_as_mapping(body) for body in await service.list()]
        if args.command == "get":
            return embodiment_as_mapping(await service.get(args.embodiment_id))
        if args.command == "delete":
            await service.delete(args.embodiment_id)
            return {"deleted": args.embodiment_id}
        if args.command == "assign":
            await service.ensure_registered_agents()
            return embodiment_as_mapping(await service.assign(args.embodiment_id, args.agent_id))
        return embodiment_as_mapping(await service.unassign(args.embodiment_id))
    finally:
        await database.close()


if __name__ == "__main__":
    main()
