#!/usr/bin/env python3
"""Estimate development context from static package import reachability.

Includes deferred and type-only imports: this is a conservative navigation
surface, not runtime module loading or a claim about actual agent token usage.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

TASKS = {
    "complete_factual_query": ["semantic_facts"],
    "entity_resolution": ["entity_resolver"],
    "speaker_behavior": ["semantic_conversation", "entity_resolver"],
    "containment": ["entity_resolver"],
    "factual_property": ["semantic_schema"],
}


def measure(root):
    sources = {}
    for path in root.rglob("*.py"):
        suffix = list(path.relative_to(root).with_suffix("").parts)
        if suffix[-1] == "__init__":
            suffix.pop()
        sources[".".join(["home_cortex", *suffix])] = path
    edges = {}
    lines = {}
    for name, path in sources.items():
        text = path.read_text()
        lines[name] = len(text.splitlines())
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        imports = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                          if node.level else node.module or "")
                imports.add(target)
                imports.update(target + "." + alias.name for alias in node.names)
        edges[name] = imports.intersection(sources)
    tasks = {}
    for task, entries in TASKS.items():
        pending = ["home_cortex." + entry for entry in entries]
        reached = set()
        while pending:
            name = pending.pop()
            if name not in reached:
                reached.add(name)
                pending.extend(edges[name] - reached)
        tasks[task] = {
            "entry_modules": entries, "files": len(reached),
            "loc": sum(lines[name] for name in reached),
            "modules": sorted(reached),
        }
    return {"production_files": len(sources), "production_loc": sum(lines.values()), "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Baseline home_cortex package directory")
    parser.add_argument("--after", type=Path, default=Path("src/home_cortex"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"method": __doc__, "before": measure(args.before), "after": measure(args.after)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
