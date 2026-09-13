"""Reduce a unified shadow report to reviewable aggregate and case deltas."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(report, *, only_deltas=False):
    rows = report["rows"]
    result = {key: value for key, value in report.items() if key != "rows"}
    result["cases"] = []
    identities = list(dict.fromkeys(
        (row.get("case_id"), row["category"], row["utterance"])
        for row in rows
    ))
    for case_id, category, utterance in identities:
        item = {"case_id": case_id, "category": category, "utterance": utterance}
        for route in ("existing", "unified"):
            selected = [
                row for row in rows
                if row["route"] == route
                and row.get("case_id") == case_id
                and row["category"] == category
                and row["utterance"] == utterance
            ]
            route_result = {
                "samples": len(selected),
                "decision_kinds": dict(Counter(row["decision_kind"] for row in selected)),
                "llm_calls": sorted({row["llm_calls"] for row in selected}),
                "validation_failures": sum(row["validation_error"] is not None for row in selected),
            }
            for metric in ("plan_correct", "answer_correct", "classification_correct", "payload_correct"):
                scored = [row[metric] for row in selected if row.get(metric) is not None]
                if scored:
                    route_result[metric] = {"correct": sum(scored), "scored": len(scored)}
            operations = Counter(
                (row.get("mutation") or {}).get("operation")
                or (row.get("normalized_plan") or {}).get("operation")
                for row in selected
            )
            route_result["operations"] = {
                str(key): value for key, value in operations.items()
            }
            item[route] = route_result
        semantic_fields = (
            "decision_kinds", "validation_failures", "plan_correct",
            "answer_correct", "classification_correct", "payload_correct",
            "operations",
        )
        if not only_deltas or any(
            item["existing"].get(field) != item["unified"].get(field)
            for field in semantic_fields
        ):
            result["cases"].append(item)
    result["paired_deltas"] = {}
    for metric in (
        "plan_correct", "answer_correct", "classification_correct", "payload_correct"
    ):
        existing = {
            (row.get("case_id"), row["category"], row["utterance"], row["sample"]): row[metric]
            for row in rows if row["route"] == "existing" and row.get(metric) is not None
        }
        unified = {
            (row.get("case_id"), row["category"], row["utterance"], row["sample"]): row[metric]
            for row in rows if row["route"] == "unified" and row.get(metric) is not None
        }
        shared = existing.keys() & unified.keys()
        if shared:
            result["paired_deltas"][metric] = {
                "regressions": sum(existing[key] and not unified[key] for key in shared),
                "improvements": sum(not existing[key] and unified[key] for key in shared),
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-deltas", action="store_true")
    args = parser.parse_args()
    report = summarize(
        json.loads(args.input.read_text()), only_deltas=args.only_deltas
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
