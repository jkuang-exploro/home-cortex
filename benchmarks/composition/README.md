# Compositional generalization evaluation

Ticket 4 dataset. Measures whether the interpreter **composes meaning** rather
than memorizing known questions. It does not replace
`benchmarks/semantic_planner_eval.yaml` and does not change `SCORING_REVISION`.

| Path | Role |
|---|---|
| `annotation-guide.md` | Plan identity, splits, households, checks |
| `coverage-matrix.md` | Approved size **42 / 36 / 8 / 8** and case inventory |
| `codex-approval.md` | Coverage review (approve with conditions, now applied) |
| `households/{alpha,beta,gamma}/` | Invented graphs; not `data/` |
| `development/` | Canonical wording; may overlap existing eval strings |
| `frozen/` | Held-out wording; never prompt examples |
| `frozen/MANIFEST.yaml` | Frozen case list |
| `fingerprints.json` | Dataset / ontology / test hashes |

Load with `home_cortex.composition_eval`. Sequence files have no `probe` block
so `load_probe_dataset` will not silently drop `history`. Ticket 6 consumes the
frozen set; this ticket does not claim GPU accuracy.

`_emit.py` rebuilds households and gold execution fields from the inventory.
Do not run it to “fix” a frozen case after the case has been used as an
acceptance gate; move tuned wording into development instead.
