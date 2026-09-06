# benchmarks/

**Dataset inputs for the semantic-planner and fact benchmarks. These are data,
not active source code.**

- `semantic_planner_eval.yaml` — the fixed planner suite (expected semantic IR
  per utterance).
- `semantic_planner_heldout.yaml` and `semantic_planner_synthetic.yaml` —
  held-out / synthetic probe datasets used to test compositional generalization.
- `fixtures/semantic-contract/` — invented household graph records referenced by
  the synthetic probe.

These files are read at fixed paths by `src/home_cortex/semantic_planner_benchmark.py`
and by `tests/test_semantic_planner_benchmark.py`, so their paths must stay stable.
Add new cases here, not in the interpreter prompts. Expected values are semantic IR
only; benchmark wording must never be copied into `_semantic_planner_examples()`.
