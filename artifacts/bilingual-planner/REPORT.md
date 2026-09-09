# Bilingual semantic planner

Date: 2026-09-09. Isolated GPU validation on `jkuang@home-cortex-0`.
No live household graph was read. Synthetic fixture only.

## Change

Planner instructions are now English-primary. Canonical IR identifiers stay
English. Chinese text remains only for wording that is itself the rule (deixis,
kinship terms, age-threshold particles). Few-shot examples were rebalanced
across Chinese, English, and mixed utterances instead of duplicating Chinese
identity/count rows. Language handling stays at the interpreter boundary.

## Local deterministic suite

`python -m pytest -q`: **608 passed**.

## Production GPU (`qwen3.5:9b`, digest `6488c96fa5fa…`, 100% GPU, ctx 8192)

Isolated trees in `cortex-cortex-api-1`: deployed `ollama.py` as baseline,
candidate package as modified. Warmup 1, repeat 3, 56 first-pass cases.

| Metric | Baseline (deployed Chinese prompt) | Candidate (bilingual) |
|---|---:|---:|
| Example user turns | 37 | 41 |
| Instruction raw tokens | 1949 | 2110 |
| Example raw tokens | 2260 | 2400 |
| Chat `prompt_eval_count` median | 6611 | 7030 |
| Planner P50 / P95 ms | 1396 / 2830 | 1418 / 2809 |
| First-pass semantic match | 49/56 (0.875) | **52/56 (0.929)** |
| All samples | 147/168 (0.875) | **156/168 (0.929)** |
| Chinese | 20/23 (0.870) | **21/23 (0.913)** |
| English | 25/29 (0.862) | **27/29 (0.931)** |
| Mixed | 4/4 (1.0) | 4/4 (1.0) |
| Chinese↔English pair both-correct | 9/10 (0.90) | 9/10 (0.90) |
| Cross-language discourse | 3/4 (0.75) | **4/4 (1.0)** |
| Negative regressions | 6/6 | 6/6 |

Two baseline “unresolved” misses were the model emitting `kind=unresolved`
with `entity_type=person`. After aligning the expected IR, fair baseline is
51/56. Candidate still leads.

`Who is in my household?` is `current_household → member` on all three
candidate samples (not `self → member` or `self → residence`).

## Remaining misses (both languages)

Named item/space location:

- `收纳盒在哪里？` / `Where is the storage box?` keep the literal name but
  emit `entity_type=person` without `location`.
- `展示区里面有什么？` keeps `展示区` but uses `entity_type=address`.
- English “gallery” still walks `current_household → room → contents`.

This is the same weak named-object area as baseline (baseline dropped the
Chinese item name entirely). Not a language-split IR.

## Architectural note

`Where is 爸爸's charger?` is demonstrated as `named_entity` value
`爸爸's charger`. The ontology has no possession relation from a person to
an item, so compiling through `father` then `location` is not a declared
path. Escalate if possession should become a real concept.

## Prompt-size note

Chat prompt tokens rose **+419 median (~6.3%)**. English instructions are
longer in characters than the previous Chinese block even after example
compression. The increase is the cost of bilingual coverage; latency is
not materially worse (P50 +22 ms). Further cuts would drop rules or
demonstrations.

## Removed as redundant

Duplicate Chinese assistant-identity rows, a second oldest-member row, the
extra `gte 30` age-count beside `gte 40`, a second `annual_occurrence`
daughter row, and the two post-discourse duplicate adult/male counts
(replaced by one English re-anchor plus a non-fact chat row).
