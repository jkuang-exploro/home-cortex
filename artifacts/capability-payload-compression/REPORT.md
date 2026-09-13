# Surgical capability-payload compression — controlled experiment

No serving capability reduction was accepted. Omitting `property_contracts.applies_to` from the planner projection saved **243 native input tokens** (2.95%) but introduced repeatable kinship and date-plan regressions. The candidate was reverted. Examples, instructions, history policy, model, mutation routing, output schema, and `OLLAMA_NUM_CTX = 16384` are unchanged.

## Baseline capability composition

Production GPU, `qwen3.5:9b`, digest `6488c96fa5fa…`, Ollama 0.32.15. Field bytes match the previous prompt-compression inventory. Native field tokens below were measured on identical JSON for every unchanged field; `property_contracts` native tokens are reported only for the rejected candidate.

| Payload section | Source of truth | Bytes | Native tokens | Planner purpose |
|---|---|---:|---:|---|
| reference_concepts | ontology `planner_payload()` ∩ available concepts | 3,537 | 797 | concept aliases and expansion paths |
| property_contracts | `ResolvedSemanticContract.payload()` | 3,124 | — | types, operators, closed values, owners |
| operation_requirements | handwritten projection in `capability_payload()` | 1,376 | 256 | per-operation argument/output constraints |
| property_aliases | ontology property aliases + ordering | 1,248 | 266 | surface forms for open-world property selection |
| filter_requirements | hardcoded planner prose | 599 | 142 | filter composition, date_range, derived age |
| property_ownership | catalog bindings inverted to entity/relation maps | 488 | 111 | `property_source` choice |
| collection_predicates | ontology predicate aliases + `definition_only` | 450 | 99 | named predicates; not emitted as operations |
| composition | hardcoded planner prose | 397 | 76 | projection / exclude / discourse fields |
| reference_kinds | hardcoded planner prose | 374 | 78 | self / assistant / named_entity / discourse / unresolved |
| relation_signatures | catalog traversal types | 351 | 84 | legal start and end types |
| operations | `FACT_OPERATORS` | 210 | 49 | executable operation names |
| relations | ontology base relations ∩ catalog | 92 | 23 | relation name list |
| entity_types | catalog | 35 | 9 | entity type names |
| predicate_disjointness | ontology `disjoint_with` | 37 | 9 | adult ⊥ minor |
| **planner payload** | derived view | **12,605** | **2,572** | compact JSON body |
| **capabilities component** | `\nCapabilities:\n` + payload | **12,620** | **2,819** | system-prompt section |

Raw component totals remain: capabilities 2,819; examples 2,505; instructions 2,413; clock/reminder 62.

`property_ownership` is the exact inverse of `property_contracts.*.applies_to` on the production catalog (and on the synthetic fixture). Canonical `contracts.payload()` still includes `applies_to`; only the planner view was a candidate for omission.

## Candidate audit

| Candidate | Why it looked redundant | Est. savings | Class | Action |
|---|---|---:|---|---|
| C1. Omit `applies_to` from planner `property_contracts` | Exact inverse of `property_ownership`; instructions name ownership, not `applies_to` | 1,045 bytes / 243 native capability tokens | B derived | **tested, rejected** |
| Empty `applies_to` arrays only | Absence can mean empty | 324 bytes | B | not tested |
| Drop `property_ownership` | Inverse of `applies_to`; instructions name this field | 510 bytes | B | not tested |
| Self-name / underscore-space aliases | Reconstructable from the canonical key | 191–512 bytes | B/D | not tested |
| Identity concept paths (`child`→`[{relation:child}]`) | Single unfiltered step named after the concept | 289 bytes | B | not tested |
| `filter_operators` | Repeated per type, but contracts are the operator authority | 942 bytes | C | not touched |
| `operation_requirements` vs Operations instructions | Overlapping prose; `unit_conversion` and inspect/storage rules are unique | 1,402 bytes | C | not touched |
| `filter_requirements` / `reference_kinds` / `composition` vs contract instructions | Similar wording, extra constraints in one side | 393–622 bytes | C | not touched |
| Merge aliases into contracts | Deduplicate property keys only | 80 bytes | B | trivial |
| Relations list / entity_types | Reconstructable from signatures / ownership | 51–105 bytes | B | trivial |

No exact duplicate (class A) was found. Complementary prose and bilingual aliases were left alone.

## Experiments

Isolated frozen package `candidate-3277d5a978363fea`, frozen graph SHA-256 `a6ab9fa06584…`, same model digest and Ollama 0.32.15 as the retained baseline. Example fingerprint unchanged (42 pairs, 10,935 bytes, 2,505 tokens).

| Candidate | Tokens saved | Plan regressions | Answer regressions | Accepted? |
|---|---:|---|---|---|
| C1 omit `applies_to` | **243** native (8,243.5 → 8,000.5 fixed median; 8,245 → 8,002 held-out median) | 6 new held-out plan failures; 3 repeat-stable | 3 new held-out answer failures; 2 repeat-stable | **No** |

Fixed 12-query × 3: no new failures. Same two baseline misses (kitchen inventory, assistant birth date) on every pass. Median output tokens 47; attempts 1; retries 0; no length stops; all calls under the 8,500-token normal budget.

Held-out 171 × 1:

| Metric | Baseline | C1 | Change |
|---|---:|---:|---:|
| Plan correctness | 155/171 (90.64%) | 149/171 (87.13%) | −6 |
| Differential answer correctness | 151/171 (88.30%) | 148/171 (86.55%) | −3 |
| Retries | 3 | 5 | +2 |
| INVALID_PLAN | 2 | 2 | 0 |
| Median planner latency | 1,481.629 ms | 1,477.912 ms | −0.25% |
| P95 planner latency | 1,691.681 ms | 1,771.350 ms | +4.71% |

New held-out plan failures: `从结婚到现在过了几天`, `哪个男孩是我儿子`, `我的男性子女何时出生`, `告诉我家庭地址`, and `有几个孩子` (two case ids). Answer regressions were the first three of those.

Three further passes over those five utterances (6 cases × 3 = 18 requests):

| Utterance | Baseline 3-pass | C1 3-pass | Repeat-stable C1 regression? |
|---|---|---|---|
| 哪个男孩是我儿子 | 3/3 plan and answer | 0/3 plan and answer | **yes** — `select` instead of `resolve_reference` |
| 我的男性子女何时出生 | 3/3 plan and answer | 0/3 plan and answer | **yes** — extra `parent` hop after `son` |
| 有几个孩子 | 3/3 plan (answer 3/3) | 0/3 plan (answer 3/3) | **yes** — `self→child` instead of household `member`+`minor` |
| 从结婚到现在过了几天 | 3/3 | 3/3 | no (one-shot in 171 only) |
| 告诉我家庭地址 | 0/3 plan, 3/3 answer | 0/3 plan, 3/3 answer | no (not candidate-only under confirmation) |

C1 overall on this slice: 3/18 plans, 12/18 answers. Baseline: 15/18 plans, 18/18 answers.

## Rejected candidates

Only C1 was tested. The planner still had `property_ownership`, so owners were not deleted; they were no longer colocated with type/operators/values. That was enough to destabilize kinship identity, kinship path composition, and the child/minor distinction.

## Accepted final state

No payload field was removed. Serving `planner_capability_payload()` again emits full contracts including `applies_to`. Deterministic suite: **768 tests passing** (767 prior plus an inverse-ownership invariant test). Planner input remains **8,243.5** median native tokens on the fixed set.

## Recommendation

Remaining payload appears semantically dense. A mechanically lossless derived field still changed planner behavior. Further capability deletion is not justified from this experiment. Mutation-planner bypass remains a separate target and was not mixed into this run.
