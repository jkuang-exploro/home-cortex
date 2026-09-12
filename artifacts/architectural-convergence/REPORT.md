# Architectural convergence — 2026-09-11

## Acceptance status

This is a verified local convergence change, **not completion of the full token-efficiency ticket**. Local behavior and ownership improvements are implemented. Production model accuracy, actual token savings, and full-series token reduction remain unverified. No deployment or production model run was performed.

The referenced Grok P0/P1 cleanup report was not found in the working tree. The available evidence is `artifacts/token-latency-audit/REPORT.md`. That report explicitly deferred resolver/IR consolidation and example ablation; it does not establish competing FactRequest/QueryRequest implementations. Source inspection finds neither model in the active package. Its historical 22-example prompt is also different from the current 42-example baseline. Do not attribute intervening prompt growth to this change.

## Final architecture and ownership

| Stage / owner | Input → output | LLM? | Alternatives and decision |
|---|---|---|---|
| API identity mapping / `identity.py` | Authenticated client mapping → exact person ID | No | Exact-ID authentication deliberately does not use fuzzy/name resolution. |
| `AgentService._prepare_request` | Trusted identity, household, clock → `AgentRequestContext` | No | Coordinator context now reaches named mutations without rebuilding it. |
| `SemanticConversationService` | Context + prior user turns → scoped discourse context | Conditional | Persistent focus and stateless replay have distinct storage requirements; both use the same fact service. |
| `SemanticFactPlanner` / provider interpreter | User language + semantic capabilities → `SemanticPlan` | Yes | Plan is a routing envelope, not another fact query. Mutation classification remains a separate call when enabled. |
| `SemanticSchemaRegistry` | Concept syntax → validated `SemanticFactRequest` | No | Concept expansion adds complete paths/filters. It is meaningful compilation, not an interchangeable DTO conversion. |
| `EntityResolver` | Semantic reference + trusted context → `ResolutionResult` | No | Names, self, household, discourse, and traversal share this authority. |
| `matching_named_entities` in `schema_catalog.py` | Candidate identity records + scope → ordered name matches | No | Removed independent DB/replay implementations of alias precedence, matching, and limits. This is a subordinate matching primitive, not another semantic resolver. |
| `SemanticSchemaRegistry`, ontology YAML, edge registry | Semantic property/relation → validated physical binding | No | Display labels and physical storage fields stay out of interpreter capabilities. |
| `HouseholdFactEngine` / operator registry | Valid request + resolved references → `FactResult` | No | Shared evidence gate, collection operations, temporal computation, containment. No second factual answer engine found. |
| `RetrievalService` / SurrealDB | Bound graph reads → records | No | Alias reads now project identity/matching fields instead of full profiles. |
| `FactRenderer` | Request + `FactResult` + ontology → text | No | `FactAnswer` additionally carries request, timing, diagnostics; it does not recompute the result. Mutation renderer now receives the active ontology. |

```text
utterance + AgentRequestContext
  → interpreter → SemanticPlan.request
  → concept expansion + strict validation → SemanticFactRequest
  → HouseholdFactEngine → EntityResolver → graph adapter → SurrealDB
  → FactResult → FactRenderer → FactAnswer
```

There is one fact request model before and after this change. From decoded model output to execution there are two meaningful transformations: ontology expansion and typed validation. Neither was removed. `ResolutionResult` must retain final relationship records for relationship-property operations; `FactResult` represents computed values, rows, missing inputs, ambiguity, and evidence. Merging these would erase responsibilities rather than simplify equivalent representations. Named mutation requests and physical write requests likewise differ: resolving names and validating writable storage fields adds information and authorization constraints.

## Deleted architecture and boundary changes

* Removed `_complete_named_object_location` and both planner branches invoking it. Previously a location utterance could turn a model-selected person plus adult predicate into an item-location lookup, silently dropping filters. Now invalid plans retry and fail closed; no replacement request is synthesized. The regression test now rejects that repair instead of requiring it.
* Removed duplicate alias-selection code from database and JSON replay adapters. Both now use global alias-over-appellation precedence, Unicode normalization, scoped appellations, stable ID ordering, and limits from one implementation. Replay ordering now matches production even when fixtures are stored out of order.
* Propagated the coordinator context through mutation dispatch using task-local state with `finally` cleanup. Concurrent requests preserve their own speaker, household, time, locale, and conversation. Direct structured tool callers without a conversation retain an explicit context-construction boundary. Calendar caller-ID scope remains an authorization adapter.
* Write resolution now uses the writer's ontology, and mutation rendering uses the coordinator's active ontology rather than loading defaults during rendering.
* Removed the redundant planner `references` list: its six names already occur as the keys of `reference_kinds`. Output schema and executable reference vocabulary are unchanged. Sorted relationship-property map construction to eliminate set-order variation before serialization.
* Alias SQL now selects the identity summary fields plus aliases/appellations. Full scans are still required for the existing Unicode-normalized matching contract; no unversioned cross-request index was introduced. Selected fact properties are still loaded deterministically when required.

No full household records, identity bindings, or factual results are introduced into the interpreter prompt. No example was removed without evidence of its accuracy contribution.

## Prompt component classification

| Component | Classification / decision |
|---|---|
| Semantic grammar and strict ownership instructions | Always required; retained. |
| Ontology aliases, concepts, predicates, property ownership | Runtime-derived from active schema; retained. Query-scoped pruning needs compositional evaluation. |
| Output JSON schema | Runtime-derived structural constraints; retained. It is not another household ontology. |
| Examples | Accuracy contribution unmeasured; 42 pairs retained pending controlled ablation. |
| Clock, current utterance | Required contextual inputs. |
| Prior user turns | Conditional discourse input; assistant answers excluded. |
| Identity/location hints and invalid-plan retry hints | Existing conditional interpreter guidance/rejection remains. No repair remains in the removed location path. These heuristics still warrant model-based evaluation. |
| Redundant reference-name list | Removed; reference-kind map is sufficient. |
| Physical bindings, identities, full records, computed facts | Deterministic responsibility; not model input. |

Compact semantic transport remains offline-only in the provider implementation pending real-model acceptance. Enabling its compressed field vocabulary without that evaluation would not establish that a smaller model understands fewer concepts.

## Measurements

Matched local replay uses an isolated `git archive HEAD` baseline, the same virtualenv, `PYTHONHASHSEED=0`, the synthetic semantic-contract fixture, frozen dataset clocks, three passes of 35 requests, and one excluded warm-up pass. Source/data fingerprints are in the summary files. Full per-case outputs are temporary files, not checked in.

| Metric | Current baseline | Candidate |
|---|---:|---:|
| Replay semantic matches | 105/105 | 105/105 |
| Replay structured results | 105/105 | 105/105 |
| Actual inference calls in replay | 0 | 0 |
| Replay pipeline p50 / p95 ms | 1.065 / 1.265 | 1.115 / 1.491 |
| Representative capability bytes | 9,224 | 9,130 |
| Representative message-content bytes | 31,732 | 31,638 |
| Representative wire-message bytes | 37,317 | 37,209 |
| Separate output-schema bytes | 9,111 | 9,111 |
| Example pairs | 42 | 42 |
| Rough content byte/4 token estimate | 7,933 | 7,909.5 |
| Actual model input/output tokens | Unavailable | Unavailable |
| Production Python lines | 17,069 | 17,032 |

There is **no demonstrated latency improvement**. The 94-byte content reduction is about 0.30%, not a deep token-efficiency result. Byte/4 is not a Qwen tokenizer measurement, particularly for bilingual text. Production Python diff is **77 additions / 114 deletions, net −37**. Tests and report files are excluded from this LOC measure.

`prompt-summary.json` contains static message sizes for all requested identity, relationship, membership, age, and container examples, plus a property-query shape. These are prompt measurements, not claims that a model answered those questions. Missing-data behavior is covered by the deterministic suite; the property-query prompt row does not claim missing data in the fixture.

The older token audit reports pre/post-cache means of 4,598 input tokens and 56 output tokens per call, 38 calls for 35 requests, and identical 31/35 structured results. Its 19,833-byte representative message content and 22 examples cannot be compared directly with current actual model tokens, which were not measured. The current prompt is larger than that historical prompt. The ticket's requirement of fewer tokens than before the cleanup series is therefore **not established**.

## Validation and remaining work

Baseline: 702 tests passed. Final candidate: **704 tests passed in 12.21 seconds**, including the SQL projection. All 26 affected retrieval/alias tests, including embedded SurrealDB execution, also passed in the targeted run. Added coverage checks DB/replay precedence and ordering, concurrent trusted-context propagation, and rejection of silent semantic repair. Existing compositional, bilingual, missing-property, relationship-ownership, and speaker-relative tests remain intact.

Concrete remaining acceptance work:

1. Supply/locate the cited Grok cleanup report to reconcile its P0/P1 claims with the actual source.
2. Run matched isolated GPU inference on this candidate, including the removed repair cases. Local replay uses gold plans and cannot detect interpreter accuracy changes.
3. Measure ablations of the 11,313-byte instructions and 10,935-byte examples with held-out compositional cases. Together these dominate current prompt content; removing them solely to meet a byte target would not demonstrate preserved capability.
4. Measure enabled mutation routing separately: its classifier adds a call before factual interpretation, so the fact-only replay does not characterize that production configuration.

No production accuracy change or completed deep prompt convergence is claimed.
