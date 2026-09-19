Date: 2026-09-08 20:27 PDT
Type: coding
Status: completed

## Objective

Audit generic query coverage, collect synthetic counterexamples, and implement
declarative location, containment, scope, completeness, and presentation fixes.

## Context

Historical work at commit `e8ce3a6`. The original Grok handoff and implementation
note were separate files from the same work cycle; they are consolidated here.

## Findings

The active schema lacked location and containment declarations, named lookup was not
household-scoped, contradictory predicates passed V1 validation, and truncated
collections could appear exact.

## Decisions

Represent recorded relations declaratively, scope contained aliases through recorded
parent paths, reject declared contradictions, and report incomplete collections
instead of false exact values. Do not guess missing containment.

## Changes

Added semantic bindings and concepts for recorded location/hosting paths, scoped
named contained entities, completeness-aware results, predicate disjointness, and
natural presentation improvements.

## Validation

Full suite: 563 passed. Synthetic counterexamples: 9 passed. A frozen evaluation
candidate and hashes were recorded. No claim is made beyond those historical runs.

## Remaining Issues

The interpreter may still produce legal but incorrect plans; missing recorded scope
paths remain not found; generic rendering can fall back to explicit condition text;
multi-property row projection remains unsupported.

## Recommended Next Step

Recheck these remaining limits against current IR and current benchmark evidence
before extending the query surface.

## Historical Detail

# Grok handoff: evidence and generic-query coverage

These tasks are prepared for Grok; they have not been dispatched through an agent
connection. Keep prompts, ontology, executor and frozen golds unchanged during
baseline measurement. No production writes or deployment.

## G1 — Deployed profile and capability inventory (P0)

Host: `jkuang@home-cortex-0` over Tailscale. Use applicable authorization for
read-only inspection; begin with code/config fingerprints and schema metadata,
not a household-wide record dump. Exclude credentials from every output.

Record source revision/file hashes, active ontology version/hash, model/digest,
Ollama version, num_ctx and effective generated capabilities/schema. Verify whether
adult AND minor is rejected by the actual serving process. Do not assume the
presence of an ontology-v2.yaml file means it is active.

For person, address, space, item and any room/product category, report available
semantic bindings, physical edge signatures, supported direction, ownership/scope,
cardinality, temporal behavior and missing capability mappings. Establish whether
room-to-home containment and item locations are actually represented. Report
metadata/aggregate evidence first; flag any need for additional private records.
Do not add guessed containment relationships.

Deliver REPORT.md plus a small capability/fingerprint summary and a gap matrix:
existing data + missing declaration / missing binding / missing data / genuinely
unexpressible operation.

## G2 — Synthetic execution counterexamples (P0)

Build fixtures, not question handlers. Reproduce scalar age + minor filtering,
then compare explicit each projection, exactly-one filtered result, zero results,
missing dates, ambiguous named roots and ambiguous final sets. Record the full
plan, result status, candidate IDs, row IDs and expected population.

Add multi-edge and >25-result cases for count/select; verify completeness and
whether filters on separate edges retain existing existential semantics. Add two
households containing same-named items to test scope before name resolution.
Do not change executor behavior or loosen scoring. Deliver failing tests/probes
for Codex to review, with the smallest reproducer for each issue.

## G3 — Frozen interpreter evaluation across domains (P1; after G1)

Use the existing frozen contract package and identical V1/V2 conditions. Run the
original transcript and held-out synthetic counterparts, but keep private records
out of the package. Score full plans and expected populations, not found/count
alone. Report wrong-but-legal interpretations separately from schema rejection,
missing knowledge and execution failures.

Include location, verified containment, item categories, members, age filters,
plural projections, nested relatives, ambiguous/missing locations and unsupported
questions. Do not feed sequence YAML through a loader that discards history.
Record first-pass success, retries, decoder errors, call counts and latency.
No prompt tuning, ontology edits or changing frozen labels during a scored run.

## G4 — Presentation fidelity matrix (P1)

Specify paired normal-language and detailed descriptions from validated plans and
synthetic results. Cover gender only, adult+gender, minor+gender, zero, intermediate
filters, edge ownership, exclusions, nested paths, missing data and incomplete
collections. Mark every effective condition and its attachment point.

Propose reusable grammar fragments/metadata, not one complete answer template per
question. Do not change numerical values or add facts. Codex reviews the display
contract and implements the shared composer.

## Return format

Separate observations, inference and proposed fixes. Include commands, synthetic
fixtures, source/schema/data fingerprints and compact summaries. Preserve all
counterexamples. Codex owns the scope/cardinality/compiler design and final change
review; do not spend tokens rewriting those boundaries independently.
# Query generalization implementation

Implemented after the G1–G4 review. These results use invented fixtures; no
production household records were copied into the test suite.

## Change boundaries

- **Ontology:** added semantic bindings for recorded `located_in`, its declared
  inverse `contains`, `hosted_by`, and `hosts_space`. The `room` concept expands
  to the production-supported path `address <- located_in - item{item_type=house}
  <- hosted_by - space{space_type=room}`. No direct space-to-address edge is
  invented.
- **Scope:** edge schemas may declare `scope_parent: true`. Named items and
  spaces are retained only when their recorded parent chain reaches the trusted
  `household_id`. Entity types without this declaration retain existing alias
  behavior, including person and kinship resolution.
- **Validation:** declared predicate disjointness is enforced by the active V1
  ontology as well as V2. `adult AND minor` is rejected unchanged and the
  planner receives the same disjointness declaration. V2 remains opt-in because
  G3 showed worse frozen interpreter accuracy and more decoder failures.
- **Execution:** collection filters run before scalar cardinality. One filtered
  entity proceeds; several remain ambiguous. `projection=each` preserves a row
  whose filter evidence is missing. Relationship reads fetch one sentinel row,
  so a truncated set returns `collection_incomplete` instead of an exact count
  or list.
- **Presentation:** normal chat no longer prepends the internal path trace.
  Count nouns and predicate modifiers come from ontology display metadata.
  Unknown conditions use a faithful generic condition clause. The previous full
  trace remains available through `FactRenderer(detailed=True)`.

## Before / after request traces

| Case | Before | After |
|---|---|---|
| scalar age of minors in an 8-person set | `ambiguous`, 8 unfiltered candidates | `ambiguous`, 3 filtered candidates; `projection=each` returns 3 rows |
| scalar age of minor females | `ambiguous`, 8 candidates | `found`, the one filtered entity |
| one missing birth date in per-person age projection | whole request `filter_input_missing` | successful row set plus one explicit `filter_input_missing` row |
| 30 members with executor cap 25 | `found: 25` | `collection_incomplete`, no false exact value |
| two households each containing an item named Milk | global ambiguity containing both items | only the item whose containment path reaches the trusted household; its location resolves normally |
| `adult AND minor` on active V1 | `VALID` | `INVALID_PLAN` / `CONTRADICTORY_PREDICATES` |
| room count on recorded containment graph | relation unavailable | `found: 1` in the two-home fixture |
| ordinary count response | `查询范围：当前家庭 → 1. 家庭成员。` plus `符合条件的记录数：8。` | `家里有8个人。` |
| adult female count | internal path/filter trace plus generic count | `家里有2位成年女性。` |

## Compatibility

- `SemanticFactRequest` is unchanged.
- Existing relation and property ownership rules are unchanged.
- Stateless requests and conversation/discourse handling are unchanged.
- Person aliases are not restricted by household containment, preserving named
  relatives who do not have a current residence edge.
- V2 includes the same new location/property declarations but is not activated.
- Exact count/list results now require evidence that the relationship fetch was
  complete. This intentionally changes a former false-success result into an
  explicit failure status.

## Verification

```text
python -m pytest -q
563 passed

PYTHONPATH=src python -m pytest -q artifacts/query-generalization/probes/test_counterexamples.py
9 passed
```

Frozen evaluation candidate:

```text
artifacts/generic-contracts/candidate-ce7a162a07a40829.tar.gz
sha256 54bce9933271d4f2d2a2df2bd1f69b12a908e9cf19f67a45d9a65d1c73cd6ed6
payload ce7a162a07a40829d5d05dc910025b14eaa202482ba7634687a9269e3fae2dec
```

Grok's original C1 expected a scalar operation to return three ages. That would
silently convert a scalar request into a row projection. The corrected oracle is
ambiguity over the three filtered entities; explicit `projection=each` is the
supported three-row form.

## Remaining limits

- The interpreter can still produce a wrong but legal plan. Declarative aliases,
  relation signatures, and disjointness reduce that space; they do not prove the
  model understood an utterance.
- A named contained entity without a recorded scope-parent path is reported as
  not found in that household. The resolver does not guess its location.
- The generic natural renderer deliberately falls back to explicit condition
  notation when the ontology has no grammatical fragment for a condition.
- The current IR projects one requested property per row. A row still carries
  the entity display name, so requests such as names plus ages work, but arbitrary
  multi-property tabular projection would require a separately reviewed IR change.
