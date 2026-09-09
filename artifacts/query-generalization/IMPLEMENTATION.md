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
