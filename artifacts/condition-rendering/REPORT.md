# Faithful condition rendering — Ticket 3

Implemented locally, not deployed. This is a renderer change, not a production
accuracy or latency result. No production graph or model was accessed.

## What changed

`FactRenderer` now composes one query-scope description from the validated,
concept-expanded request, followed by the execution result. It preserves:

- Every field filter and declared predicate, combined with AND, including zero
  counts, empty selections and empty projections.
- Ordered traversal steps and their entity-versus-relationship filter ownership.
- Final collection filters separately from intermediate filters, exclusions and
  the other operand of a comparison. Repeated relationship hops remain intact.
- Anchor comparisons against the original reference root.
- Independent any-associated-edge conditions on entity collections versus
  same-edge conditions on relationship projections.
- Per-row results and missing evidence, with the scope printed once.

Removed `_count_noun()` and its child-role shortcut. Count/select responses no
longer silently replace the condition set with “adult”, “child” or “household
member”. Existing natural summaries remain where applicable, with corrections
for non-equality gender filters, non-age comparisons and non-household scopes.
Numerical execution, resolution, IR, interpreter instructions and retries are
unchanged. The service supplies its own engine ontology to the renderer.

## Before / after

These are actual renderer outputs from baseline `9c8135e` and the working tree,
using identical synthetic requests and `FactResult` values. Baseline renderer
methods were loaded from `git show` into an isolated Python namespace; neither
checkout nor production files were changed. The deterministic integration tests
also execute these three queries against the invented household fixture in
`tests/test_semantic_contract.py` and verify values 2, 3 and 0 respectively.

### Adult AND female, result 2

Before:

```text
家里目前有两位成年人。
```

After:

```text
查询范围：当前家庭 → 1. 家庭成员；结果筛选：成年人 且 实体.性别 = 女性。
符合条件的记录数：2。
```

### Female only, result 3

Before:

```text
家里目前有三个人。
```

After:

```text
查询范围：当前家庭 → 1. 家庭成员；结果筛选：实体.性别 = 女性。
符合条件的记录数：3。
```

### Adult AND female AND birth date in [1800-01-01, 1801-01-01), result 0

Before:

```text
家里目前有零位成年人。
```

After:

```text
查询范围：当前家庭 → 1. 家庭成员；结果筛选：成年人 且 实体.性别 = 女性 且 实体.出生日期: date_range(["1800-01-01", "1801-01-01"]) [含起点，不含终点]。
符合条件的记录数：0。
```

An intermediate-filter description is deliberately explicit:

```text
Query scope: you → 1. spouse {entity.gender = female} → 2. parent {entity.gender = male}.
```

This does not attach the spouse's gender to the parent, or remove either condition.

## Display metadata and compatibility

The Ticket 2 display interface is implemented as a small additive V1 extension:
optional `label` maps on properties, predicates and concepts, plus property
`value_labels`. These maps are immutable after loading and excluded from planner
capabilities/schema. They contain presentation text, not domain restrictions,
alias-based normalization, type validation or semantic repair.

```yaml
properties:
  gender:
    fields: [gender, sex]
    aliases: [性别, gender, sex]
    label: {en: gender, zh: 性别}
    value_labels:
      female: {en: female, zh: 女性}
      male: {en: male, zh: 男性}
```

The full Ticket 2 V2 contract implementation remains separate. When it is
implemented, migrate `value_labels` into the designed `values.<value>.label`
representation where appropriate; do not maintain two authoritative label maps.
Having a display label now does not establish membership of a future closed domain.

New loaders accept old unlabeled V1 files. Old loaders reject the newly introduced
metadata keys, so deploy/roll back source and ontology together. No HTTP or IR
shape changes occur. Response wording changes intentionally; consumers must not
parse natural-language counts or depend on one output line per result row.

## Verification

Run `.venv/bin/python -m pytest -q` locally. The test suite covers:

- English/Chinese adult + gender, minor + gender, gender alone and zero results.
- Date bounds, membership, equality/inequality, ordered comparisons and exists.
- Intermediate and repeated concept paths, dynamic anchors, edge ownership,
  exclusions and comparison operands.
- The counterexample where two edge conditions match different edges: entity
  count is 1, but same-edge projection is empty. Presentation distinguishes them.
- Partial projections and missing evidence without an invented count of zero.
- A new synthetic `preferred_language` property added through ontology/catalog
  metadata and rendered by the real service without any condition-specific code
  or LLM call.
- Missing translations/metadata, unknown literal values, plain-text escaping and
  invalid display declarations.
- Identical planner payloads, capabilities and output schema with/without labels;
  request/result objects unchanged by rendering.

Final full-suite result: **499 passed in 7.93s**. `git diff --check` also passed.

## Remaining presentation limitations

- Generic descriptions favor precision: numbered paths, ownership markers and
  operator notation can be verbose. They are not fluent conversational Chinese.
  Unknown operators are not admitted by this change; supported operators without
  prose labels use explicit notation, including `exists(false)` and `date_range`.
- English and Chinese are the supported response languages. Label fallback is
  exact locale, base language, English, then semantic key/literal. New value labels
  are presentation-only and never infer gender or age from another field.
- Internal record IDs remain hidden; trusted ID references are described as a
  “specified entity”. Discourse references show offset and cardinality rather than
  guessing a name from conversation prose. Named references retain name/type.
- Query scope describes the submitted validated plan even when execution stops
  for missing evidence. It does not claim every conjunct was evaluated, or expose
  an exhaustive trace of short-circuiting, default predicate policy or graph reads.
- Existing specialized date/identity result wording remains. This ticket does not
  redesign every operation's prose or introduce a natural-language grammar engine.
- Rendering cannot recover an omitted gender filter or correct a legal but wrong
  relative path. Such plans are described faithfully and remain interpreter errors.
