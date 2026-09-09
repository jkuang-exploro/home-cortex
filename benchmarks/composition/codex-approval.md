# Ticket 4 coverage review

Verdict: APPROVE WITH CONDITIONS

## Summary

The annotation guide is the right contract: full expanded plan identity, no silent
repair, predicates ≠ gender, `found`/lucky counts are not correctness, and frozen
wording is split by expression family. The three invented households actually
implement path-sensitivity (alpha FIL≠father; beta inverted ids; gamma
empty/ambiguous/missing/date-boundary/speaker-change). Required ticket contrasts
are present.

Do **not** generate households or YAML until the blocking conditions below are
applied. The claimed **48 / 36** standalone size is arithmetic fiction: §4 sums
to **43 / 32**, §5.1 unique development rows are **42** after the E3/E8-1 merge,
and §5.2 reaches 36 only by adding four extra frozen rows plus the E8 speaker
pair. Padding with `dev-beta-F7-1` to restore 48 is rejected: beta male count is
also 5, the same collision the filler pretends to fix.

K2 `acceptable_plans` including `father_in_law` would let a **non-identical**
path share gold person on alpha. That violates annotation-guide §2 and §5
(expansions differ by the spouse gender filter; they are not equivalent on beta).
K3 existing is not a license to collapse K1/K2.

No frozen utterance is an exact copy of `_example_text`, `_PLANNER_INSTRUCTIONS`,
or the existing eval/heldout/synthetic/age/date YAML. Development may overlap
eval (`家里有几个人`, `我女儿是谁`); that is allowed and must not be copied into
frozen.

## Answers to §10 questions

1. **Size 48+36+8/8 with `dev-beta-F7-1` filler?** No. Merge E3 into E8-1 (do
   not list both). Do not pad. Approved size is **42 development standalone +
   36 frozen standalone + 8/8 sequences** (94 scored last turns). `dev-beta-F7-1`
   may be added only as an honest household transfer with `expected_entity_ids`,
   making development 43 — not as rounding.

2. **Three households and §3 count/id table?** Yes, with entity-id discipline on
   every count collision. Sharp collisions: alpha female=5 vs male=5; alpha my
   children=3 vs household minors=3 (different ids); alpha and beta members=10;
   alpha and beta female=5; proposed beta male=5 vs beta female=5. Lists C1/C2
   already distinguish children vs minors; F1/F7 must not be count-only.

3. **G8 gold: lock `discourse` only, or accept restart-as-wife/husband?**
   **Lock `discourse` only** (`kind=discourse`, `turn_offset=1`,
   `annual_occurrence`, `mode=days`). “她” is a pronoun, not a complete
   first-person kinship phrase. Restart-as-`wife` is execution-equal on alpha
   and **wrong** on frozen G8 (顾南 has no wife; antecedent is 闺女). Accepting
   restart would score a non-discourse plan as correct and would not survive the
   contrast household. Same lock for G7: `argmin` on `discourse` /
   `cardinality=collection`; do not accept `current_household→member` restart
   even though it returns 梅山 after a household-list turn.

4. **E7: accept both `unresolved→ambiguous` and
   `discourse→discourse_context_missing`?** **Yes**, both are clarification, not
   a guessed person. Gold: `kind=unresolved` (cardinality must stay `single`;
   collection cardinality is discourse-only and would fail model construction).
   Acceptable: `kind=discourse`, `turn_offset=1`, `entity_type=person`. Because
   execution statuses differ, the case must accept **either** `ambiguous` **or**
   `discourse_context_missing`. A named-entity or household guess remains wrong.

5. **Frozen wording disjoint from prompt examples?** Exact-string: **yes**.
   Closest non-copies (allowed): eval `Who is my father-in-law?` vs
   `Name the person who is my father in law.`; eval `我妻子的父亲叫什么` vs
   frozen `我妻子的父亲是哪位？`; example `本户符合成年条件的成员有多少？` vs
   `本户已满十八岁的成员有几位？`. Frozen over-uses example-flavored `本户`, but
   that is style, not an exact copy. **Not** disjoint by expression family:
   `frz-beta-K2-1` is canonical family D (`我妻子的父亲是哪位？`) on a household
   transfer — see Issue 4.

6. **K2 `acceptable_plans` including `father_in_law` because K3 exists?**
   **No.** Expanded paths are not identical (`spouse`+`parent[male]` vs
   `spouse[gender=female]`+`parent[male]`). They execute equally on alpha and
   **diverge** on beta (K1 finds 海峰; K2-empty is `relationship_not_found`).
   Annotation-guide §5 requires equivalence on the contrast household. K3
   remaining distinct does not make K1/K2 interchangeable. K2 gold is
   `wife` then `father` only. `father_in_law` is a plan mismatch even when
   alpha ids match; execution-forbidden checks on alpha are `father` and
   `spouse` then `father_in_law` (both 安石). Beta K2-empty is the
   execution-level K1 vs K2 distinguisher.

7. **Any missing ticket contrast?** Required axes are covered: female / adult
   female / minor female (F1/F4/F5); my children / household minors (C1/C2
   lists, different ids); father-in-law / wife’s father / spouse’s father-in-law
   (K1/K2/K3); same last utterance after a different preceding turn (G4, and G8
   if discourse-locked); empty/missing/ambiguous/date-boundary/speaker-change
   (E1–E8, K6); three graphs. Not missing: adult-male as its own cell (ticket
   named the female triple). Weak but acceptable: wife stored-date / countdown
   standalone is alpha-only; husband variants live in frozen G3/G4.

**`frz-beta-K2-1`:** Valid contrast, **not** an invitation to repair. Female
speaker, `wife` then `father` → `relationship_not_found`; K1 `公公` still finds
海峰. Do not list `father_in_law` as acceptable. Rewrite the utterance to a
held-out family (Issue 4).

**Loader:** A new composition loader that reads `history`, leaving
`load_semantic_eval_cases`, `load_probe_dataset`, `DEFAULT_EVAL_PATH`, and
`SCORING_REVISION` unchanged, **preserves Ticket 4 constraints**. Extra keys
must remain ignored by the existing probe loader. Sequence YAML must not be
fed to `load_probe_dataset` (it would drop history and skip leak tests).
Speaker variants (E8) require uniqueness by `id` / `(utterance, speaker_id)`,
not utterance-only as in `load_semantic_eval_cases`.

## Conditions (if any)

Blocking (must land in the matrix before generation):

1. Publish the honest size: **42 / 36 / 8 / 8**. Merge E3 into E8-1. Do not add
   filler rows to restore 48. Correct the §4 footer (cells sum to 43/32 before
   extras).
2. K2 (and K5) `acceptable_plans` must **not** include `father_in_law`. K1 must
   **not** include `wife`+`father` or `spouse`+`father_in_law`.
3. Lock G8 and G7 to `discourse`. No restart-as-`wife`/`husband`/`daughter` /
   household-member acceptable plans.
4. Every count-collision cell asserts `expected_entity_ids` in addition to
   `expected_value` (at least F1, F7, S2-alpha if ever transferred to a 10-member
   graph, and C2-beta). Paired list cells remain required where counts match
   (C1 vs C2).
5. Replace `frz-beta-K2-1` wording with a held-out paraphrase of “wife’s father”
   on the female speaker (or move the transfer to development and add a new
   frozen expression). Keep gold `relationship_not_found`.
6. E7: if both clarification plans are accepted, accept both execution statuses;
   never a guessed referent. `unresolved` gold cannot set `cardinality=collection`.
7. New loader only; no edits to existing eval paths, scoring, ontology, prompts,
   or executor. Frozen strings stay out of `_semantic_planner_examples`.

Non-blocking:

8. Optional `dev-beta-F7-1` as a real transfer (development 43) with male ids ≠
   female ids ≠ alpha male ids — not as padding.
9. Rename `frz-alpha-O2b` / `frz-alpha-F4b` to `split-household-cell-n`.
10. Frozen K5 `我先生的父亲叫什么身份？` is awkward (`身份` may attract
    `household_role`); keep gold as `husband` then `father` `resolve_reference`.
    `先生`/`太太`/`闺女` are absent from ontology aliases — that is acceptable
    held-out language, not a gold change.
11. Collection gender/predicate filters gold on `request.filters`, not member
    path-step filters, and not `adult`+`minor` conjunctions.

## Issues

### Issue 1 — Severity: blocking

- Topic: Size arithmetic and F7 padding
- Description: §1 and the §4 footer claim 48 development standalone. The Dev
  column sums to 43. §5.1 lists 43 rows including duplicate E3/E8-1 (42 unique).
  The proposed 48th case `dev-beta-F7-1` counts males=5 on beta, colliding with
  alpha males, beta males, and beta females.
- Required change: Freeze size at 42/36/8/8 (94 scored last turns). Document
  E3 as E8-1. Do not generate padding.

### Issue 2 — Severity: blocking

- Topic: K2 plan identity
- Description: `father_in_law` vs `wife` then `father` are not the same expanded
  IR. Accepting both on alpha makes a named-concept substitution score as
  nested composition. They are not equivalent on beta, which is the documented
  contrast household.
- Required change: K2 gold = `self→wife→father` only. Point notes at K3 and at
  beta K2-empty. Same rule for K5 (`husband` then `father` only).

### Issue 3 — Severity: blocking

- Topic: Pronoun gold (G8/G7)
- Description: Accepting restart-as-wife/husband lets a plan that ignores
  discourse share the alpha person. Frozen G8’s antecedent is 闺女, not wife.
  G7 household `argmin` is likewise execution-equal after a member list and
  would miss the leak test.
- Required change: Lock discourse-only golds as in §10.3.

### Issue 4 — Severity: blocking

- Topic: Frozen expression family for K2-empty
- Description: `frz-beta-K2-1` uses development canonical wording
  `我妻子的父亲是哪位？` (punctuation-only vs `dev-alpha-K2-1`). Household
  transfer belongs in development; frozen is held-out families.
- Required change: New frozen string, same meaning and gold
  `relationship_not_found`. Do not exact-copy eval `我妻子的父亲叫什么` or
  synthetic `我妻子的爸爸是哪位家庭成员？`.

### Issue 5 — Severity: blocking

- Topic: Count collisions without entity ids
- Description: Alpha female count and male count are both 5. A gender-swapped
  plan would pass a value-only gold. Annotation-guide §8 already requires ids
  plus a paired cell that still differs (F4=3, F5=2, F2=7).
- Required change: Put `expected_entity_ids` on F1/F7 (and any other colliding
  counts). Keep C1 vs C2 as list/id cases.

### Issue 6 — Severity: blocking

- Topic: E7 status union
- Description: `unresolved` executes `ambiguous`; bare `discourse` with no
  prior turn executes `discourse_context_missing` (via `caller_context_missing`
  + `discourse_antecedent`). A single `expected_status: ambiguous` would mark
  the allowed discourse plan wrong.
- Required change: Accept both statuses, or lock one gold and drop the other
  from `acceptable_plans`.

### Issue 7 — Severity: non-blocking

- Topic: Loader uniqueness for E8
- Description: `load_semantic_eval_cases` rejects duplicate utterances.
  `frz-gamma-E8-1` and `E8-2` share `我闺女是哪位？`.
- Required change: Composition loader keys cases by `id` (and speaker), not
  utterance string alone.

### Issue 8 — Severity: non-blocking

- Topic: No invalid golds in the intended IR
- Description: Inspected against current ontology/validator: S3 does not gold
  `self→member` (forbidden, `semantic_plan_unsupported`); adult+minor
  conjunction is not gold; O1/O2/O2b keep `select(birth_date)` distinct from
  `annual_occurrence`; E1 empty adult daughters is `count` → 0 `found`; E2
  missing dob is `property_unavailable` on a named non-resident; E4/E5 birthday
  rule matches `date_difference` years (顾成=18 today, 顾未=17). Unresolved
  with a path or collection cardinality would be invalid — do not emit that.
- Required change: Keep those golds; do not “repair” them during generation.

## Size

**Approve 42 development standalone + 36 frozen standalone + 8 development
sequences + 8 frozen sequences.**

Reject 48 and the F7 filler-as-rounding. 94 scored last turns is enough for
the required contrasts and small enough that every gold and forbidden plan can
be executed. Frozen 36 is reviewable; the extra frozen rows (O2b, F4b, C2
transfer, E8 speaker pair, K2-empty) are real contrasts, not padding, once
K2-empty wording is held-out.

## Coverage

| Ticket axis | Status |
|---|---|
| Scope (`current_household`/`self`, S3 not `self→member`) | Covered |
| Relationship paths (K1/K2/K3/K4/K5/K6) | Covered if K2/K5 stay uncollapsed |
| Property filters vs predicates (F1/F4/F5, E1) | Covered |
| Operations (select / count / argmin / date_difference / annual_occurrence) | Covered; O2b needed so stored-date ≠ countdown |
| Language families (canonical vs 本户/咱家/English/paraphrase) | Covered except K2-empty (Issue 4) |
| Conversational context (G1–G8 leak/restart; E8 speaker change) | Covered if G7/G8 lock discourse |
| Female / adult female / minor female | Covered (F1/F4/F5; F6 list) |
| My children / household minors | Covered as lists with different ids |
| FIL / wife’s father / spouse’s FIL | Covered on graphs where the wrong hop changes person |
| Same question after different preceding turns | Covered (G4; G8 pair if discourse-locked) |
| Multiple households, wrong path ≠ same person/count | Covered if collision cells carry ids |
| Empty / missing / ambiguous / date boundary / speaker change | Covered (E1–E8, K6, gamma clock 2026-09-03) |
| Dev vs frozen by pattern, not random rows | Covered after Issue 4 |
| Ambiguous wording marked | Covered (E3/E8, E7) |
| Frozen disjoint from examples/eval | Exact-string yes |
| Existing eval paths / scoring preserved | Covered by the new-loader plan |
