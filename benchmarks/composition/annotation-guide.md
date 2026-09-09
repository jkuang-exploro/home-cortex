# Compositional evaluation — annotation guide

Ticket 4. This set measures whether the interpreter **composes meaning**.
It does not replace `semantic_planner_eval.yaml`, held-out, synthetic,
age-filter, or date-interval datasets. Those paths and `SCORING_REVISION`
stay unchanged.

## 1. What a case is

Each case has:

| Field | Role |
|---|---|
| `id` | Stable `split-household-cell-n` identifier |
| `utterance` | One current-turn user string |
| `history` | Optional prior **user** turns only (never assistant prose) |
| `plan` | Expected `SemanticFactRequest` after concept expansion |
| `acceptable_plans` | Rare; see §5 |
| `speaker_id`, `household_id`, `frozen_time` | Context |
| `expected_status`, `expected_entity_ids`, `expected_value`, `expected_unit` | Deterministic execution gold |
| `notes` | Why this cell exists; which wrong path must not score as correct |

`found` or an accidentally correct number is **not** semantic correctness.
A male-count plan that returns 5 because it counted adults is a failure even
if 5 also happens to be the male count.

## 2. Plan identity

Compare **full expanded** requests with existing
`normalize_semantic_request`:

- `operation`, `property`, `property_source`, `mode`, `projection`, `exclude`, `other`, `amount`
- subject/other `kind`, `entity_type`, `value`, `turn_offset`, `cardinality`
- every path hop `relation` **and** filters, in order
- request-level `filters` (predicates vs field filters are distinct)

Do **not** collapse:

- `father_in_law` with `wife` then `father` unless both expansions are identical
- `spouse` then `father_in_law` with either of the above (extra hop)
- `child` with household `minor`
- `gender=female` with `adult`
- `select(birth_date)` with `annual_occurrence(birth_date)`
- `self → member` with `current_household → member`

`select(display_name)` may still normalize to `resolve_reference` under the
existing rule. Do not add new equivalences.

## 3. Scope and paths

| User meaning | Subject |
|---|---|
| Household members / “家里…” | `current_household` + `member` |
| “我的孩子/儿子/女儿” | `self` + `child` / `son` / `daughter` |
| “我家里都有谁” | still household members, **not** `self → member` |
| Nested kinship named in **this** turn | ordered concepts from `self` |
| New complete first-person kinship after another kinship turn | restart from `self`; do not keep leftover hops |

Invalid `self → member` is an expected **rejected** plan
(`semantic_plan_unsupported`), not a repaired household query.

## 4. Filters and predicates

- Gender is a **field filter** (`property: gender`, `eq`, `male`/`female`).
- Adulthood is a **predicate** (`adult` / `minor`), never a substitute for gender.
- “成年女性” is `adult` **AND** `gender=female`.
- “女孩子/未成年女性” is `minor` **AND** `gender=female` when the scope is household members; “我的女儿” is `self → daughter`, not household `minor`.
- Collection filters live on `request.filters`. Traversal gender on a named concept stays on that path step (concept expansion).

## 5. Ambiguity and acceptable plans

Mark `expected_status: ambiguous` when the graph has two equally good
referents and the utterance does not uniquely identify one.

Use `acceptable_plans` only when two IRs are **identical after expansion**, or
are execution-equivalent on this household **and** remain equivalent on the
documented contrast household.

`father_in_law` and `wife` then `father` are **not** interchangeable: the
expanded paths differ by the spouse gender filter, and they diverge on the
female-speaker household (K1 finds the husband’s father; K2 is empty).
`spouse` then `father_in_law` is a third path (extra hop). Do not list any of
these as alternatives for each other.

Do **not** accept:

- Wrong operation with the right person
- Missing gender/predicate that happens to count the same set here
- Discourse `kind=discourse` for an explicit kinship phrase (`我老婆`)

Genuinely underspecified wording (`他们`) may expect clarification
(`ambiguous` / `discourse_context_missing`) rather than a guessed person.

## 6. Conversation cases

History is prior user turns with the existing assistant boundary inserted by
the planner. Score **only the last turn**. Preceding turns exist to test leak
and restart, not to be jointly scored.

Required contrasts:

- Gender-count after adult-count (must drop `adult`, add gender)
- Father-in-law after father (must not emit `father → spouse`)
- Wife birthday countdown after in-law identity (must be `wife` + `annual_occurrence`)
- Same last utterance after an unrelated turn (must match standalone)

Speaker, household, conversation, and agent isolation follow existing
`SemanticConversationService` rules. A speaker change is a new context, not
discourse from the previous speaker.

## 7. Development vs frozen acceptance

Split by **composition and expression pattern**, not random rows.

| Split | Purpose | Wording |
|---|---|---|
| Development | Debugging, future prompt work | One canonical family per cell (direct 家里/我的/请列出) |
| Frozen acceptance | Release gate | Held-out families (本户/咱家/English/paraphrase) and household transfer |

Acceptance utterances **must never** appear in `_semantic_planner_examples`
or `_PLANNER_INSTRUCTIONS`. If a case is used to tune prompts, move it to
development and replace the acceptance cell with a new expression pattern.

Existing eval/heldout/synthetic utterances may overlap development; they
must not be copied into the frozen set when the frozen cell is meant to be
unseen.

## 8. Households

At least three invented graphs. Critical path cells must change the **person
or count** under the documented wrong path.

| Household | Speaker | Why it exists |
|---|---|---|
| alpha | male `person:alpha_self` | FIL ≠ father; adult son + minor daughter + minor son; adult female (wife) + minor female (daughter) |
| beta | female `person:beta_self` | Inverted genders/ages; wrong in-law hop yields a different id than alpha |
| gamma | male without spouse; later a second speaker | Empty in-laws; two daughters (ambiguity); missing `dob`; date-boundary 18th birthday; zero-match year filter |

Do not use production `data/` records.

## 9. Deterministic checks (required)

Before any model run:

1. Every expected plan `validation_code == VALID`, except cells whose gold is
   `semantic_plan_unsupported`.
2. Execute gold plans on the named household/clock; status, entity ids, value,
   and unit must match.
3. For each critical contrast, execute the **forbidden** plan and assert it
   is not gold-equivalent (different ids or count).
4. Frozen utterances are absent from interpreter examples/instructions.
5. `load_semantic_eval_cases()` still loads the original eval path.

## 10. Out of scope

No prompt, ontology, executor, or scoring-rule edits in this ticket.
No GPU accuracy claims. Ticket 6 consumes this frozen set.
