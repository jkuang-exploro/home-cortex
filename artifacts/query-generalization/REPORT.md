# Generic household querying — Grok G1–G4

Inspected 2026-09-09 on `jkuang@home-cortex-0` (Tailscale). Read-only.
No credentials, identity maps, or household record dumps appear here.
Private names were not copied into fixtures or reports.

Commands used:

```sh
ssh jkuang@home-cortex-0 'hostname; docker ps; git -C /home/jkuang/Workspace/home-cortex rev-parse HEAD'
ssh jkuang@home-cortex-0 'docker inspect cortex-cortex-api-1'   # env keys only; secrets redacted
ssh jkuang@home-cortex-0 'docker exec cortex-cortex-api-1 python -c "…load_default, validates, aggregates…"'
ssh jkuang@home-cortex-0 'docker exec cortex-ollama-1 ollama --version; ollama ps; ollama list'
# in-network
docker exec cortex-cortex-api-1  # urllib http://ollama:11434/api/tags and /api/show
```

---

# G1 — Deployed profile and capability inventory (P0)

## Observations

### Serving process

| Field | Value |
|---|---|
| Host | `home-cortex-0` |
| API container | `cortex-cortex-api-1`, image `sha256:d21e0910f473…`, created 2026-09-09T02:21:42Z |
| Host git | `adee6e65a986a601646a9836b6be9061d4f3bdcb` “V2 contract candidate”, `master` |
| Package | `/usr/local/lib/python3.12/site-packages/home_cortex` (hashes match this workspace’s `src/home_cortex` for the files below) |
| Data | bind-mount `/home/jkuang/Workspace/home-cortex/data:/app/data:ro` |
| LLM | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen3.5:9b` |
| Ollama | `0.32.15`, model digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, 9.7B Q4_K_M, resident 100% GPU, **effective context 8192** (card context_length 262144) |
| Planner settings | `PLANNER_NUM_CTX=8192`, `OLLAMA_NUM_CTX=8192`, `keep_alive=24h` |
| Resolver cap | `MAX_TOOL_RECORDS=25` passed into `HouseholdFactEngine.max_records`; settings `retrieval_limit=100` is **not** the executor cap |

File hashes (container = this workspace):

| File | sha256 |
|---|---|
| `/app/schemas/semantic/ontology.yaml` | `fb5f4796d0880d702083273f3f9ee9179722eb582acd1741fb9a664c3708d056` |
| `/app/schemas/semantic/ontology-v2.yaml` | `48cca1ca097f4cf81c281fac7c1fd664373a91d8b7179f01098f1b6c651ab8e8` |
| `semantic_ontology.py` | `7b379b14465b852e30c1ff32e240d78f0fb02b2722c6ee2241533baf612d346f` |
| `semantic_facts.py` | `caa080082f2246d116f593c53bf877901b9f29ecbd4bedeaa6bdf110b5e00bd7` |
| `semantic_contracts.py` | `e59af36b64515d838cf84a0ae9f2e8b2b8303795850762f2b4ef20bb79103569` |
| `agent_service.py` | `40055729ca208fb343aefd391d6c502d48e26de06913998dbafe4496829fc5fb` |
| `ollama.py` | `bc004e6f6b158d10ace48ba62fd3a33ff8f4f0ab975cea1d99b00fd77650d679` |

`AgentService` constructs `SemanticSchemaRegistry(schema_catalog)` with **no ontology argument**. `SemanticOntology.load_default()` only searches `ontology.yaml` (package-relative or `/app/schemas/semantic/ontology.yaml`).

### Active ontology is V1

In the serving interpreter:

```
SemanticOntology.load_default().version == 1
schema.contracts is None
schema.validation_code(count members + adult AND minor) == "VALID"
schema.validates(...) is True
schema.contract_error(...) is None
```

`ontology-v2.yaml` **is in the image**. Loading it explicitly yields version 2, `contracts` set, `predicate_disjointness: {adult:[minor], minor:[adult]}`, and:

```
validation_code == "INVALID_PLAN"
contract_error == "CONTRADICTORY_PREDICATES"
validates is False
```

The V2 file is therefore a candidate, not the serving contract. Planner capabilities on the live process have **no** `predicate_disjointness` and **no** `property_contracts` keys.

### Advertised V1 vocabulary (generated, not guessed)

Entity types from the catalog: `person`, `address`, `space`, `item`.

Semantic base relations (only these): `child`, `parent`, `spouse`, `member`, `residence`.

Signatures:

| Relation | From | To | Physical edge | Direction | Temporal | Cardinality notes |
|---|---|---|---|---|---|---|
| `child` | person | person | `parent_of` | out | no | |
| `parent` | person | person | `child_of` (inverse of `parent_of`) | in | no | |
| `spouse` | person | person | `spouse_of` | either (symmetric) | yes | |
| `member` | address | person | `has_resident` (inverse of `lives_in`) | in | yes | |
| `residence` | person | address | `lives_in` | out | yes | |

Not advertised: `located_in`, `hosted_by`, `hosts_space`.

Property ownership (V1 planner payload):

- person: `birth_date`, `display_name`, `family_name`, `form_of_address`, `gender`, `given_name`
- address: `address_type`, `display_name`, `full_address`
- space: `display_name`, `space_type`
- item: `display_name`, `item_type`
- relationship `member`/`residence`: `start_date`, `end_date`, `household_role`, `residence_type`
- relationship `spouse`: `start_date`, `end_date`
- relationship `parent`/`child`: catalog field `type` (from `parent_of.type`)

Predicates: `adult`, `minor` (age fallback 18; role match on `household_role` when present). Concepts: kinship + `member` + `residence` only. No room, item, location, or product concepts.

### Production graph metadata (no identities)

Data tree sha256 `5d065b8e4c99544412e068a5a8f7d619a92e8e8c94d308c45e7fa7bfd035142a`.

| Table / edge | n | Notes |
|---|---:|---|
| person | 8 | 4m/4f; all have `dob`; 8 unique names |
| address | 1 | `address_type=home` |
| space | 13 | 7 room, 3 outdoor_space, 3 storage; all 13 have `hosted_by` |
| item | 10 | types include house, refrigerator, food, furniture/appliances; 7 have `located_in` |
| lives_in | 5 | all current (`end` null); roles owner 2, minor_dependent 2, adult_dependent 1 |
| located_in | 7 | 1 → address, 6 → space |
| hosted_by | 13 | 2 distinct host items |
| parent_of | 8 | |
| spouse_of | 1 | |

Room-to-home: **10/13** spaces are hosted by an item that is itself `located_in` an address. **3/13** spaces are hosted by an item that is not address-located (storage interiors on a nested host). There is **no** space→address edge. `located_in.unique_from` and `hosted_by.unique_from` are true in the edge YAML.

3 items have no `located_in`. 3 persons have no `lives_in`. Within-table name collisions: none. Additional private records are **not** required to answer G1; they would be required only to name specific production items (G3 uses synthetic counterparts instead).

## Inference

1. A live “adult AND minor” gender-count plan is **legal on production**. That matches Ticket 1 traces. Presence of `ontology-v2.yaml` in the image does not change decoder or validator behavior.
2. Location and containment **data exist** in the shape `data/Readme.md` describes (rooms hosted by a house item; some items in spaces). The interpreter cannot select those hops because they are not semantic relations. Wrong plans that use `member` for rooms are therefore **wrong-but-legal**, not schema-rejected.
3. `space_type` and `item_type` are V1-visible properties of space/item records, but there is no legal traversal from `current_household` (address) to space or item. A plan `current_household → member` filtered by `space_type` is a type error; a plan that never leaves address/person cannot list rooms.
4. Executor `max_records=25` is the live cap even though `retrieval_limit=100`. Counts of truncated collections have no incomplete status.

## Proposed fixes (Codex-owned; not implemented here)

1. Do not activate V2 by renaming files without an evaluation run. Serving still uses V1.
2. If adult∧minor should be illegal in production, load the V2 ontology (or the disjointness slice) in `AgentService` after G3, not by assuming the file’s presence.
3. Location/containment: declare relations only if Codex wants them in the catalog; map to existing `located_in` / `hosted_by`; do not add space→address facts.
4. Completeness: counts/selects over a truncated fetch should not report a bare integer as exact. Design is Codex’s.

See `capability-summary.json` and `gap-matrix.md`.

---

# G2 — Synthetic execution counterexamples (P0)

Fixtures live under `benchmarks/fixtures/query-generalization/`. Probes:
`artifacts/query-generalization/probes/test_counterexamples.py`.

These tests assert **desired** execution. They are **not** on pytest `testpaths`, so `python -m pytest -q` stays green. Failures:

```sh
PYTHONPATH=src python -m pytest -q artifacts/query-generalization/probes/test_counterexamples.py
```

No executor, scoring, or prompt changes in this ticket.

Counterexample catalog (smallest reproducers):

| ID | Setup | Desired | Observed on current executor | Class |
|---|---|---|---|---|
| C1 | 8-person invented household; `date_difference(birth_date)` on `current_household→member` + `minor` | Filter then cardinality: 3 minor ages | `ambiguous` with **8** candidates (unfiltered members). `projection=each` returns 3 rows | Filter-after-cardinality for scalar transforms |
| C2 | Same graph; filter that yields **exactly one** minor-age row | Scalar age `found` for that one person | Scalar still `ambiguous` (8) because filters run after resolve | same |
| C3 | Year filter with zero matches; `count` | `found` / 0 | `found` / 0 (control: already correct) | — |
| C4 | `projection=each` + missing `dob` on one minor | row-level missing status; other minors still returned | whole query `filter_input_missing`, `rows=[]` (predicate input aborts the collection) | Predicate missing-input vs per-row |
| C5 | Two people share `name=milk` is wrong domain; two **persons** named `Sam` | named_entity `ambiguous` before traversal | `ambiguous` (control) | Named-root ambiguity must not become a bulk query |
| C6 | Two daughters, `resolve_reference` daughter | `ambiguous` final set | `ambiguous` (control) | — |
| C7 | One person, two current `lives_in` edges, relation filter on `start_date` matching one edge | Document **existential** any-edge match (current spec): person included if any associated edge matches | Person included | Not a bug if Codex keeps existential semantics; recorded |
| C8 | 30 residents; `count` members with `max_records=25` | incomplete / not a silent 25 | `found` value **25**, 25 entity ids | Truncation looks exact |
| C9 | Two households, each has item `name=Milk`; speaker in household A | name resolution must not return B’s milk as a unique hit; prefer scoped miss/ambiguous | Table-wide alias match: **ambiguous** both milks, no household filter | Scope-before-name |

C3/C5/C6 are passing controls so the file is not “everything fails”. C1, C2, C8, C9 fail.

---

# G3 — Frozen interpreter evaluation (P1)

Depends on G1: serving is V1; V2 is opt-in via `SemanticOntology.from_file(ontology-v2.yaml)` on **both** planner and executor.

Package: `artifacts/generic-contracts/candidate-1ad58579b9335830.tar.gz`
(`archive_sha256` `7cdb127bdaaf1e3a794c178f9a66dbbed5a645e2f82fb014fad166998f9e4b6a`).
It excludes `data/` and secrets. Sequences are loaded with `home_cortex.composition_eval.load_composition_file` (keeps `history`). `load_probe_dataset` is not used on sequence YAML.

Harness: `artifacts/query-generalization/probes/run_frozen_interpreter.py`.
It scores **full expanded plans** and expected populations, and buckets:

- `wrong_but_legal` — VALID plan, not gold
- `schema_rejection` — `INVALID_PLAN` / `semantic_plan_unsupported`
- `missing_knowledge` — `relationship_not_found` / `entity_not_found` / `property_unavailable` on a gold that needed data
- `execution_failure` — gold plan itself does not execute as recorded
- `match`

Location/containment/item cases use **synthetic** rooms/items (`benchmarks/fixtures/query-generalization/locus/`), not production records. Gold for “how many rooms” / “where is milk” is **unsupported** under the active V1 catalog (no location relation). A member/adult plan on those utterances is `wrong_but_legal`.

### G3 observations (GPU, isolated package, synthetic graphs)

Command (host `home-cortex-0`, image `cortex-cortex-api`, network `cortex_default`, `PYTHONPATH=/work/src`, **not** `/app/data`):

```sh
python artifacts/query-generalization/probes/run_frozen_interpreter.py \
  --ollama-url http://ollama:11434 --model qwen3.5:9b --profile both
```

49 items × V1 and V2: 36 frozen standalone + 8 frozen sequences (history kept as user turns + planner boundary; **not** `load_probe_dataset`) + 5 location probes. Repeat = 1. Model `qwen3.5:9b` digest `6488c96fa5faab64…`, num_ctx 8192.

| Profile | match | wrong_but_legal | schema_rejection | execution_failure | retries | decoder errors | LLM calls | mean planner ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 serving | 31 | 11 | 6 | 1 | 8 | 4 | 57 | 1867 |
| V2 opt-in | 29 | 9 | 10 | 1 | 13 | 10 | 62 | 2682 |

Frozen standalone V1: 27/36 plan+population match. Sequences V1: 4/8 match. Location: 0 matches.

Wrong-but-legal examples (VALID IR, not gold):

- Gender/adult-female paraphrases compiled as `member`+`adult` without `gender`, or `member` then `child` female (`frz-alpha-F5-1`).
- `Which of them was born first?` → `earliest` on household members, not `discourse` `argmin` (`frz-seq-G7`).
- `家里食物类物品有几件` → household `member` count (V1).

Schema rejection / malformed: several held-out kinship paraphrases (`把我自己的子女都列出来`, `我自己的千金都有谁`) and most location questions. V1 `家里有几个房间` compiled to `select(full_address)` on `current_household` with empty path (found the address string — not rooms). V2 more often `INVALID_PLAN` / `MALFORMED_OUTPUT` / `ValueError` on rooms/milk/food.

`frz-seq-G8`: interpreter emitted gold `discourse` + `annual_occurrence`. Bucket `execution_failure` is a **harness** miss — `DiscourseContext` was not filled from the prior turn. The plan itself matched.

V2 did **not** create location hops. It increased rejection and decoder errors. Adult∧minor was not the location failure mode in this sample (those utterances did not compile to adult∧minor here).

Full rows: `probe-g3-summary.json`. Compact: `probe-g3-compact.json`.

### G3 inference

Held-out gender/adulthood wording is still the main frozen miss, not missing V2 disjointness. Location/item questions are unexpressible; the model substitutes household/self/residence. Activating V2 without new relations will reject more illegal JSON but will not answer rooms or milk. Sequence discourse needs trusted focus in the eval harness before population scoring.

### G3 proposed fixes

Codex: location/containment declarations if those questions should be legal; scoped item names; display vs plan scoring unchanged. Grok should not retune prompts from this run. Re-score G8 after threading discourse focus (plan already gold).

---

# G4 — Presentation fidelity matrix (P1)

Display contract proposal only. Codex implements the shared composer.
`FactRenderer` / `SemanticDisplay` today always emit a path trace (`查询范围：…`). That is the details view, not normal chat.

## Pairing rule

One validated plan + result → two strings from the **same** condition list:

1. **Normal**: grammar fragments composed in order, using ontology labels. No extra facts, no changed numbers.
2. **Detailed**: current `SemanticDisplay.describe` plus per-row status.

Every effective condition is listed with its **attachment point**:

| Attachment | Where in IR | Example |
|---|---|---|
| A0 | `subject.kind` | current household / self |
| A1..An | `subject.path[i].filters` | wife = spouse{gender=female} |
| C | `request.filters` collection AND | `adult`, `gender=female` |
| E | `exclude` | other people |
| O | `other` | pairwise compare |
| P | `property` + `property_source` | entity.birth_date vs relationship.start_date |
| X | `projection=each` | per-entity rows |

## Cases (synthetic; numbers from Ticket 4 alpha / G2, not production)

| Case | Plan conditions | Normal (zh) | Detailed must include | Forbidden |
|---|---|---|---|---|
| Gender only | C: gender=female; count 5 | 家里有五位女性。 | A0 household+member; C gender=female; not adult/minor | “成年人” |
| Adult+gender | C: adult AND gender=female; count 3 | 家里有三位成年女性。 | both predicates | dropping either |
| Minor+gender | C: minor AND gender=female; count 2 | 家里有两位未成年女性。 | both | “孩子” hiding female |
| Zero | C: daughter+adult; count 0 | 家里没有符合条件的成年女儿。 | keep both conditions | “没有人” without filters |
| Intermediate filter | path wife then father | 妻子的父亲是梅山。 | hop1 spouse{female}; hop2 parent{male} | “岳父” if executed path is wife+father (distinct IR) |
| Edge ownership | relationship start_date on spouse | 婚姻开始于 2004-06-18。 | P source=relationship | calling it a birth date |
| Exclusion | collection minus self | 除您以外… | E present | silently including self |
| Nested path | spouse then father_in_law | 配偶的岳父是安石。 | three hops | collapsing to father_in_law |
| Missing data | named select birth_date, no dob | 没有出生日期。 | status `property_unavailable`; no invented date | “0 岁” |
| Incomplete collection | count truncated at 25 | 目前查到 25 人，结果可能不完整。 | incompleteness | “一共 25 人” as exact |
| Each-rows mixed | each age, one missing dob | list names+ages; one “缺出生日期” | row ids and per-row status | dropping the missing row |

## Reusable fragments (not per-question templates)

- `scope_household` / `scope_self`
- `rel_noun(concept)` only if the executed expanded path **equals** that concept’s declared path
- `path_clause(hops)` otherwise (“配偶的父亲”, then extra filters in 的-clauses)
- `pred_adult` / `pred_minor` / `field_gender(value)`
- `and_join(conditions)`
- `count_clause(n, noun)` / `zero_clause(noun)`
- `owner_entity` / `owner_relationship`
- `missing_property(p)` / `incomplete_prefix`

Do not implement a new sentence per utterance. Do not change numerical values.

---

## Cross-ticket status

| Ticket | Status |
|---|---|
| G1 | Done. Serving V1; V2 file present unused; adult∧minor legal; location data without semantic relations. |
| G2 | Fixtures + failing probes for Codex. Executor untouched. |
| G3 | Harness + synthetic location cases; GPU scored run recorded separately. |
| G4 | Display contract only; Codex implements. |
