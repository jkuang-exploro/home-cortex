# Generic semantic contracts — implementation candidate

Status: implemented and locally verified; **V2 is opt-in and not deployed**.
This candidate implements the minimal contracts from Ticket 2 and uses Ticket 4's
existing composition data unchanged. It makes no live-model accuracy or latency claim.

## Review boundaries

| Layer | Change |
|---|---|
| Interpreter instructions/examples | `ollama.py` unchanged. No extra model or verification chain. |
| Ontology | New `schemas/semantic/ontology-v2.yaml`; existing default V1 file unchanged. |
| Model-facing grammar | Generated property contracts, available vocabulary and typed filter branches. Existing concept-use representation retained. |
| Runtime IR | `SemanticFactRequest`, reference, filter and projection wire fields/defaults unchanged. |
| Deterministic compiler | `semantic_contracts.py` holds immutable type/domain/operator descriptors and deployment bindings; `SemanticSchemaRegistry` consumes them. |
| Validation/execution | V2 rejects unsupported literals, ownership and bindings; validates values at reads. No moving/deleting/inventing conditions. |
| Display | V2 closed-domain labels feed the existing display descriptors. Open domains retain their optional presentation-only `value_labels`; two label maps on a closed domain are rejected. |
| Evaluation | New contract tests reuse the unmodified Ticket 4 gold plans, inputs and scoring. No benchmark wording was copied into prompts. |

The resolved fingerprint covers declarations, catalog types/bindings, relation
resolution and operator signatures. The frozen package separately fingerprints
all source files, including operator implementations. Returned capabilities and
schemas are defensive copies; caller mutation cannot alter cached contracts.

## Implemented contracts

- Explicit property type, entity/relationship applicability and allowed filter
  operators. A property must bind on every possible endpoint type before use.
- Strict booleans, integers, finite numbers, ISO dates and timezone-bearing
  datetimes; no float-to-integer, alias-to-value, owner or path correction.
- Canonical closed string domains and scoped language aliases. Aliases are model
  guidance; a noncanonical value is rejected rather than normalized downstream.
- Nonempty typed membership operands, temporal range shape/order, explicit
  `exists` convention and typed original-anchor comparisons.
- Shallow scalar/list unions for stored name representations and opaque object
  projection for existing address/form-of-address storage. Structured values have
  only `exists` filtering; this does not add object queries or list operators.
- Declared same-collection predicate disjointness; the candidate declares adult
  and minor disjoint. Known asymmetry, overlapping role matches and identical
  fallbacks fail loading. The declaration remains a domain assertion, not a
  general proof of logical inconsistency for arbitrary predicates.
- Complete unavailable concepts are omitted, never shortened. Missing bindings
  are unavailable; known incompatible catalog types fail configuration validation.
- Unknown raw storage properties are not added to V2's public vocabulary.

A new property of a supported type requires its ontology declaration and catalog
binding. It does not require an utterance branch, new renderer sentence, new
operator or new interpreter example. A test adds a numeric property this way.

## Compatibility and deliberate restrictions

V1 remains the production/default loading path, with its existing validation and
vocabulary behavior. The loader accepts both versions. To evaluate V2, construct
**both planner and executor from the same V2 registry**, as shown below. Activating
V2 by replacing the default file should happen only after evaluation and a
supported-storage audit; ship/roll back code and ontology together.

The candidate's gender domain contains male/female, the categories in the
synthetic evaluation graphs. This is not a claim about every production record.
It deliberately does not guess additional categories. The role property remains
an open string domain so recognized-role/fallback policy retains its meaning.
Date properties in this candidate are date-only. Audit datetime-bearing bindings
before activation; do not truncate or rewrite records to satisfy the declarations.

Shallow mixed-type unions support projection and universally compatible filters.
Operations that require one scalar field kind still reject mixed representations;
this preserves the executor's existing scalar/collection boundary. Datetime
validation checks representation, timezone and range order; comparison execution
continues using the existing operator semantics, not a new temporal algebra.

Unsupported typed plans use existing top-level rejection codes. `contract_error()`
provides internal semantic reason strings without raw values. Stored malformed
filter inputs produce `filter_unsupported`; absent collection inputs retain
`filter_input_missing`. Malformed projection inputs use existing property failure
statuses. Resolver-side filter errors now pass through those same status envelopes
instead of becoming an unhandled exception. Missing inputs never become zero
counts by contract coercion.

No duplication-prevention heuristic was added. Repeated spouse hops, intermediate
filters, complete concept expansions, exclusions, pairwise operands and literal
conditions remain unchanged. Duplicate compatible conjuncts also remain present.
Stateless queries and conversation/speaker/household/agent isolation use the
existing paths. Lost or missing discourse still requests clarification.

## JSON Schema limits and model cost

Generated filter branches constrain property names, owner, operator, literal type,
closed-domain value, array length and eligible anchor property names. Runtime
validation additionally checks path-dependent applicability, anchor compatibility,
calendar validity, temporal ordering, predicate disjointness, operation inputs
and the values actually read. Some of these checks are expressible in a larger
schema but are deliberately left to the authoritative runtime rather than
expanding every possible path or operator/property combination.

Local tests inspect generated branches and exercise runtime enforcement. They do
not establish that Ollama enforces every JSON Schema keyword. Grok must check
schema acceptance and constrained decoding on the pinned provider/model. Do not
weaken runtime validation if a decoder ignores a constraint.

Types cannot reject a legal adult-only plan merely because the utterance meant
female, or distinguish two legal but different kinship paths. Full-plan scoring
and counterexample households remain necessary.

There is no additional default model call. A mocked-transport integration test
verifies that a valid persistent V2 follow-up uses one interpreter call. Generated
capabilities and schemas are larger; `candidate-summary.json` and the archived
`contract-views/` record exact bytes/hashes for both versions on each household.
Those byte sizes are not token counts, prefill savings or measured latency.

## Verification

The full local deterministic suite is run with:

```sh
.venv/bin/python -m pytest -q
```

The V2 compatibility test executes all **78 standalone gold cases and 16
conversation sequences**, including acceptable alternatives, against V1 and V2.
It compares complete `FactResult` objects and separately checks gold scoring.
The remaining contract tests cover negative types/domains, dates, malformed data,
projection/edge ownership, unavailable bindings, new properties, immutable caches,
process-independent fingerprints, repeated paths, explicit conjuncts, missing
discourse/isolation and one-call interpretation. No golds or expected populations
were changed to make these checks pass.

Final test totals are recorded in the handoff summary after package verification.

## Frozen package and Grok handoff

Create the deterministic package from this workspace:

```sh
PYTHONPATH=src .venv/bin/python scripts/freeze_contract_candidate.py
```

The archive and `candidate-summary.json` live in this directory. The archive
includes source, schemas, tests, benchmark inputs, both profiles' generated
capabilities/schema views and `CANDIDATE-MANIFEST.json`. It excludes runtime
`data/`, production environment/config files, secrets and old probe outputs.
Its source entries are read-only; changing any payload file requires a new freeze.
Ticket 4's existing uncommitted work is included as dependency input with exact
hashes, not edited or claimed as this ticket's implementation.

After extracting into an isolated directory with the project dependencies:

```sh
PYTHONPATH=src python -m pytest -q tests/test_semantic_v2_contracts.py tests/test_composition_eval.py
```

Select the candidate without changing any package file:

```python
from pathlib import Path
from home_cortex.composition_eval import household_engine
from home_cortex.semantic_ontology import SemanticOntology
from home_cortex.semantic_facts import HouseholdFactEngine, SemanticSchemaRegistry, SemanticFactPlanner

baseline, _ = household_engine("alpha")
ontology = SemanticOntology.from_file(Path("schemas/semantic/ontology-v2.yaml"))
schema = SemanticSchemaRegistry(baseline.schema.catalog, ontology)
engine = HouseholdFactEngine(baseline.dispatcher, schema)
# ollama_service is the pinned, evaluation-only transport instance.
planner = SemanticFactPlanner(ollama_service, schema)
```

Use Ticket 4's sequence loader and trusted focus updates for conversation cases;
do not feed sequence files through the standalone probe loader. Keep model,
model digest, Ollama version, context length, keep-alive, clock and data identical
between V1/V2 runs. Record first-pass accuracy, retries, full-plan and population
correctness, status/clarification behavior, rendering separately, decoder errors
and latency. Preserve the frozen split; report failures rather than patching
prompts or gold plans during the run.

Any production-host evaluation uses `jkuang@home-cortex-0` over Tailscale under
applicable session authorization. This task packages the candidate locally; it
does not deploy it or perform live-model measurements.
