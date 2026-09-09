# Compact semantic transport v1

> **WITHDRAWN FROM SERVING.** The user reported universally negative responses
> after enabling compact transport. Ollama and OpenRouter now use expanded JSON
> as their sole serving format. The codec and archive remain offline research
> artifacts; do not deploy `candidate.tar.gz`. Earlier token/test results below
> do not demonstrate real-model correctness. A production-host synthetic
> reproduction now confirms invented filters and semantic validation failures;
> see `artifacts/compact-transport/production-diagnosis/REPORT.md`.

The codec is an internal serialization boundary. Ollama and OpenRouter decode
into the same expanded mapping, then `SemanticFactPlanner` performs the existing
ontology expansion, `SemanticPlan` validation, semantic validation and execution.
No resolver, scope, cardinality, discourse, operator or renderer rules changed.
The codec's `decode_plan` convenience method uses that same canonical type.
Concept uses remain pre-expansion syntax; direct relation steps remain supported
by the canonical IR codec and remain forbidden where the deployed planner schema
already forbids them.

## Wire contract

The envelope is `[1, body]`. An object becomes an array containing its required
properties in **alphabetical** order. Optional properties, when present, occupy
one final object whose keys are compact aliases. Original arrays remain ordered
arrays; an object's positional array and a list have different schema positions.
Required slots cannot be omitted. There is no sentinel, delimiter escaping,
implicit null conversion or inferred operand. Null, false, zero, empty arrays,
missing optional fields and arbitrary literal strings retain their schema meaning.
Typed encoders may omit canonical defaults using Pydantic `exclude_defaults`.
Explicit optional fields in mapping inputs are preserved.

For example, the deployed schema encodes this synthetic request:

```json
{"requires_fact":true,"request":{"operation":"count","subject":{"kind":"current_household","path":[{"concept":"member"}]},"property":null,"property_source":"entity","filters":[{"predicate":"minor"}]}}
```

as:

```json
[1,[true,{"s":["count",null,"entity",["current_household",{"m":[["member"]]}],{"f":[["minor"]]}]}]]
```

Here the request slots are operation, property, property_source, subject; its
optional tail contains filters. Path steps and predicates each have one required
slot. Required layouts and the optional-key dictionary are generated for the
prompt. The constrained response schema uses JSON Schema `prefixItems`,
`minItems`, `maxItems`, enums, unions and closed optional-key objects.

Aliases are bijective base-26 identifiers allocated over sorted fields from
`SemanticPlan.model_json_schema()` plus `SemanticConceptUse.model_json_schema()`.
There is no manually maintained property/concept/intent vocabulary. Domain names
and operation enums stay literal: short field aliases alone measured worse in
local token tests, and domain symbol dictionaries add prompt overhead. The
schema/dictionary fingerprints accompany traces. The dictionary snapshot test
requires an intentional format-version review when canonical field allocation
changes. Slot rules, alias allocation and envelope changes require a version bump.
Payloads must be interpreted against their advertised schema; schema fingerprints
must match for replay. There is no expanded-JSON model-response fallback.

Capabilities use lossless `$table=[columns,rows]` nodes when sibling objects have
identical key sets and tabulation saves bytes. A row starts with its map key and
then contains column values. Reserved `$table`/`$literal` user keys are escaped as
`$literal` key/value pairs. Mapping keys and sets are sorted; lists, path steps,
interval endpoints and other ordered sequences are preserved. Compression does
not scope the ontology or remove unavailable-condition explanations.

## Validation and diagnostics

Decoding rejects incompatible versions, duplicate keys, trailing data, markdown
fences, nonfinite numbers, unknown aliases/symbols and schema violations. It
validates both the wire value and reconstructed expanded value. Canonical model
validators and deployment semantic checks remain authoritative downstream.
Object-schema constraints the transformer does not implement fail closed.

`encode(validate=False)` is only for encoding existing authored demonstrations
against catalogs where some demonstration vocabulary may be unavailable, and
for deliberate invalid-output tests. It does not repair fields and is never used
by the decoder. The synthetic complete-catalog profile validates all 37 examples.
The older authored gender examples omitted `operator=eq` despite the model schema
requiring it; this canonical default is now explicit in those demonstrations.
That is an evaluation/prompt-representation correction, not new filtering behavior.

Existing two-attempt retry behavior remains. Retry prose now asks for a structured
value, matching the array envelope. Invalid model IDs fail schema validation
before canonical validation, so their transport diagnostic is `MALFORMED_OUTPUT`;
the engine's existing ID rejection remains covered separately.

Transport diagnostics contain codec version, schema and field-dictionary hashes,
input/schema/output sizes, parse success and parse duration. Per-attempt usage
and transport metrics survive retries in `PlannerDiagnostics.transport.attempts`
and benchmark serialization. They contain no prompts, names, IDs or raw malformed
output. Existing expanded-plan diagnostics remain available for explicit debugging.
Provider token counts are actual usage; offline tokenizer counts are separately
labelled. No inferred latency or retry-rate improvement is reported.

Schema-derived codecs and synthetic demonstrations use bounded caches. Neither
user turns nor trusted identities are cached there. Expanded message building
without `output_schema` is retained for diagnostics/offline comparison only;
serving clients always supply the schema and require compact responses.
