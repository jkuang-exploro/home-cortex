# Home Cortex API

This package provides a FastAPI RAG service over SurrealDB. The API calls
Ollama, dispatches the model's allowlisted Cortex tools, and returns the final
grounded answer. Household graph reads run only through the semantic fact
pipeline; model-facing tools provide local calculation and read-only Google
Calendar access.

## Endpoints

- `GET /health` checks SurrealDB and does not require a Cortex API key.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` and
  `/app/data/edges/*.json`. It validates all input before writing, then makes
  each JSON file authoritative for its table, including pruning records removed
  from that file. When `CORTEX_API_KEY` is set, this route requires that key.
- `POST /v1/chat` runs the default steward agent for backward compatibility.
- `POST /agent/steward/chat` invokes the named household steward directly.
- `POST /agent/steward/conversations` initializes a conversation and returns
  one deterministic, relationship-aware greeting.
- `GET /agent/steward/conversations/{id}` reloads that initialization record
  without generating another greeting.
- `GET /v1/models` advertises `老管家` to OpenAI-compatible clients.
- `POST /v1/chat/completions` provides an OpenAI-compatible chat endpoint backed
  by the agent loop. It supports ordinary JSON responses and token-streamed SSE
responses for clients such as Open WebUI.

## Authentication and identity

`GET /health` is public. When `CORTEX_API_KEY` is set, every other route
requires `Authorization: Bearer <key>`.

V1 uses one household API key. The key authenticates the client (typically
the Open WebUI server-side proxy); it does not identify a person. Person
identity comes only from `X-OpenWebUI-User-Id` / `X-OpenWebUI-User-Email`
through `CORTEX_IDENTITY_MAP`. Cortex never treats a client-supplied
`person:` record ID as identity.

Mapped Person records and the configured home are loaded by exact record ID.
Fuzzy entity search is not used for identity or authorization. A mapped ID
that does not exist fails closed as `identity_record_not_found`.

Conversation records are owner-only. A caller who knows another person's
conversation ID receives the same `conversation_not_found` response as for
an unknown ID.

Anyone holding the household API key can present any mapped Open WebUI user
header. Per-person credentials are out of scope for V1.

## Relationship-aware greetings

The steward selects greetings without an LLM call. Cortex combines the mapped
Person, localized `address_as`, the Person's `household_role` edge property,
the configured home, the agent's reception templates, and the requested
language. Supported V1 categories are `owner`, `minor_dependent`,
`adult_dependent`, `guest`, and `unknown`.

Add the role to a household relationship, not the Person node:

```json
{
  "from": "person:jian_kuang",
  "to": "address:fort_cerritos",
  "residence_type": "primary",
  "household_role": "owner"
}
```

Unknown, missing, unrecognized, or conflicting relationship roles use a
neutral greeting and never default to owner. Agent-specific templates and
optional person overrides live in the agent's `config.yaml`; switching Ollama
models does not change the greeting policy.

Create a Chinese steward conversation explicitly with:

```sh
curl -sS -X POST http://localhost:8001/agent/steward/conversations \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"language":"zh"}' | jq
```

For OpenAI-compatible clients, Cortex treats a request containing one user
message and no prior assistant message as the start of a conversation. It
deterministically prefixes the first answer with the resolved greeting.
Subsequent requests containing conversation history do not repeat it.

For `stream: true`, Cortex consumes Ollama's async response stream on every
agent step. Tool-selection responses stay internal; chunks from the final
answer are forwarded as OpenAI-compatible SSE events after the grounding gate
has observed the required successful tool evidence. If the client
disconnects, cancellation closes the active agent and Ollama streams and records
a privacy-safe `stream_cancelled` log.

## Edge schemas and graph truth

Relationship meaning is defined in `schemas/edge/*.yaml`; relationship facts
remain in human-editable `data/edges/*.json`. The initial registry defines:

- `spouse_of` as a symmetric temporal Person-to-Person relationship;
- `parent_of` as a directed, non-temporal Person-to-Person relationship with
  the derived inverse name `child_of`;
- `lives_in` as a directed temporal Person-to-Address relationship;
- `located_in` as a directed, non-temporal Item-to-Address-or-Space
  relationship;
- `hosted_by` as a directed, non-temporal Space-to-Item relationship with the
  derived inverse name `hosts_space`.

Store each fact once. Model an addressable home as an Address, its physical
house as an Item located at that Address, and its rooms as Spaces hosted by the
house Item. Do not add a reverse spouse edge or a `child_of.json` file.
Likewise, do not add `hosts_space.json`; inverse hosted-space traversal uses the
canonical `hosted_by` table. `get_relationships` consults the registry,
accepts `out`, `in`, or `both` directions, and excludes ended temporal edges
unless `include_ended` is true.

## Entity aliases and semantic ontology

Household identity comes from node data. A person can have multilingual full
names in `name` and additional stable names in `aliases`:

```json
{
  "id": "person:example",
  "name": ["Example Person", "示例人"],
  "aliases": ["Example", "小示"]
}
```

Alias lookup reads the runtime database, applies Unicode NFKC normalization,
case folding, whitespace folding, and basic punctuation normalization, and
returns every exact match. It never chooses the first record when an alias is
ambiguous. To add a nickname, edit only the person's node data and re-run
ingestion; application source changes are unnecessary.

Names whose meaning depends on household or speaker belong in scoped
`appellations`, not global `aliases`:

```json
{
  "appellations": [
    {
      "value": "Papa",
      "household_id": "address:example",
      "speaker_ids": ["person:child"]
    }
  ]
}
```

Every appellation must declare at least a household or speaker scope. The
resolver tries direct names/aliases first, then appellations matching the
trusted active-speaker and household context. Relational descriptions such as
“my son” still belong in the ontology and graph rather than this list.

Language-independent property mappings and composable kinship concepts live in
`schemas/semantic/ontology.yaml`. For example, `father_in_law` expands to
`spouse -> parent[gender=male]`, while the base `parent` relation points to the
declarative inverse name from `schemas/edge/parent_of.yaml`. Add a new kinship
term to the appropriate ontology `aliases` list, then rebuild/restart the API.
Do not add context-dependent phrases such as “my son” to a person's static
aliases.

The symbolic semantic reference `self` is resolved from the authenticated
speaker on each request. It is not tied to a default household member. The
ontology can therefore compile the same “my son” plan for different speakers,
and the resolver starts traversal from each request's active speaker.

The semantic fact IR is a bounded algebra backed by the explicit operator
registry. Collection filters compose with operations such as `count`, `argmin`,
and `argmax`; `adult` and `minor` are declarative ontology predicates rather
than fact handlers. Their policy prefers a recognized `household_role` on the
membership edge and falls back to `completed_years(birth_date)` using the one
`adulthood_years` value in `schemas/semantic/ontology.yaml`. A person's `child`
relationship remains distinct from a minor household member.

Properties can explicitly come from the final relationship edge. Semantic
`start_date`/`end_date` map to deployment edge fields in the ontology, allowing
the same spouse traversal to resolve either the partner entity or the
relationship start date. The executor validates all operations, filters,
predicate names, property sources, and types before it reads or computes data.

Tier 0 is a removable latency optimization. It recognizes only six exact,
canonical utterances covering speaker identity, assistant identity,
household-member count, and household-member list. Kinship, age extrema, adult/minor filters,
marriage, birthdays, and addresses are owned by the semantic planner. To
exercise the authoritative planner and entity resolver without Tier 0, set:

```dotenv
HOME_CORTEX_DISABLE_TIER0=1
```

The fact benchmark supports both execution modes and records speaker ID,
utterance, semantic plan, scope, filters, operators, relationship references,
entity/relationship properties, failure stage, canonical IDs, LLM/DB calls,
timing, and answer. `MODE` and `--mode` are equivalent; repeat `--question` to
run a focused Tier-1 suite:

```sh
MODE=tier0_enabled home-cortex-fact-benchmark --backend surrealdb
MODE=tier0_disabled home-cortex-fact-benchmark --backend surrealdb --repeat 1
home-cortex-fact-benchmark --backend surrealdb --mode tier0_disabled \
  --question 谁最年幼 --question 我们什么时候结婚的
```

The separate planner-quality benchmark loads more than 100 paraphrases from
`benchmarks/semantic_planner_eval.yaml`. It bypasses Tier 0, executes every valid
plan against the deterministic engine, compares normalized semantic meaning,
and reports accuracy by entity reference, traversal, multi-hop kinship,
property selection, filtering, aggregation, temporal operation,
relationship-property lookup, and speaker-relative reference:

```sh
home-cortex-semantic-planner-benchmark \
  --data-dir /app/data --schema-dir /app/schemas/edge
home-cortex-semantic-planner-benchmark --one-per-plan
```

All currently retained Tier-0 forms are included in that dataset and must remain
semantically equivalent to planner output. Before adding any new fast path,
first inspect the prompt, capability payload, strict output schema, examples,
model quantization, and runtime. A new path is justified only when the expression
is stable and unambiguous, materially frequent in measured traffic, the planner
still passes the same planner-only case, and the measured latency saving matters
to the deployment SLO. A category accuracy below 95% is a signal to improve the
planner path, not permission to add a phrase handler.

The six retained forms are intentionally narrow: `我是谁` / `Who am I?` and
`你是谁` / `Who are you?` are common session diagnostics with immutable
reference semantics; `家里有几个人` and `家里都有谁` are common dashboard
queries with a single declared `current_household -> member` plan. Variants and
paraphrases always go through the planner.

The ingestion endpoint rejects unknown relationship files, invalid endpoint
types, references to nodes missing from the source data, temporal fields on
non-temporal edges, reverse duplicates of a symmetric fact, and a registered
relationship without a corresponding JSON source file. An empty relationship
must be represented by `[]` so re-ingestion can prune previously stored facts.
Ingestion also clears records from explicitly retired relationship and node
tables during ontology migrations.

This version renames the former `resides_in` relationship to `lives_in`. Since
the repository intentionally does not track private household data, rename the
deployment's `data/edges/resides_in.json` to `lives_in.json` before ingesting.
Remove `start` and `end` from `parent_of.json`; `parent_of` is non-temporal in
the V1 schema. The old SurrealDB `resides_in` table is no longer queried, so it
cannot contribute facts after the application is redeployed.

The agent also applies a deterministic grounding gate. A structured LLM planner
decides whether a request depends on household world state and plans only
against entity fields and relations discovered from the runtime data/schema.
The bounded executor then validates every required field, relation, record
count, and optional freshness constraint before answer generation. Missing
entities, missing fields, incomplete evidence, and stale evidence produce
different fixed responses rather than model-authored facts. Caller-supplied
system messages are discarded before the trusted Cortex policy is applied.

## First run

From `docker/cortex` on the server:

Copy `.env.example` to `.env`, then set `SURREAL_PASS`, `OLLAMA_MODEL`,
`CORTEX_API_KEY`, and `CORTEX_IDENTITY_MAP` in that file. `OLLAMA_MODEL` is the
single deployment setting used by the Cortex API when selecting an Ollama
model. Map the email used to sign in to Open WebUI to Jian's graph record:

```dotenv
CORTEX_API_KEY=replace-with-a-long-random-secret
CORTEX_IDENTITY_MAP={"email:your-login@example.com":"person:jian_kuang"}
```

An Open WebUI user ID is a stronger mapping key when it is known:

```dotenv
CORTEX_IDENTITY_MAP={"id:open-webui-user-uuid":"person:jian_kuang"}
```

```sh
docker compose up -d --build
curl http://localhost:8001/health
curl -X POST http://localhost:8001/admin/ingest \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
curl -X POST http://localhost:8001/v1/chat \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Who lives at Fort Cerritos?"}'
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"model":"老管家","stream":false,"messages":[{"role":"user","content":"Where do I live?"}]}'
```

The base Compose file runs on macOS and CPU-only Docker hosts. On a Linux host
with an NVIDIA Container Toolkit installation, opt into GPU access explicitly:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Open the interactive API documentation at
`http://192.168.68.59:8001/docs`.

## Open WebUI

The Compose stack configures two model paths:

- Direct Ollama at `http://ollama:11434` for model debugging.
- The OpenAI-compatible Cortex API at `http://cortex-api:8000/v1` for grounded
  graph answers.

Select `老管家` in Open WebUI to use the steward and its SurrealDB tools.
The agent refers to itself as `the butler` in English and `老管家` in Chinese.
Selecting a raw Ollama model bypasses Cortex. Open WebUI persists its
connection settings, so an existing deployment may require adding the Cortex
connection once in the administrator connection settings with API key
matching `CORTEX_API_KEY`.

The Compose deployment builds the pinned Open WebUI v0.9.5 source with the
small patch in `docker/cortex/open-webui`. When `老管家` is selected in a blank
new chat, the UI creates a steward conversation and persists Cortex's
relationship-aware greeting as the first assistant message—before the user
sends anything. The browser calls an authenticated Open WebUI proxy; only that
server-side proxy receives `CORTEX_API_KEY` and forwards the verified Open
WebUI user ID and email to Cortex.

The proactive greeting language is controlled by
`CORTEX_GREETING_LANGUAGE`; Compose sets it to `zh`. This affects only the
initial greeting, not the language used for later answers.

Compose enables Open WebUI's authenticated user-info forwarding. Cortex maps
the forwarded user ID or email to a stable `person:` record, resolves it before
the first model call, and supplies only its `id`, `name`, and `address_as` as
trusted context. Other private fields still require an intentional semantic fact lookup.
User-written messages cannot change this mapping. If identity mappings are
configured, an unknown Open WebUI user receives an `identity_not_mapped` error
instead of being treated as somebody else.

## Agent runtime architecture

The runtime is intentionally split into focused layers:

- `agent_service.py` is the public coordinator. It normalizes trusted identity and
  conversation input, then selects the appropriate execution path.
- `schema_catalog.py` discovers queryable node properties, edge properties, and
  relationship semantics from the deployed data and edge registry. Adding a
  field to a node JSON file makes it available to planning without changing
  factual-grounding code.
- `semantic_facts.py` owns trusted request context, strict semantic plans, and
  bounded entity resolution, traversal, filtering, sorting, aggregation,
  date/duration, unit conversion, and freshness operations. Its shared evidence
  gate validates every required field and relation before the deterministic
  renderer sees a value.
- `model_loop.py` owns the bounded Ollama loop, tool limits, display-name repair,
  and streaming for ordinary conversation and non-graph tools. Graph tools are
  deliberately not exposed through this path, so household facts cannot bypass
  schema-aware planning and evidence validation.

The factual domain is determined by deployed schema and data. The LLM interprets
meaning but never executes SQL or Python. After the deterministic gate establishes
that every declared requirement is present and fresh enough, a deterministic
renderer formats the validated value. Household evidence is not sent through a
second model call, so missing values cannot be replaced with model knowledge.

## Named agents

Home Cortex is the shared platform; named agents are role-specific interfaces
on top of it. The `steward` definition lives under
`home_cortex/agents/steward` and owns its display name, prompt, model preference,
settings, and tool allowlist. The public coordinator remains in
`agent_service.py`, and the generic Ollama loop lives in `model_loop.py`.

The steward's model is `OLLAMA_MODEL` when its `config.yaml` model name is null.
A future specialized agent can select a different model and tools without
changing the shared runtime. For example, a future `accountant` directory can
define `账房` and finance-only tools; those tools will not be granted to
`steward` automatically.

## Shared Cortex tools

Tools are registered centrally in `home_cortex.tools` and granted per agent
through that agent's `ALLOWED_TOOLS`. Graph lookup primitives are internal to
the semantic fact executor. The ordinary model loop receives only `calculate`,
`calendar.list_events`, and `calendar.check_availability`.

`calculate` evaluates arithmetic with an allowlisted AST parser. It does not
use Python `eval()`, has no network dependency, and returns a structured
numeric result such as `{"result": 14}`.

Calendar Phase 1 is read-only. Google Calendar is the source of truth; events
are not copied into SurrealDB. Bind household calendars with OAuth settings
and `CALENDAR_BINDINGS`. Each binding has a Cortex calendar ID, the owning
person, the Google calendar ID, and optional extra `readers`. A caller cannot
read another person's calendar merely by supplying that person or calendar ID.

```dotenv
GOOGLE_CALENDAR_CLIENT_ID=...
GOOGLE_CALENDAR_CLIENT_SECRET=...
GOOGLE_CALENDAR_REFRESH_TOKEN=...
CALENDAR_TIMEZONE=America/Los_Angeles
CALENDAR_BINDINGS=[{"id":"jian_primary","person_id":"person:jian_kuang","provider_calendar_id":"primary","readers":[]}]
```

Do not place Google credentials or access tokens in prompts or tool results.
Unauthorized or unconfigured calendars fail closed with a structured error
instead of crashing the agent loop. Event creation, updates, and deletion are
out of scope for Phase 1.

## Human-facing entity names

Tool calls and graph traversal retain stable IDs such as
`address:fort_cerritos`. Before a final answer reaches the user, the shared
display-name resolver replaces known IDs with stored names appropriate to the
conversation language. It supports both localized name objects and the current
ordered alias lists. Explicit requests for internal IDs and debugging details
leave IDs visible. This presentation step does not alter graph records, edges,
or tool-call arguments and is reusable by future agents.

Person records may optionally define a localized `address_as` object. It is a
presentation preference—not a name alias or relationship—and is stored
explicitly rather than inferred from age, gender, or household role:

```json
{
  "id": "person:jian_kuang",
  "name": {"en": "Jian Kuang", "zh": "匡健"},
  "address_as": {"en": "Mr. Kuang", "zh": "先生"}
}
```

The shared resolver exposes explicit `address`, `name`, and `id` modes. Normal
person-ID rendering prefers `address_as`, falls back to `name`, and uses the ID
only when no human-facing value exists. Existing records without `address_as`
remain valid.

## Observability

Every HTTP response includes a server-generated `X-Request-ID`. Error responses
use the same JSON envelope for validation errors, API errors, and unexpected
failures:

```json
{
  "error": {
    "code": "internal_server_error",
    "message": "An unexpected server error occurred",
    "request_id": "96f149cf430442d48fb6010899cde986"
  }
}
```

Grounding logs record whether grounding was required, the schema-level subject
type, requested field and relation names, deterministic operator, and evidence
status. The ordinary agent loop logs each model step, tool name, success status,
record count, execution time, and final stop reason. Stop reasons are `answer`,
`step_limit`, `tool_error`, or `timeout`. Logs intentionally omit prompts, tool
arguments, tool results, entity references, and private record values. View them
with:

```sh
docker compose logs -f cortex-api
```
