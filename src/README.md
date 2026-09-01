# Home Cortex API

This package provides a FastAPI RAG service over SurrealDB. The API calls
Ollama, dispatches the model's allowlisted Cortex tools, and returns the final
grounded answer. Shared tools include household-graph lookup, local
`calculate`, and read-only Google Calendar access.

## Endpoints

- `GET /health` checks SurrealDB and does not require a Cortex API key.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` and
  `/app/data/edges/*.json`. It validates all input before writing, then makes
  each JSON file authoritative for its table, including pruning records removed
  from that file. When `CORTEX_API_KEY` is set, this route requires that key.
- `POST /v1/retrieve` returns the graph context used for a question. When
  `CORTEX_API_KEY` is set, this route requires that key. Retrieval is
  household-scoped; per-person graph authorization is not implemented yet.
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

The agent also applies a deterministic grounding gate. Household answers
cannot complete without a successful tool call in the current turn; birthday
questions require `get_entity` data containing `dob`, and relationship
questions require `get_relationships`. Empty results produce a fixed
no-matching-data response rather than model-authored facts. Caller-supplied
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
curl -X POST http://localhost:8001/v1/retrieve \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Fort Cerritos"}'
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
trusted context. Other private fields still require an intentional tool lookup.
User-written messages cannot change this mapping. If identity mappings are
configured, an unknown Open WebUI user receives an `identity_not_mapped` error
instead of being treated as somebody else.

## Agent runtime architecture

The runtime is intentionally split into four flat layers:

- `agent_service.py` is the public coordinator. It normalizes trusted identity and
  conversation input, then selects the appropriate execution path.
- `request_analysis.py` converts supported household-fact language into a small
  `FactRequest` (`subject`, `field`, and `cardinality`) and derives the privacy
  and ordered graph-evidence requirements shared by both execution paths. A
  relationship registry supplies graph paths such as spouse, child, parent, and
  parent-in-law. Elliptical follow-ups inherit the last explicit structured
  subject, so a request such as `他们的生日分别是哪天？` can reuse the people
  established by the preceding turn.
- `facts.py` traverses the analyzed graph paths and renders verified structured
  facts deterministically.
- `model_loop.py` owns the bounded Ollama loop, tool limits, evidence gate,
  privacy filtering, display-name repair, and streaming. It handles informal or
  open-ended conversation and has no household-specific answer renderer.

This keeps natural-language aliases at the semantic boundary rather than
adding a new `if question == ...` method for each phrasing. Verified structured
facts do not pass through Ollama again, so the model cannot alter a birthday,
relationship, count, or anniversary after retrieval.

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
through that agent's `ALLOWED_TOOLS`. The steward currently receives graph
lookup, `calculate`, `calendar.list_events`, and `calendar.check_availability`.

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

The agent logs each model step, tool name, success status, record count,
execution time, and final stop reason. Stop reasons are `answer`, `step_limit`,
`tool_error`, or `timeout`. Logs intentionally omit prompts, tool arguments,
tool results, and private record fields. View them with:

```sh
docker compose logs -f cortex-api
```
