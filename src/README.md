# Home Cortex backend

Home Cortex is a local FastAPI service that answers household questions from a
SurrealDB graph. A language model interprets an utterance into typed semantic
meaning; deterministic code resolves entities, reads evidence, computes the
answer, and renders it. The model never reads physical storage or computes a
household fact.

The repository also contains `home_gui`, a browser client. Device-side capture,
media, and robot integration belong to the sibling `home_cortex_client` project.
The backend never imports that client.

## Runtime architecture

```text
browser / OpenAI-compatible client
    -> HTTP routes and authentication
    -> conversation execution and persistence
    -> AgentService
        -> semantic planner -> deterministic fact engine -> renderer
        -> MutationService for typed household writes
        -> ModelLoop for ordinary chat and non-graph tools
    -> SurrealDB and configured model provider
```

The important ownership rules are:

- `semantic/` owns semantic IR, ontology, schema binding, planning, and
  interpretation policy. The declarative vocabulary remains in
  `schemas/semantic/ontology.yaml`.
- `facts/` owns reference grounding, deterministic fact execution, operators,
  and rendering. It has no HTTP or provider-transport responsibility.
- `providers/base.py` is the provider-neutral contract;
  `providers/ollama.py` and `providers/openrouter.py` are transport adapters.
- `mutation/service.py` is the preview/commit application boundary;
  `mutation/writing.py` owns deterministic transactional graph mutations.
- `capabilities/catalog.py` owns model-facing schemas and
  `capabilities/dispatcher.py` validates and dispatches the configured tools.
  Household graph primitives stay internal to fact execution.
- `runtime/` coordinates agents and the ordinary model/tool loop;
  `conversation/` owns transcripts, browser sessions, and greetings.
- `persistence/` owns SurrealDB access, graph retrieval, edge/schema catalogs,
  ingestion, and export.
- `api/` owns transport; its package initializer preserves the deployment entry
  point `home_cortex.api:app`.

The semantic layers must not inspect utterance text after planning to repair a
request. Property ownership, predicates, pairwise operands, relationship paths,
and contextual scope stay explicit through execution.

## Spatial and Vision boundaries

`home_cortex.spatial` retains graph-level spatial contracts and pure math:
coordinate bases, units, poses, transforms, observation parsing, and deterministic
localization solving. Its package initializer is intentionally empty so chat
startup does not load optional localization modules. Live camera, SLAM, pose
acquisition, fiducial loops, and robot control belong in `home_cortex_client`.

Vision execution is paused pending MicroDuck hardware. The backend keeps only
transport-neutral visual evidence contracts, ingestion ports, enrollment
contracts, and artifact metadata. It has no camera routes, YOLO/OpenCV runtime,
FFmpeg relay, or Tapo integration. See
[the Vision boundary](home_cortex/vision/README.md).

## Package map

```text
home_cortex/
  api/             HTTP application, schemas, SSE, dependencies, and routes
  agents/          named agent definitions and registration
  capabilities/    model-facing calculation/calendar schemas and dispatch
  common/          identity, display, text, and tracing primitives
  conversation/    transcript persistence, GUI sessions, and greetings
  facts/           grounding, deterministic execution, operators, rendering
  mutation/        mutation IR, semantic intent, service, transactional writes
  persistence/     database, retrieval, schemas, ingestion, and export
  providers/       provider protocol and Ollama/OpenRouter adapters
  runtime/         application coordination and ordinary model loop
  semantic/        semantic IR, ontology, schema, prompts, and planners
  spatial/         spatial contracts, units, transforms, and localization math
  vision/          paused visual-evidence contracts and ingestion policy
  config.py        process configuration
```

Package initializers are intentionally small. Import concrete owners directly;
the two deliberate public surfaces are `home_cortex.api` for deployment/client
integration and `home_cortex.agents` for the named-agent registry.

## Where does new code go?

- New HTTP routes and transport policy go in `api/` and `api/routes/`.
- New semantic meaning or planner behavior goes in `semantic/` and, when it is
  reusable vocabulary, `schemas/semantic/ontology.yaml`.
- New deterministic household operations, grounding, or rendering go in `facts/`.
- New write intent or graph mutation behavior goes in `mutation/`.
- New provider adapters go in `providers/`; generic app coordination goes in
  `runtime/`.
- New storage adapters and graph maintenance behavior go in `persistence/`.
- Pure server-useful geometry stays in `spatial/`; localization-specific math
  goes in `spatial/localization/`.
- Camera capture, media, device polling, robot control, and other edge runtime
  belong in `home_cortex_client`, not this backend.

## HTTP surface

- `GET /health` checks SurrealDB and is public.
- `POST /session` exchanges the household API key and mapped identity for an
  HttpOnly GUI session.
- `POST /v1/chat` invokes the default steward.
- `POST /agent/steward/chat` invokes the named steward.
- `POST /v1/chat/completions` provides OpenAI-compatible JSON and SSE responses.
- `GET /v1/models` lists agents and configured bare language models.
- `POST /conversations` creates a persistent transcript.
- `GET /conversations/{id}` and related routes read or update an owned transcript.
- `POST /conversations/{id}/messages` executes and persists one turn; streaming
  and non-stream responses consume the same answer stream.
- `POST /admin/ingest` validates and synchronizes canonical graph files.
- `POST /admin/export` writes a canonical graph snapshot to an absolute server
  path.

Ingestion and export are imported lazily by their maintenance routes. Importing
the chat application does not load maintenance, Vision, or robotics-localization
modules.

Every response includes `X-Request-ID`. API, validation, and unexpected failures
use a stable error envelope:

```json
{
  "error": {
    "code": "internal_server_error",
    "message": "An unexpected server error occurred",
    "request_id": "..."
  }
}
```

## Authentication and identity

When `CORTEX_API_KEY` is configured, all routes except `/health` require
`Authorization: Bearer <key>`. The key authenticates a client; it does not identify
a person. Trusted `X-OpenWebUI-User-Id` or `X-OpenWebUI-User-Email` values are
mapped through `CORTEX_IDENTITY_MAP`, or captured in the GUI session cookie.

The server never accepts a caller-supplied `person:` record as proof of identity.
Mapped people and the configured household are loaded by exact record ID and fail
closed if missing. Conversations are owner-only; an unauthorized ID is
indistinguishable from an unknown conversation.

## Data and schemas

Canonical graph input lives under `data/nodes` and `data/edges` at runtime. That
data is not source code or a test fixture. Relationship meaning is declared in
`schemas/edge/*.yaml`. Semantic vocabulary, property types, ownership, predicates,
and reusable concepts are declared in `schemas/semantic/ontology.yaml`.

Store each fact once. Relationships such as `spouse_of`, `parent_of`, `lives_in`,
`located_in`, and `hosted_by` are traversed according to their edge schemas;
derived inverse names do not require duplicate edge files. Spatial pose on
`located_in` remains optional.

Generated benchmark output lives under `artifacts/`. Read summary JSON and reports,
not large per-case output. Benchmark inputs under `benchmarks/` have stable paths
and must not be copied into model prompt examples.

## Model providers and agents

`LLM_PROVIDER` selects `ollama` (default) or `openrouter`. The steward definition
under `home_cortex/agents/steward` owns its prompt, identity, settings, and tool
allowlist. A named agent receives only the capabilities declared there.

The ordinary model loop can use bounded calculation and read-only calendar tools.
Household fact retrieval cannot bypass semantic planning and evidence validation.
Writes require a typed current-turn mutation intent, authenticated context, and the
dedicated mutation service; a native model tool call cannot commit a write.

Google Calendar remains an external read-only source of truth. Bind calendars with
`CALENDAR_BINDINGS`; authorization is checked against the authenticated household
person. Credentials and provider tokens never enter prompts or tool results.

## First run

At the repository root, create `.env` and set at least:

```dotenv
SURREAL_PASS=replace-me
CORTEX_API_KEY=replace-with-a-long-random-secret
CORTEX_IDENTITY_MAP={"email:your-login@example.com":"person:jian_kuang"}
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
```

For OpenRouter, set `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, and
`OPENROUTER_MODEL` instead.

```sh
docker compose --env-file .env -f docker/docker-compose.yml up -d --build
curl http://home-cortex-0/health
curl -X POST http://home-cortex-0/admin/ingest \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
curl -X POST http://home-cortex-0/v1/chat \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'X-OpenWebUI-User-Email: your-login@example.com' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Who lives here?"}'
```

The base Compose file supports macOS and CPU-only hosts. On Linux with NVIDIA
Container Toolkit, also pass `-f docker/docker-compose.gpu.yml`. Interactive API
documentation is served at `/docs`.

The GUI is available at `http://home-cortex-0/` through nginx on standard HTTP
port 80. The GUI container is not published directly. The API's port 8001 binding
is restricted to host loopback for local maintenance and development. For local
frontend development, run `npm install && npm run dev` in `src/home_gui`; Vite
proxies API paths to that loopback-only development endpoint.

## Development

Install the development extra, then run the deterministic suite:

```sh
uv sync --extra dev
python -m pytest -q
```

Useful focused suites include:

```sh
python -m pytest -q tests/test_api.py tests/test_open_webui_conversations.py
python -m pytest -q tests/test_agent_service.py tests/test_tools.py tests/test_writing.py
python -m pytest -q tests/test_ollama.py tests/test_openrouter.py
python -m pytest -q tests/test_spatial.py tests/test_spatial_transforms.py
python -m pytest -q tests/test_semantic_composition.py tests/test_semantic_facts.py
```

Real-model accuracy runs belong on the production GPU host from an isolated,
fingerprinted package. Do not treat deterministic replay as model accuracy or
fabricate benchmark numbers. Compare models with `hc-bench`
(`benchmarks/HARNESS.md`). Other benchmark entry points and data are documented
in `scripts/README.md` and the relevant `artifacts/*/REPORT.md`.

Set `CORTEX_PROFILE_REQUESTS=1` to emit bounded request-stage and provider timing
without prompts, results, identities, or graph values. `X-Request-ID` connects HTTP
responses to privacy-safe server logs.

Before changing a core boundary, read `AGENTS.md` and recent related `.llm/` work
logs, then verify their assumptions against current source. Every meaningful coding,
debugging, refactoring, benchmark, architecture, or review cycle ends with a concise
`.llm/` entry following the repository work-log skill.

## Current deliberate non-goals

- No direct LLM access to SurrealDB, SQL, or physical field names.
- No Tier-0 factual shortcut around the semantic interpreter and executor.
- No implicit repair that drops predicates or changes property ownership.
- No camera, media relay, robot-control, or device-heartbeat runtime in the backend.
- No browser dependency on backend Python internals.

Historical architecture decisions and evaluation details are retained in `docs/`,
`.llm/`, and `artifacts/`; this README describes the active system.
