Date: 2026-09-19 11:42 PDT
Type: architecture
Status: completed

## Objective

Map the backend by responsibility and define a safe sequence for reducing duplicate
code, runtime coupling, and the active deployment surface.

## Context

The package contained 20,523 Python lines across 74 modules. Vision was paused
pending MicroDuck, while the default API still imported and initialized camera code.

## Findings

The largest safe product-scope reduction is removing Vision from chat startup. The
next structural targets are the 1,500-line API module, duplicated streaming paths,
parallel provider adapters, and private mutation dispatch. The semantic core is
large but holds necessary deterministic boundaries.

## Decisions

Quarantine Vision, split HTTP by route family, establish one execution stream and a
provider-neutral model protocol, then separate tools/mutations before reorganizing
semantic modules.

## Changes

No runtime code was changed. This architecture work log was added.

## Validation

Inspected module definitions, imports, route ownership, tests, Docker composition,
runtime startup, and line counts. No behavioral tests were required for documentation
only changes.

## Remaining Issues

The proposed cleanup stages are not yet implemented.

## Recommended Next Step

Remove Vision imports, routes, settings, and startup state from the default chat
application as a bounded first refactor.

## Detailed Breakdown

# Home Cortex backend functional breakdown

Status: cleanup design baseline  
Updated: 2026-09-19  
Scope: `src/home_cortex` and the backend process started by `home_cortex.api:app`

## Purpose

This document maps the current backend by responsibility and gives a safe order for
reducing it. Line count is a signal, not the goal. The cleanup must preserve the
central rule of Home Cortex: the model interprets language, the ontology defines
meaning, and deterministic code resolves entities, reads the graph, computes facts,
and renders answers.

The immediate product is chat plus household graph access. Vision is paused until
MicroDuck arrives. Paused Vision code should not be initialized by, imported into,
or configured on the normal chat server.

## Baseline

The Python package currently contains 20,523 lines across 74 modules. Including the
Vision documentation and browser assets brings the source tree to roughly 22,064
lines. Importing `home_cortex.api` reaches 56 of the 74 Python modules.

| Functional area | Python lines | Files | Assessment |
| --- | ---: | ---: | --- |
| HTTP, bootstrap, sessions, configuration | 2,088 | 7 | Too centralized in `api.py` |
| Conversation and agent coordination | 1,608 | 4 | Necessary, with duplicate stream/run paths |
| Model providers | 949 | 2 | Provider adapters repeat the same interface |
| Semantic fact pipeline | 7,053 | 14 | Core product; complex but has clear layers |
| Tools and capabilities | 1,534 | 3 | Registry, schemas, and execution are mixed |
| Graph persistence and import/export | 1,396 | 5 | Cohesive, mostly maintenance-facing |
| Mutations | 1,066 | 3 | Necessary if household writes remain enabled |
| Agent definitions | 144 | 4 | Small and configuration-driven |
| Spatial | 1,711 | 8 | Partly used by graph validation; not all is Vision-only |
| Vision | 2,654 | 22 | Paused feature, but currently imported by the API |
| Shared display/package code | 320 | 2 | Shared presentation utilities |

The largest files are `api.py` (1,500), `semantic_schema.py` (914),
`vision/contracts.py` (837), `household_fact_engine.py` (828), `calendar.py`
(763), `writing.py` (697), `model_loop.py` (644),
`unified_semantic_planner.py` (636), `semantic_ontology.py` (628), and
`entity_resolver.py` (609).

## Runtime request flow

```text
Browser / OpenAI-compatible client
    -> FastAPI transport and authentication
    -> conversation ownership and persistence
    -> AgentService
        -> SemanticConversationService
            -> semantic planner (language -> typed request)
            -> HouseholdFactEngine
                -> EntityResolver
                -> ToolDispatcher / RetrievalService
                -> deterministic operators
            -> FactRenderer
        -> ModelLoop for ordinary conversation and non-graph tools
    -> SSE or JSON response
```

Startup currently does this in `api.lifespan`:

1. Load settings and connect SurrealDB.
2. Load edge schemas and the runtime schema catalog.
3. Create retrieval, writing, greeting, conversation, and calendar services.
4. Build a language-model client, tool dispatcher, and `AgentService` for each
   configured agent.
5. Create a Tapo camera hub even when Vision is unused.

The first four steps belong to the chat application. Step five does not.

## Functional ownership

### 1. HTTP application and composition

Current modules:

- `api.py` owns application construction, lifespan, middleware, exception mapping,
  Vision routes, GUI sessions, health and admin routes, conversation routes, agent
  routes, OpenAI-compatible routes, model discovery, authentication, identity
  resolution, SSE formatting, and response serialization.
- `config.py` defines environment-backed settings for the database, model provider,
  authentication, calendar, semantic data, and Vision.
- `db.py` wraps the SurrealDB client and request tracing.
- `gui_session.py` signs and validates browser-session cookies.
- `identity.py` maps trusted headers or GUI identity to graph person IDs.
- `request_tracing.py` records request stages and model usage.
- `text.py` contains small language and logging helpers.

Decision: keep the behavior but split the transport by route family. `api.py` should
become an application factory and composition root of about 200-300 lines. Routers
should receive services through a typed application container instead of repeatedly
reading loosely typed values from `app.state`.

Suggested modules:

```text
home_cortex/http/app.py              application factory and lifespan
home_cortex/http/dependencies.py     auth, identity, service lookup
home_cortex/http/errors.py           APIError and exception handlers
home_cortex/http/sse.py              OpenAI-compatible SSE encoding
home_cortex/http/routes/session.py
home_cortex/http/routes/health.py
home_cortex/http/routes/admin.py
home_cortex/http/routes/conversations.py
home_cortex/http/routes/chat.py
home_cortex/http/routes/models.py
```

Keep `home_cortex.api:app` as a compatibility shim until Docker and external callers
switch to the factory.

### 2. Conversation and agent coordination

Current modules:

- `conversations.py` stores conversation metadata and messages in memory or
  SurrealDB and enforces title/update behavior.
- `agent_service.py` builds the semantic and ordinary-chat paths, creates trusted
  request context, and chooses fact, mutation, or general model execution.
- `semantic_conversation.py` carries bounded discourse state between semantic turns.
- `model_loop.py` runs bounded non-graph tool calls and streams ordinary answers.
- `greetings.py` resolves household-aware greetings.

Decision: retain the boundary, but give `AgentService` one internal execution result
that can be consumed as a stream or collected as JSON. Today `answer`,
`answer_messages`, and streaming variants repeat routing logic. Likewise, the API
duplicates stream collection and persistence for conversation and OpenAI endpoints.

Target ownership:

- `AgentService.execute(...) -> AgentExecution` chooses the path once.
- `AgentExecution.tokens()` streams output.
- `AgentExecution.collect()` consumes the same stream for a non-stream response.
- A single conversation application service owns append, execute, persist, and
  title/update behavior.
- HTTP routers only validate transport input and serialize the result.

### 3. Model-provider boundary

Current modules:

- `ollama.py` contains Ollama transport plus semantic planner prompts/examples.
- `openrouter.py` reimplements ordinary chat, streaming chat, tool chat, mutation
  planning, semantic planning, unified planning, and close behavior for OpenRouter.

Decision: define one `LanguageModel` protocol and keep provider files limited to
wire-format adaptation. Move planner prompts and examples to a provider-neutral
`planning/prompts.py`. Move JSON decoding and planner result validation into the
planner layer. This removes duplicated method contracts and stops semantic code
from depending on provider details.

Do not force both providers through identical HTTP payloads. Share the behavioral
interface and response types; keep provider-specific request construction local.

### 4. Semantic fact pipeline

This is the core and should be reorganized carefully rather than deleted by size.

- `semantic_ir.py`: typed request, plan, context, evidence, result, timing, and
  planner outcome types. This is the dependency center and must remain independent.
- `semantic_contracts.py`: property type contracts shared by ontology and schema.
- `semantic_ontology.py`: loads and validates declarative semantic meaning.
- `schema_catalog.py`: maps deployed node/edge data into a runtime catalog.
- `semantic_schema.py`: exposes model capabilities, generates planner schemas,
  expands concepts, and validates plans. Its four jobs should become separate
  collaborators behind one public registry facade.
- `semantic_planner.py`: provider-neutral fact interpretation workflow.
- `unified_semantic_planner.py`: combined fact/mutation interpretation. It currently
  overlaps the fact planner and is active whenever mutation is enabled.
- `semantic_conversation.py`: bounded discourse context between turns.
- `entity_resolver.py`: grounds self, names, relations, concepts, and household scope.
- `household_fact_engine.py`: deterministic traversal, projection, filtering,
  predicates, and operators.
- `operator_registry.py`: validates and executes generic operators and predicates.
- `semantic_facts.py`: coordinates planner, executor, evidence checks, and renderer.
- `semantic_display.py`: builds language-neutral descriptions of semantic scope.
- `fact_renderer.py`: renders English and Chinese fact answers.

Decision: preserve the planner -> resolver/executor -> renderer direction. The first
cleanup should package these modules under `semantic/` without behavior changes.
Then split large classes by phase rather than by arbitrary line count:

```text
semantic/ir.py
semantic/ontology.py
semantic/schema/catalog.py
semantic/schema/capabilities.py
semantic/schema/validation.py
semantic/planning/facts.py
semantic/planning/unified.py
semantic/resolution/entities.py
semantic/execution/engine.py
semantic/execution/operators.py
semantic/rendering/description.py
semantic/rendering/text.py
semantic/service.py
semantic/conversation.py
```

The unified and fact planners should share prompt construction, decoding, validation,
diagnostics, and retry policy. They should remain separate strategies until benchmark
evidence proves one can replace the other.

`FactRenderer._en` and `FactRenderer._zh` are large parallel branches. Replace them
incrementally with a small message catalog keyed by semantic outcome. Keep semantic
description construction in `SemanticDisplay`; do not move meaning into translated
templates.

### 5. Tools and capabilities

Current modules:

- `tools.py` declares argument models, publishes model-facing definitions, dispatches
  calls, maps failures, invokes graph reads, calculations, calendar operations, and
  mutations, and also exposes internal semantic execution hooks.
- `calculate.py` safely evaluates a bounded arithmetic AST.
- `calendar.py` contains domain types, authorization, calendar service behavior,
  Google OAuth, Google HTTP transport, normalization, and settings construction.

Decision: replace the central conditional dispatcher with registered `Tool` objects
that each own their argument model, public schema, authorization, and execution.
Keep semantic graph reads on an internal interface; do not expose physical graph
tools to the model. Split Google transport from calendar policy:

```text
capabilities/registry.py
capabilities/calculation.py
capabilities/calendar/service.py
capabilities/calendar/google.py
capabilities/mutations.py
```

### 6. Graph persistence and maintenance

Current modules:

- `retrieval.py` owns graph reads and JSON-safe results.
- `edge_schema.py` loads physical edge contracts.
- `record_ids.py` canonicalizes SurrealDB record IDs.
- `ingestion.py` validates source files and synchronizes graph records.
- `export.py` emits deterministic canonical graph files.

Decision: keep retrieval in the runtime dependency graph. Move import/export behind
an `admin` package or maintenance command boundary so ordinary chat startup does not
need those modules. Admin HTTP endpoints can lazily import their service or be
enabled by an explicit setting.

### 7. Household mutations

Current modules:

- `mutation_ir.py` defines named write intents and planner schemas.
- `writing.py` validates and applies item creation, movement, attribute update, and
  deletion.
- `semantic_writing.py` resolves names for writes and renders mutation results.

Decision: retain this as a separate command side. `AgentService` currently calls the
private `ModelLoop._dispatch` method for planned writes; replace that dependency with
an explicit `MutationService`. Reads and writes should not meet inside the generic
model tool loop.

### 8. Agent definitions

Current modules and resources:

- `agents/registry.py` loads agent YAML, prompt, and allowed-tool policy.
- `agents/steward/config.yaml`, `prompt.md`, and `tools.py` define the single active
  steward.

Decision: keep. The registry is small and already configuration-driven. Avoid a
plugin framework until a second real agent requires one.

### 9. Spatial domain

Current modules:

- `spatial/contracts.py` validates spatial graph fields and location edges.
- `spatial/units.py`, `transforms.py`, and `pose.py` provide measurements and poses.
- `spatial/anchors.py`, `observation.py`, and `localize.py` cover fiducials and
  localization.

Decision: do not remove the whole package with Vision. Ingestion and semantic schema
use parts of the spatial model. Separate `contracts`, `units`, `transforms`, and
`pose` as the graph-facing spatial core. Treat anchors, observations, and
localization as optional robotics functionality and keep them out of chat startup.

### 10. Paused Vision feature

Current modules:

- `vision/contracts.py`, `ports.py`, and `artifacts.py` define the visual domain.
- `vision/edge/*` captures and serves frames from edge hardware.
- `vision/camera/*` implements camera source validation, codecs, Tapo transport, and
  shared stream handling.
- `vision/relay.py` proxies MJPEG/Tapo streams and signs Vision sessions.
- `vision/web/*` is the standalone page served by `api.py`.

Decision for the current product: quarantine it from the deployed chat application.

1. Remove Vision settings and environment variables from the default chat compose
   service.
2. Remove `SharedTapoHub` construction and shutdown from the chat lifespan.
3. Remove Vision imports and routes from the default HTTP application.
4. Keep the implementation in an optional package or a separate historical branch
   until MicroDuck defines the actual camera/robot boundary.
5. Keep generic spatial graph contracts that are independently used by ingestion.

This immediately removes camera, codec, relay, FFmpeg, and Tapo concerns from the
normal runtime. If the code is retained in this repository, a separate
`home_cortex_vision` optional package is clearer than a feature flag inside the chat
server. If there is no near-term need to maintain it, Git history is the archive and
the source can be deleted after its tests are separated.

## Confirmed duplication and coupling

### Duplicated session signing

`gui_session.py` and `vision/relay.py` both implement URL-safe base64 helpers,
HMAC-SHA256 tokens, expiry checking, parsing, and `valid_session`. If Vision remains,
extract a small signed-token primitive. If Vision is removed from active source,
leave the GUI implementation local and delete the duplicate with the feature.

### Duplicated provider behavior

`OllamaService` and `OpenRouterService` both expose ordinary chat, stream chat, tool
chat, semantic planning, unified planning, mutation planning, and shutdown. The
protocol is implicit, so callers depend on concrete methods and provider-specific
response shapes. A typed protocol and canonical message/tool-call result removes
that repetition from every caller.

### Duplicated streaming paths

The API separately implements streamed agent conversations, non-stream collection,
bare model streaming, OpenAI SSE formatting, persistence while streaming, and
greeting prepending. `ModelLoop` also has separate `run` and `stream` loops. Build
one token-stream execution path and collect it where JSON is required.

### Repeated route orchestration

Conversation ownership, identity resolution, model/agent lookup, message cleanup,
execution, assistant persistence, and serialization are repeated across custom
conversation routes and `/v1/chat/completions`. Move that workflow into an
application service and keep route-specific response formats at the edge.

### Schema registry doing four jobs

`SemanticSchemaRegistry` loads model capability payloads, generates output JSON
schema, expands concepts, and validates executable requests. Keep one facade for
callers, but split these implementations so planner transport changes do not touch
execution validation.

### Private cross-layer calls

`AgentService` invokes `ModelLoop._dispatch` for mutations. This bypasses the public
boundary and makes the generic model loop responsible for a planner-approved write.
Use a public mutation service instead.

### Runtime and maintenance code loaded together

`api.py` imports ingestion and export eagerly for two admin routes. The Docker image
also installs test dependencies as runtime dependencies. Move pytest packages to the
development extra and load maintenance services only for their command or router.

## Target deployed backend

The target chat process should have six visible layers:

```text
HTTP transport
    -> conversation application service
        -> agent coordinator
            -> semantic fact service
            -> ordinary model loop
            -> capability registry
        -> conversation repository
    -> provider-neutral language model interface
    -> SurrealDB adapters
```

Allowed dependency direction:

```text
http -> application -> domain services -> ports <- adapters
semantic planner -> semantic IR <- deterministic executor
renderer -> semantic result + ontology
```

Disallowed direction:

- Semantic IR importing database, tools, planners, renderers, or HTTP.
- Deterministic execution importing model providers or renderers.
- Provider adapters owning semantic validation.
- HTTP routes performing graph queries directly.
- The ordinary model loop receiving unrestricted household graph access.
- Paused robotics or camera modules importing into chat startup.

## Cleanup sequence

Each stage should be a separate reviewable change. Do not combine file movement with
semantic behavior changes.

### Stage 1: remove Vision from chat startup

- Extract or remove the four Vision routes from `api.py`.
- Stop constructing `SharedTapoHub` in the default lifespan.
- Remove Vision environment variables from the default compose file.
- Remove FFmpeg from the default backend image if no other feature uses it.
- Keep or move Vision tests with the optional package.

Acceptance:

- Importing the chat app loads no `home_cortex.vision` modules.
- The chat image contains no camera-specific runtime dependency.
- Session, health, models, conversations, streaming chat, ingestion, and export
  continue to behave as before.

### Stage 2: split the HTTP monolith

- Introduce an application factory and typed service container.
- Move routes by family without changing paths or payloads.
- Move auth, error mapping, and SSE helpers into transport modules.
- Keep `home_cortex.api:app` compatible.

Acceptance:

- `api.py` is a small compatibility/composition module.
- Existing API and OpenAI compatibility tests pass unchanged.
- No router constructs core services or reads environment variables.

### Stage 3: unify execution and streaming

- Introduce one `AgentExecution` stream.
- Collect the stream for non-stream responses.
- Centralize conversation append/execute/persist behavior.
- Preserve disconnect handling and partial-answer persistence policy explicitly.

Acceptance:

- Stream and non-stream answers are byte-for-byte equivalent after SSE decoding.
- Each request invokes semantic routing once.
- Assistant persistence happens once.

### Stage 4: make model providers adapters

- Add the `LanguageModel` protocol and canonical response types.
- Move prompts and decoding out of provider clients.
- Keep provider-specific metrics as optional metadata.

Acceptance:

- `AgentService`, planners, and `ModelLoop` do not import Ollama or OpenRouter
  concrete classes.
- The provider contract has shared conformance tests.

### Stage 5: separate tools and mutations

- Register capabilities instead of branching in one dispatcher.
- Split calendar policy from Google transport.
- Give planned mutations a public service boundary.

Acceptance:

- Allowed tools remain agent-configured.
- Household reads still go only through the semantic pipeline.
- Writes require typed planner intent and authenticated caller context.

### Stage 6: package the semantic core

- Move modules into `semantic/` with compatibility imports temporarily.
- Split schema capability generation from execution validation.
- Share common fact/unified planner mechanics.
- Refactor rendering through a message catalog only after snapshot coverage exists.

Acceptance:

- Existing composition, bilingual, held-out, and synthetic evaluations retain their
  scores and semantic equivalence rules.
- Architecture import guards pass.
- Physical storage fields and household identity mappings remain hidden from the
  model.

### Stage 7: trim packaging and maintenance surface

- Move pytest and pytest plugins out of runtime dependencies.
- Put ingestion/export behind admin commands or an optional router.
- Review public exports and remove compatibility shims after callers migrate.

Acceptance:

- The production image installs only runtime dependencies.
- Chat startup does not import maintenance modules.
- Import/export commands remain deterministic.

## Expected impact

Removing Vision from the default application isolates 2,654 Python lines plus about
1,470 lines of Vision documentation/browser assets from the chat product. It also
removes camera startup state, Tapo/codec imports, and the need for FFmpeg in the
default image. This is the largest safe product-scope reduction.

The remaining semantic core will still be substantial. Its 7,053 lines implement a
typed ontology, schema compilation, grounding, deterministic execution, evidence,
bilingual rendering, and planner validation. The goal there is clearer ownership and
less duplicated orchestration, not an arbitrary deletion target.

A reasonable end state is:

- default chat startup imports no Vision or robotics code;
- no production module exceeds roughly 500-600 lines without a documented reason;
- `api.py` is only a compatibility entry point;
- one execution stream serves every transport;
- model providers implement one explicit protocol;
- semantic types and executor boundaries remain protected by import tests.

## Verification for every cleanup change

Run the smallest relevant tests during development, then the deterministic suite:

```bash
UV_CACHE_DIR=/tmp/home-cortex-uv-cache uv run --with pytest pytest -q
```

Also verify:

- `python -c "import home_cortex.api"` does not load paused feature modules;
- GUI conversation creation, history, streaming, and persistence work through the
  proxy;
- both configured model providers pass the same adapter contract tests;
- Docker builds without Vision dependencies after Stage 1;
- composition fingerprints are updated only when an intentional ontology or
  evaluation input change requires it.

Do not use reduced line count as proof of success. Use preserved behavior, fewer
runtime imports, explicit dependencies, smaller ownership surfaces, and measured
startup/request performance.
