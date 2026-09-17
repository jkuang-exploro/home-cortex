# Home Cortex API

This package provides a FastAPI RAG service over SurrealDB. The API calls
Ollama, dispatches the model's allowlisted Cortex tools, and returns the final
grounded answer. Household graph reads run only through the semantic fact
pipeline; model-facing tools provide local calculation and read-only Google
Calendar access.

## Faithful semantic answer descriptions

`FactRenderer` uses the executed semantic request and result, with the engine's
ontology, to describe scope and conditions. `semantic_display.py` composes every
filter, predicate, intermediate traversal, exclusion and comparison operand;
counts and empty results retain the same description. Relationship conditions
distinguish independently matching associated edges from a projected single edge.
Rendering neither reads the utterance nor changes execution values.

Ontology properties, predicates and reference concepts accept optional localized
`label` maps. Properties also accept presentation-only `value_labels`, for example
`female: {en: female, zh: 女性}`. These labels do not define a closed value domain
or normalize input values. Missing labels fall back to English, then the semantic
key or literal. No new sentence template is needed for an additional filter.
The model-facing capabilities and output schema omit this display metadata.

`schemas/semantic/ontology.yaml` is the single canonical contract, declared as
`version: 2`. Every property declares its `type`, its `applies_to` owners and its
`filter_operators`; a closed domain declares `values` instead of `value_labels`.
The registry generates typed filter constraints from those declarations and
validates declared types, domains, ownership and stored inputs, so an undeclared
catalog field is not addressable as a semantic property.
See the [contract implementation and evaluation handoff](../artifacts/generic-contracts/REPORT.md).
Deploy the loader with the
labeled ontology; older loaders reject the typed fields. See the
[before/after report](../artifacts/condition-rendering/REPORT.md)
for validation and presentation limitations.

## Vision live preview

`GET /vision` is a build-free page hosted by this API, independent of the chat GUI.
The live preview uses a fixed server-side MJPEG relay:

```text
Browser -> home-cortex-0:8001/vision/stream -> 192.168.68.65:8088/live.mjpg
```

The browser never connects to the Mac directly. No CORS or browser private-network
permission is needed for the camera connection. The API streams chunks without
buffering the full response or decoding video. Recent Observations, Selected
Observation, Evidence Clip, and Label / Enroll remain placeholders; no detector,
ingestion, or household writes are added.

### Launch on the Mac

Stop the previous loopback-only streamer with Ctrl-C, then run:

```sh
uv run --extra vision python -m home_cortex.vision.edge --source mac
```

The CLI defaults to `0.0.0.0:8088` so the Cortex server can connect over the LAN.
Use `--host 127.0.0.1` for local-only access. The address `0.0.0.0` is a bind
address; configure the Mac’s actual LAN IP on the server. Allow the process
through the Mac firewall if prompted. This
existing development streamer is unauthenticated on the LAN; keep it on a trusted
network. Use `--source synthetic` for a hardware-free test.

### Configure and launch on home-cortex-0

Set this in `docker/cortex/.env` (update it if the Mac's address changes):

```dotenv
VISION_STREAM_URL=http://192.168.68.65:8088/live.mjpg
```

From the updated repository root:

```sh
docker compose --env-file docker/cortex/.env -f docker/cortex/docker-compose.yml up -d --build --no-deps cortex-api
```

This passes the configured URL into the API container. `VISION_STREAM_URL` is
optional; an unset value gives a clear 503 response. The relay accepts only that
server-configured HTTP(S) source, never a browser-supplied target, credentials in
the source URL, or upstream redirects. Existing API startup dependencies remain.
For a local API process, set the same environment variable and launch with
`uv run uvicorn home_cortex.api:app --host 0.0.0.0 --port 8001`.

Open <http://home-cortex-0:8001/vision> from any device that can reach Home Cortex.
If `CORTEX_API_KEY` is configured, enter it in the password field and click
**Connect**. It is sent once in an Authorization header to `/vision/session`,
cleared from the input, and replaced by a signed, one-hour HttpOnly, SameSite
cookie scoped to `/vision` (Secure on HTTPS). The key is never placed in a URL,
HTML response, or browser storage. Use HTTPS for access outside a trusted LAN.
Existing valid sessions can reconnect without re-entering the key. The Vision
cookie does not authorize other APIs; normal bearer authorization still works.
The public page and script contain no media; `/vision/stream` requires API auth
or a valid Vision session when a key is configured. No-key development remains
supported. Future enrollment still requires a trusted mapped human identity.

**Disconnect** releases the viewer's relay connection; Ctrl-C on the Mac stops
capture. Each viewer gets its own upstream connection, with a 3-second connect
and 10-second idle read timeout. Initial connection/type failures return 502;
midstream failures close playback. A browser may retain its last decoded frame
when the source stops; reconnect if the picture freezes.

### Verify from the server

Run on `home-cortex-0` first, to check Mac reachability:

```sh
curl --connect-timeout 3 --max-time 5 http://192.168.68.65:8088/health
```

Check from inside the API container too:

```sh
docker compose --env-file docker/cortex/.env -f docker/cortex/docker-compose.yml exec cortex-api python -c 'import os, urllib.request; r=urllib.request.urlopen(os.environ["VISION_STREAM_URL"], timeout=5); print(r.status, r.headers["Content-Type"]); print(r.read(100))'
```

Then verify actual JPEG bytes through the relay (export the configured key in
this shell; omit the header for no-key development):

```sh
curl --max-time 3 -D /tmp/vision-headers.txt -H "Authorization: Bearer ${CORTEX_API_KEY}" http://home-cortex-0:8001/vision/stream -o /tmp/vision-preview.mjpg
```

A timeout after 3 seconds is expected for an endless live stream. The headers
should show HTTP 200 and `multipart/x-mixed-replace; boundary=...`, and the output
file should contain JPEG frames. A 401 means the key is missing/wrong; 503 means
configuration is missing; 502 means the server cannot receive a valid stream.
Ensure the URL ends in `/live.mjpg` without a trailing comma.

Tests: `python -m pytest -q` includes an actual HTTP integration test from a
synthetic camera through the Cortex server, plus auth and upstream failure tests.
Run `node --test tests/vision_player.test.cjs` for player lifecycle checks.
Follow-up observation/clip/enrollment contracts are in
[Vision frontend architecture](home_cortex/vision/FRONTEND.md).

## Endpoints

- `GET /health` checks SurrealDB and does not require a Cortex API key.
- `POST /admin/ingest` imports `/app/data/nodes/*.json` (or shards under
  `/app/data/nodes/<table>/*.json`) and `/app/data/edges/*.json`. It validates
  all input before writing, then makes each table authoritative, including
  pruning records removed from that table's source files. When `CORTEX_API_KEY`
  is set, this route requires that key.
- `POST /admin/export` writes the current SurrealDB graph to an explicit
  absolute `target_dir` as canonical `nodes/` and `edges/` JSON. The path is
  on the API server, not the curl client. From Docker Compose use
  `/app/export`, which is mounted to `tmp/db-export` on the host. It does not
  default to `/app/data`. The response includes the resolved `target_dir`.
  The same snapshot is available from the CLI as
  `home-cortex-db-export ./tmp/db-export`. When `CORTEX_API_KEY` is set, this
  route requires that key.
- `POST /v1/chat` runs the default steward agent for backward compatibility.
- `POST /agent/steward/chat` invokes the named household steward directly.
- `POST /agent/steward/conversations` initializes a conversation and returns
  one deterministic, relationship-aware greeting.
- `GET /agent/steward/conversations/{id}` reloads that initialization record
  without generating another greeting.
- `GET /v1/models` advertises household agents (`老管家`) and bare language models
  from the configured provider.
- `POST /conversations` creates a transcript. Steward chats store a greeting as
  the first assistant message; bare-model chats do not.
- `POST /conversations/{id}/messages` appends a user turn, runs the steward or
  the bare LLM, persists the assistant reply, and can stream OpenAI-compatible SSE.
- `POST /session` exchanges the household API key and a mapped email for an
  HttpOnly GUI cookie. Curl can still send `Authorization` plus identity headers.
- `POST /v1/chat/completions` provides an OpenAI-compatible chat endpoint backed
  by the agent loop. It supports ordinary JSON responses and token-streamed SSE.

## Authentication and identity

`GET /health` is public. When `CORTEX_API_KEY` is set, every other route
requires `Authorization: Bearer <key>`.

V1 uses one household API key. The key authenticates the client; it does not
identify a person. Person identity comes from `X-OpenWebUI-User-Id` /
`X-OpenWebUI-User-Email`, or from a GUI session that stores the same map keys,
through `CORTEX_IDENTITY_MAP`. Cortex never treats a client-supplied
`person:` record ID as identity. The GUI cookie is not a person record.

Mapped Person records and the configured home are loaded by exact record ID.
Fuzzy entity search is not used for identity or authorization. A mapped ID
that does not exist fails closed as `identity_record_not_found`.

Conversation records are owner-only. A caller who knows another person's
conversation ID receives the same `conversation_not_found` response as for
an unknown ID.

Anyone holding the household API key can present any mapped user header or
create a session for a mapped email. Per-person passwords are out of scope for V1.

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
  relationship. Optional SI pose (`position` in meters, `orientation` in
  radians) is interpreted in the target space's intrinsic coordinates and is
  invalid when the target is an Address. Coordinate-free edges remain valid.
  Ingestion also accepts Space-to-Space `located_in` as physical placement;
  the edge schema's `from_types` stay item-only so semantic `contents` remain
  item-typed. `hosted_by` remains structural containment;
- `hosted_by` as a directed, non-temporal Space-to-Item-or-Space relationship with the
  derived inverse name `hosts_space`.

A Space may optionally declare intrinsic Cartesian metadata. Canonical
`coordinate` fields are `unit`, a 3-vector `basis`, and optional embedded
`anchors`. Local coordinates times the basis yield physical SI displacement;
basis magnitude may encode scale. Geometry, `origin`, and a surveyed zero
marker are not required. `navigable` and `accessible` remain explicit local
booleans. Validation lives in `home_cortex.spatial`; this is not a planner
property and does not introduce localization or ROS. Structured metric/imperial
conversion and nested pose composition are pure functions in `spatial.units`
and `spatial.transforms` (z-up, intrinsic yaw-pitch-roll). The LLM never
converts measurements. Runtime robot pose (`spatial.pose`) is ephemeral and is
not stored on `space.json`. Optional surveyed anchors live on
`space.coordinate.anchors`; deleting them leaves the coordinate system valid.
`spatial.localize` solves body pose from those anchors plus camera-frame
observations and a camera-in-body transform. The start pose is not an input.
Detectors must emit `spatial.observation` records; they do not define the
ontology.

Visual evidence is a separate bounded package (`home_cortex.vision`). Its
[domain boundary](home_cortex/vision/README.md) distinguishes detector category
belief, persistent visual candidates, human-confirmed enrollments, and household
items. Detector-native types (YOLO xyxy, tensors, class IDs) do not cross that
boundary. An observation is evidence only: it does not create items, confirm
identity, or write `located_in`. Unknown observer and object positions are null;
available spatial estimates reuse `home_cortex.spatial` types. Media bytes live
outside SurrealDB behind content-addressed references. Evidence clips have an
independent pending/available/failed lifecycle, so observations remain valid
when clip generation is delayed or fails.
The [edge interface](home_cortex/vision/EDGE_INTERFACE.md) defines three
independent channels for live media, observation events, and asynchronous clips,
using at-least-once observation delivery with idempotent ingestion and replay.
The [enrollment boundary](home_cortex/vision/ENROLLMENT.md) separates detector
categories, candidate matches, scored household-item hypotheses, and auditable
human confirmation. Missing items use the existing `write_item` path before a
separate enrollment can be created; visual identity never implies location.
The [Vision page architecture](home_cortex/vision/FRONTEND.md) defines a minimal
live view, polled observation feed, asynchronous clip review, and existing-item
enrollment workflow without introducing a frontend framework.

The Mac EdgeVision runtime (`python -m home_cortex.vision.edge`) is a separate
process from the semantic API. It captures the built-in camera through a
replaceable `CameraSource` and serves an encoded MJPEG-over-HTTP preview
(`http://127.0.0.1:8088/live.mjpg`). MJPEG was chosen because ffmpeg/MediaMTX/
aiortc are not in the current environment; JPEG is encoded, and a browser or
VLC can view it. Frame timestamps are timezone-aware ISO-8601. Identity is
`device:dev_macbook` / `camera:built_in`. Use `--source synthetic` without a
camera. Mac capture needs `pip install 'home-cortex[vision]'`.

Store each fact once. Model an addressable home as an Address, its physical
house as an Item located at that Address, and its rooms as Spaces hosted by the
house Item. A Space may also be `located_in` another Space. Do not add a reverse
spouse edge or a `child_of.json` file.
Likewise, do not add `hosts_space.json`; inverse hosted-space traversal uses the
canonical `hosted_by` table. `get_relationships` consults the registry,
accepts `out`, `in`, or `both` directions, and excludes ended temporal edges
unless `include_ended` is true.

Entities may optionally set `collapse: true` (a boolean) to make semantic
`contents` traversal include items located in all directly or recursively hosted
spaces. Missing or false preserves direct containment. This is executor metadata;
the interpreter still emits the ordinary `contents` intent. Hosting cycles fail
with `computation_impossible` and `hosting_cycle`; oversized adjacency reads fail
with `collection_incomplete`. Results reuse relationship evidence: each `contents`
edge retains the item's source ID and its actual location's target ID. No parent
location edges are synthesized or written. Direct subspace queries and subsequent
`location` traversals retain the concrete graph semantics.

Selecting collapsed contents also returns `content_groups`, preserving each
item's actual containing space and that space's stored display name. The renderer
lists one group per nonempty space. The flat entity value remains authoritative
for counts and further composition; filters and exclusions also restrict the
groups. Direct subspace queries are not expanded or grouped unless that subspace
itself declares `collapse: true`. Rendering neither invents contents nor rewrites
space names. An unspecified named-reference type may remain null for exact-name
resolution; explicit types remain binding, and cross-type name collisions remain
ambiguous.

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
membership edge and falls back to `date_difference(birth_date, mode=years)` using the one
`adulthood_years` value in `schemas/semantic/ontology.yaml`. A person's `child`
relationship remains distinct from a minor household member.

Properties can explicitly come from the final relationship edge. Semantic
`start_date`/`end_date` map to deployment edge fields in the ontology, allowing
the same spouse traversal to resolve either the partner entity or the
relationship start date. The executor validates all operations, filters,
predicate names, property sources, and types before it reads or computes data.

All natural-language fact requests pass through the semantic interpreter,
including identity and household lists/counts. Chinese, English, and mixed
Chinese-English utterances use the same semantic rules and compile to one
canonical language-neutral IR; surface language is not an intermediate
translation step, and IR identifiers stay English. Sentence-based Tier-0 dispatch
and its configuration switch have been removed. Structured callers can pass a
`SemanticFactRequest` directly to `HouseholdFactEngine.execute`.

The fact benchmark records semantic plans, speaker scope, validation, execution,
and latency. Run it on the production GPU host:

```sh
home-cortex-fact-benchmark --backend surrealdb --repeat 1
home-cortex-fact-benchmark --backend surrealdb \
  --question 谁最年幼 --question 我们什么时候结婚的
```

The separate planner-quality benchmark loads more than 100 paraphrases from
`benchmarks/semantic_planner_eval.yaml`. It invokes the interpreter, executes every valid
plan against the deterministic engine, compares normalized semantic meaning,
and reports accuracy by entity reference, traversal, multi-hop kinship,
property selection, filtering, aggregation, temporal operation,
relationship-property lookup, and speaker-relative reference:

```sh
home-cortex-semantic-planner-benchmark \
  --data-dir /app/data --schema-dir /app/schemas/edge
home-cortex-semantic-planner-benchmark --one-per-plan
```

Interpretation resources describe reusable grammar and ontology concepts.
The model explicitly chooses property ownership and supplies both operands for
pairwise comparisons. It selects declared reference concepts through path steps
such as `{"concept":"wife"}`; the ontology expands these into the existing IR
relation paths with all declared filters. Extra step filters are conjunctive and
can only narrow the selected concept. No utterance is examined during expansion.
Validation rejects invalid scope, properties and predicates without semantic
repair. Evaluation alternatives do not change production interpretation.

The interpreter distinguishes an output projection from a filter operand:
`select` with `property=null` returns the matching entity set, while
`select(birth_date)` returns a stored date and `date_difference(birth_date, mode=years)`
returns a computed integer age. Calendar-year constraints use the existing
half-open `date_range` predicate, composed with a list or count operation.
Request-level field filters have literal operands; anchor-relative comparisons
are supported only on traversal steps. Invalid dynamic collection filters are
rejected before graph access, never interpreted as an empty result.

`benchmarks/semantic_planner_age_filters.yaml` exercises these distinctions on
the invented household under `benchmarks/fixtures/semantic-contract`. Its
question wording is kept separate from interpreter examples. Run deterministic
tests locally and real-model evaluations on the isolated production GPU path.

The interpreter exposes one elapsed-date operation: `date_difference`, from
an entity or relationship date to the trusted household clock. `mode` selects
`years`, `months`, `days`, or `seconds`; the structured result carries `unit`
alongside its numeric `value`. Age, relationship tenure, and residence tenure
are different property/reference compositions of this same operation. Renderers
retain the unit instead of showing a bare conversion result.

Years and months count complete calendar periods, truncated toward zero for
negative intervals, rather than using a fixed number of days. A February 29
year anniversary is complete on March 1 in a non-leap year; a month starting
on the 31st is not complete on the last day of a shorter month. Date-only
values use local calendar dates; datetime calendar periods also respect the
local time of day. Days/seconds retain the existing interval semantics.
The adulthood ontology explicitly requires a past input date, so a future
date cannot silently classify a person as a minor.

Structured callers and the interpreter use this same canonical operation;
historical aliases are not part of the execution vocabulary. Existing benchmark
expectations use the canonical syntax; historical result artifacts are not rescored.


## Collection projections, calendar offsets, and discourse

The [bounded composition contract](../docs/semantic-composition-contract.md)
defines validation, result shapes, missing data, and conversation ownership.
`projection="each"` applies an existing scalar operation or property selection
independently to a collection. The result has `shape="rows"`; each `FactRow`
carries its entity, value, unit, evidence, status, and missing requirements.
Missing projected values do not erase successful rows. Relationship properties
produce one row per final edge, including multiple residence periods associated
with the same entity. Edge filters in a relationship projection bind that same
edge. Ordinary singular operations still report ambiguity for multiple matches.

`exclude` subtracts explicitly resolved references before filtering and projection.
The interpreter chooses self, a named entity, or a discourse reference according
to the utterance; unresolved references clarify. The comparison operand `other`
keeps its existing meaning.

`date_add` consumes an entity or relationship date, a bounded signed integer
`amount`, and `mode=years|months|days`. It returns the specified date even in the
past. A nonexistent target day becomes the first day of the following month:
February 29 plus one year and January 31 plus one month both become March 1.
Offsets apply once, preserve date-only values, and preserve household wall time
for datetimes. Ambiguous/nonexistent target wall times and out-of-range dates
fail explicitly. `annual_occurrence` continues to find the next valid recurrence.

`kind="discourse"` references the trusted resolved focus of a preceding user turn,
with `turn_offset=1..8`, `entity_type`, and `cardinality=single|collection`.
The model receives user discourse but no entity bindings or canonical IDs.
The resolver reloads bound identities from authoritative storage. Assistant prose
is neither replayed nor used as evidence. A plural focus cannot silently become
a singular antecedent; unsuccessful and non-fact turns have no focus.

For persistent discourse, create a conversation using
`POST /agent/steward/conversations` or `POST /conversations`, then send its `id`
as `conversation_id` on `/agent/steward/chat`, `/v1/chat`, or
`/v1/chat/completions`, or post turns to `/conversations/{id}/messages`.
The API checks ownership. Transcripts persist in SurrealDB (`gui_conversation`,
`gui_message`); they are not household graph facts. The semantic coordinator
scopes pronoun focus by conversation, speaker, household and agent, serializes
concurrent turns, and retains eight user turns in at most 1,000 process-local
sessions. Discourse eviction or API restart loses focus and requires
clarification; the stored transcript remains. Requests without an ID interpret
the current turn once, then re-ground only the earlier user turns explicitly
referenced by its discourse plan. Unauthenticated requests never persist
discourse state.


The ingestion endpoint rejects unknown relationship files, invalid endpoint
types, references to nodes missing from the source data, temporal fields on
non-temporal edges, reverse duplicates of a symmetric fact, and a registered
relationship without a corresponding JSON source file. An empty relationship
must be represented by `[]` so re-ingestion can prune previously stored facts.
Ingestion also clears records from explicitly retired relationship and node
tables during ontology migrations, including the former `resides_in` table.
Export is the deterministic inverse: it reads SurrealDB and writes canonical
`nodes/<table>.json` and `edges/<relationship>.json` files. Empty node tables
are omitted; registered relationships with no facts are written as `[]`. Item
shard directories are an ingest-only layout and collapse to a single
`item.json` on export. Leftover retired tables are omitted from the snapshot
and named in the export result; unknown relationship tables still fail.

This version renames the former `resides_in` relationship to `lives_in`. Since
the repository intentionally does not track private household data, rename the
deployment's `data/edges/resides_in.json` to `lives_in.json` before ingesting.
Remove `start` and `end` from `parent_of.json`; `parent_of` is non-temporal in
the V1 schema. The old SurrealDB `resides_in` table is no longer queried and is
pruned on ingest, so it cannot contribute facts after the application is
redeployed.

The agent also applies a deterministic grounding gate. A structured LLM planner
decides whether a request depends on household world state and plans only
against entity fields and relations discovered from the runtime data/schema.
The bounded executor then validates every required field, relation, record
count, and optional freshness constraint before answer generation. Missing
entities, missing fields, incomplete evidence, and stale evidence produce
different fixed responses rather than model-authored facts. Caller-supplied
system messages are discarded before the trusted Cortex policy is applied.

## Canonical item mutations

Runtime callers create, move, update attributes, or delete authoritative Items through
`ItemWritingService`. Its closed request union exposes only `create`,
`update_location`, `update_attributes`, and `delete`, with `preview` and `commit` modes. The service
validates runtime schema ownership and executes each compound change as one
SurrealDB transaction. The steward enables `write_item` through a structured
current-turn mutation intent. The interpreter emits `requires_fact: false`,
`request: null`, and a `mutation` containing `operation`, the literal `item_name`,
the full `location_name` for creation/movement, and `mode`. Create also carries
bilingual display names (`name_en`, `name_zh`) and a readable `item_key` used as
`item:<item_key>`. Creation and attribute
updates carry `attributes` keyed by semantic property names. The one-call interpreter
chooses exactly one fact, mutation, conversation, or multi-intent branch. Read requests
and mutations are mutually exclusive. The agent invokes the tool after interpretation, including
for streaming responses. Discourse replay returns mutation intents without running
them; the ordinary conversation loop cannot invoke `write_item`.

Mutation-enabled agents use `UnifiedSemanticPlanner`, which composes the declared
fact grammar and canonical named-item mutation contract in one structured call.
A turn with more than one independent fact or mutation objective emits only the
non-executable multi-intent branch; the service performs no graph query or write
and asks the user for one instruction at a time. Read-only agents and the fact
benchmark continue to use `SemanticFactPlanner`.

The model-facing adapter resolves names within the configured household and
stores create names as `{en, zh}` using the interpreter translations. New
items use the interpreter `item_key` as `item:<item_key>` rather than a hash.
It does not expose physical IDs, properties, or raw graph state to the model.
Repeating a creation already
recorded at the same location with matching explicit attributes is a no-op.
Conflicting attributes require an explicit update. An existing item at another location
requires an explicit move; an ambiguous or missing destination does not write.
A container name never implies a particular internal space. Confirmation text is
rendered from the actual mutation status, and preview is explicitly marked unsaved.

Writable attributes are declared with `item_writable` types in the ontology;
the catalog includes those fields even before any item has a value. Current fields
are item_type, brand, model, color, quantity, unit, description, and expiration_date.
The user permits inferring a clear item category from its name; uncertain categories
use `unknown`. Other attributes require explicit input. Every new canonical item
gets at least item_type; existing uncategorized records can be updated explicitly.
No retrospective household classification is performed during deployment.

`update_attributes` patches only declared fields and preserves names, locations,
and unspecified values. Preview and repeat no-ops do not write. A transaction
compares the current entity against the validated snapshot before merging,
rejecting concurrent changes. IDs, resolver metadata, and relationships cannot be
modified through the attribute patch. Removing attributes is not part of this API.

The generic read operation `inspect` returns declared semantic attributes for one
entity and marks missing values as unrecorded; single-attribute reads remain
`select(property)`. Both use deterministic retrieval/rendering.

The direct Python `ItemWritingService` contract also adds `update_attributes`,
using item_id and properties; the name adapter owns semantic-to-storage mapping.
The `write_item` tool now takes names instead of that low-level contract. The
offline compact codec is version 4 because create now carries bilingual names and a readable item_key;
old version 1, 2, and 3 payloads are rejected rather than reinterpreted. Production continues
using expanded JSON. See
[`docs/design/item-writing-api.md`](../docs/design/item-writing-api.md) for the
request/result contract, transaction boundaries, and ingestion constraint.

## First run

From `docker/cortex` on the server:

Copy `.env.example` to `.env`, then set `SURREAL_PASS`, `CORTEX_API_KEY`, and
`CORTEX_IDENTITY_MAP` in that file. `LLM_PROVIDER` selects the steward's
language-model backend (`ollama` by default, or `openrouter`). For Ollama, set
`OLLAMA_MODEL`. For OpenRouter, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`
to a model id such as `anthropic/claude-sonnet-4`. Map the email used to sign
in to the household GUI to Jian's graph record:

```dotenv
CORTEX_API_KEY=replace-with-a-long-random-secret
CORTEX_IDENTITY_MAP={"email:your-login@example.com":"person:jian_kuang"}
```

A stable user id is a stronger mapping key when it is known:

```dotenv
CORTEX_IDENTITY_MAP={"id:household-user-uuid":"person:jian_kuang"}
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

## Household GUI

Compose runs three application containers behind an independent nginx reverse
proxy on port 3000: `home-gui` (static Svelte client), `cortex-api` (this
package), and the proxy. The GUI is stateless. Chat transcripts, greetings, and
identity live in Cortex.

Open http://localhost:3000 and sign in with a mapped email plus `CORTEX_API_KEY`.
The key is exchanged for an HttpOnly cookie at `POST /session`; it is not kept
in JavaScript storage. Select `老管家` for the steward (tools and graph facts)
or a bare Ollama/OpenRouter model to test the underlying LLM without Cortex
tools. New steward chats show the relationship-aware greeting before the first
user turn. Switching models starts a new conversation.

Local GUI development: from `src/home_gui`, `npm install && npm run dev` (Vite
proxies API paths to uvicorn on port 8001).

Curl still uses `Authorization` and `X-OpenWebUI-User-Email` / `X-OpenWebUI-User-Id`.
If identity mappings are configured, an unknown email receives `identity_not_mapped`.

## Agent runtime architecture

The runtime is intentionally split into focused layers:

- `agent_service.py` is the public coordinator. It normalizes trusted identity and
  conversation input, then selects the appropriate execution path.
- `schema_catalog.py` discovers queryable node properties, edge properties, and
  relationship semantics from the deployed data and edge registry. Adding a
  field to a node JSON file makes it available to planning without changing
  factual-grounding code.
- `semantic_facts.py` coordinates interpretation, execution, rendering, and
  diagnostics. Canonical request/context/result types live in `semantic_ir.py`.
  `semantic_planner.py` interprets language; `entity_resolver.py` grounds references
  and containment paths; `household_fact_engine.py` computes and validates evidence;
  `fact_renderer.py` formats the result. Import these owners directly. The engine
  does not import the planner or renderer.
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

The steward's model is `OLLAMA_MODEL` or `OPENROUTER_MODEL` when its
`config.yaml` model name is null, according to `LLM_PROVIDER`.
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

```sh
# From the repository root; this copies both runtime and engineering packages.
./scripts/maintenance/copy_tier1_bench_into_api.sh
# Follow the isolated invocation printed by the helper.
```

### Opt-in latency and token audit

Set `CORTEX_PROFILE_REQUESTS=1` in the API process environment before startup to
emit one bounded `request_profile` JSON log per HTTP request. It includes the
complete ASGI lifetime (including streamed bodies), named nested stage intervals,
and every model transport call, including validation retries, discourse replay,
and ordinary chat/tool selection. Logs retain at most 512 intervals and count
any dropped intervals. They contain numeric usage/timing and exception types;
prompts, model output, SQL, identities, and graph values are omitted.

For an isolated Python invocation, use
`with home_cortex.request_tracing.trace_request() as trace:` around the awaited request
and inspect `trace.events`. Tracing is disabled by default. Each event has a
start offset and elapsed duration; nested intervals must not be added together.
The audit harness calculates exclusive contributions for its sequential requests.

Provider-reported input/output/reasoning tokens and load/prefill/generation times
remain `null` when unavailable. Non-streaming planner calls cannot measure TTFT.
Streaming calls measure the first observed content, reasoning or tool delta;
post-TTFT wall time includes consumer backpressure. Provider generation duration
is a distinct field. The older `routing_ms`, `llm_ms`, and `request_ms` diagnostic
fields overlap; `last_planner_runtime` retains only the latest attempt and must
not be used to total retries or concurrent requests. Use the request trace instead.

Reproduce the synthetic audit from an isolated package on the GPU host:

```sh
PYTHONHASHSEED=0 PYTHONPATH=src python -m scripts.profiling.token_latency_audit \
  --mode ollama --repeat 1 --output /tmp/probe-audit.json
PYTHONHASHSEED=0 PYTHONPATH=src python -m scripts.profiling.token_component_probe \
  --output /tmp/component-summary.json
```

`--mode replay --repeat 10` on the first command measures deterministic processing
with declared expected IR and no inference; its scores are not model accuracy.
The benchmark uses each dataset's frozen clock and an excluded full warm-up pass.
Pin the hash seed in both comparison processes: relationship-property dictionaries
currently inherit set iteration order. The separate raw-generation component
probe consumes one output token per component and does not measure chat framing.
Run it outside timed benchmark passes.

`python -m scripts.profiling.http_latency_audit --output /tmp/http-audit.json` supplements this with
the full ASGI route and real SurrealDB queries. It uses existing DB credentials,
creates a UUID-named `hc_latency_audit_*` namespace with the invented fixture,
and removes that namespace in `finally`. Run only in an isolated process;
it replaces that process's FastAPI runtime state. It never selects the configured
production namespace. ASGI timings exclude browser, reverse-proxy and TCP ingress.
See [the audit report](../artifacts/token-latency-audit/REPORT.md) for results and limits.

## Semantic ownership and convergence

`AgentService` constructs the trusted `AgentRequestContext`; conversation focus
extends it and named mutations receive that same context. `UnifiedSemanticPlanner`
or the read-only `SemanticFactPlanner` compiles model output through ontology
expansion and validation into the canonical `SemanticPlan` envelope.
`HouseholdFactEngine` executes its `SemanticFactRequest` using `EntityResolver`
and returns `FactResult` for deterministic rendering. Invalid location plans
are never rewritten into a different request.

The request path has one representation at each semantic boundary:

| Stage | Owner | Authoritative value | Input → output |
|---|---|---|---|
| HTTP and identity | `api.py`, `agent_service.py` | `AgentRequestContext` | authenticated headers and messages → trusted context |
| Conversation focus | `semantic_conversation.py` | `DiscourseContext` | prior resolved focus → scoped context extension |
| Interpretation | `unified_semantic_planner.py`, `semantic_planner.py` | `SemanticPlan` | utterance plus capabilities → one validated branch |
| Read request | `semantic_ir.py` | `SemanticFactRequest` | semantic plan → unchanged executor input |
| Grounding | `entity_resolver.py` | `ResolvedEntities` | semantic references plus context → graph entities and edges |
| Computation | `household_fact_engine.py` | `FactResult` | grounded request → deterministic value and evidence |
| Persistence | `retrieval.py`, `db.py` | graph records | bounded internal graph calls → storage records |
| Presentation | `fact_renderer.py` | answer text | request plus `FactResult` → localized answer |

`SemanticFactService` coordinates these stages and records diagnostics; it does
not introduce another request or result DTO. `model_loop.py` handles ordinary
conversation, calculation, and calendar tools. It cannot route household graph
questions around semantic planning. The two provider adapters share planner
messages and schemas, while retaining their small protocol-specific request,
streaming, and error code because those wire contracts differ.

The schema-named modules own different facts:

| Module | Owns | Does not own |
|---|---|---|
| `semantic_ontology.py` | parsing and validating reusable property, concept, predicate, and relation declarations | deployed storage fields |
| `edge_schema.py` | physical relationship tables, directions, endpoints, temporal fields, and inverses | natural-language concepts |
| `schema_catalog.py` | deployed node/edge fields and types, plus shared name/appellation matching | planner grammar |
| `semantic_schema.py` | binding ontology vocabulary to the catalog, plan validation, and derived planner capabilities | a second ontology |
| `semantic_contracts.py` | typed property constraints resolved from the canonical ontology | the deployed catalog fields |
| `operator_registry.py` | the public operation type, predicate allowlist, validation contracts, and deterministic implementations | entity resolution or rendering |

`semantic_ir.py` is the canonical read IR. `mutation_ir.py` remains separate
because writes add preview/apply modes, update constraints, and confirmation
semantics that no factual request carries. `semantic_contracts.py` holds the
typed contract that `SemanticSchemaRegistry` resolves once per deployment; plan
validation and capability generation both read that single resolved contract.

Speaker identity is established once by `AgentService` from the authenticated
identity mapping. The same `AgentRequestContext` reaches discourse resolution,
the fact engine, and named mutations. Internal graph calls may carry its caller
and household IDs as scoped lookup parameters, but they never reconstruct them
from model arguments. `EntityResolver` alone interprets self, names, discourse,
kinship paths, and household-relative references.

Rendering boundaries are similarly explicit. `FactRenderer` interprets only the
canonical request/result pair. `semantic_display.py` describes semantic scopes
and filters; `display.py` sanitizes generic model tool records; `greetings.py`
selects deterministic reception text; and `text.py` contains small shared text
normalization helpers. None of them reads graph facts or changes computed values.

Storage adapters share `schema_catalog.matching_named_entities` for normalized
aliases, scoped appellations, ordering, and limits. Authentication remains an
exact-ID lookup. Alias SQL projects identity metadata rather than full profiles.
Mutation resolution and rendering use the active ontology.

See [the convergence report](../artifacts/architectural-convergence/REPORT.md)
for ownership decisions, measured local results, and outstanding GPU acceptance.
The [context-surface cleanup report](../artifacts/context-surface-cleanup/REPORT.md)
records the operation/IR audit, branch coverage, and final sizing.
