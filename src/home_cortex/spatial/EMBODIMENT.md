# Embodiment and physical telemetry V1

`Embodiment` is a durable `embodiment:` identity in SurrealDB. Its node contains a
name, type, configured capabilities, and any calibrated box and intrinsic
`LocalFrame` that are actually known. Geometry and frame may be absent until
configuration establishes them; neither is inferred from the device type. Its
optional `agent:` assignment is a separate `assigned_to` graph edge.
The record ID survives client disconnects and reconnects. This contract does not
infer an agent from a device connection. Configured capabilities and the
association are persistent; active connection and temporarily available
capabilities are runtime state.

When configured, the box stores **full** length, width, and height in meters, plus a **required**
center offset `{x,y,z}` in the embodiment frame. The embodiment origin is a
calibrated body datum; it is not implicitly the box center. Length follows the
frame's `forward` axis, width follows `left`, and height follows `up`. Every
direction is an explicit signed axis such as `+x`; the three axes must form a
right-handed frame (`forward × left = up`). Hardware calibration must select the
origin and axes. No default orientation is assumed.

Example persistent body record:

```json
{
  "id": "embodiment:microduck-01",
  "name": "MicroDuck",
  "embodiment_type": "robot",
  "capabilities": ["mobility.move", "vision.observe"],
  "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
  "geometry": {"box": {
    "length_m": 0.32,
    "width_m": 0.24,
    "height_m": 0.18,
    "center": {"x": 0.04, "y": 0.0, "z": 0.09}
  }}
}
```

The corresponding persistent assignment is
`embodiment:microduck-01 → assigned_to → agent:butler`. The inverse query name
is `embodied_by`; there is no second reciprocal edge. `assigned_to` is unique
per embodiment, enforced by a SurrealDB unique index on its `in` endpoint;
ingestion also rejects two targets for one source. Export writes
`nodes/embodiment.json` and `edges/assigned_to.json`; ingest restores both.
Neither the node nor edge stores connection, heartbeat, session, or telemetry
state. `agent_id` on a domain `Embodiment` is a read projection of that edge,
never a node field. Existing JSON with `agent_id` needs migration to an edge
before ingest.

The MacBook #0 configuration uses `embodiment:macbook-0`, `name: MacBook`,
`embodiment_type: computer`, and `capabilities: [vision.observe]`. Its geometry
and local frame are absent because no physical calibration has been supplied.
The detail API returns `null` for those fields and the GUI shows “Not
configured.” Registration and offline identity reads still work; geometry
calculations require a configured box and frame.

`PhysicalTelemetry` is one client-produced estimate. Its `embodiment_id` names
the durable body. A valid estimate requires one `space_id` and a complete
`RealtimeTransform` representing **T(space ← embodiment)**. `measured_at` is the
physical measurement time with an ISO-8601 offset, normalized to UTC on the wire;
it is never replaced with ingestion time. `validity: "no_estimate"` requires
`transform: null`; `space_id` may be null when even the containing space is
unknown. A no-estimate message means the producer cannot currently produce a
valid pose, not that uncertainty is arbitrarily large. No cross-space transform
or trajectory is implied.

Each transform degree of freedom is `{value, p95}`. `x`, `y`, `z` use the named
space's coordinate units (`coordinate.unit` is canonical `m`); `yaw`, `pitch`,
`roll` use radians. `p95` is a finite, non-negative absolute 95% error bound in
the **same unit as its value**. It is a physical uncertainty claim from the
producer, not a model confidence, standard deviation, or joint six-dimensional
coverage guarantee. Zero is allowed for exact/synthetic measurements. All six
degrees must be present. Orientation uses the existing ZYX convention: yaw about
+z, pitch about +y, then roll about +x, with right-handed positive rotations.
Values are not silently converted or wrapped; producers must send canonical SI
and account for angle periodicity when reporting p95.

The existing `Pose`, `transform_point`, `canonical_basis`, `local_to_physical`,
and `physical_to_local` implement nominal mapping. `nominal_pose()` discards
uncertainty **only for deterministic geometry math**. `box_corners_in_space`
returns eight nominal vertices; it does not propagate p95. A space basis maps
space coordinates to physical SI displacement. For a rotated or scaled
orthogonal basis, geometry helpers use its unit directions for orientation and
its inverse for translation, so scale does not change the physical dimensions
of a rigid body. A sheared or left-handed basis is valid for general point
conversion in the existing spatial module but cannot define this rigid-body
orientation and is rejected by these helpers. The space basis does not create a
global household frame.

Qualitative quality is derived by a caller's policy. `estimate_within_limits`
checks validity, measurement age, and all six p95 values against **supplied**
limits. When given a scaled space basis, it converts position p95 bounds to
physical meters before comparing limits. The contract defines no production
thresholds or authoritative `GOOD`/`DEGRADED`/`LOST` label. The earlier `localization.pose.RuntimePose`
remains a legacy sigma/quality contract for its existing callers; it is not a
canonical physical telemetry message, and no sigma-to-p95 conversion is assumed.

Example valid telemetry:

```json
{
  "embodiment_id": "embodiment:microduck-01",
  "space_id": "space:kitchen",
  "measured_at": "2026-09-27T19:00:00+00:00",
  "validity": "valid",
  "transform": {
    "x": {"value": 1.372, "p95": 0.024},
    "y": {"value": 2.048, "p95": 0.024},
    "z": {"value": 0.0, "p95": 0.03},
    "yaw": {"value": 1.571, "p95": 0.04},
    "pitch": {"value": 0.0, "p95": 0.01},
    "roll": {"value": 0.0, "p95": 0.01}
  }
}
```

Hardware evidence is still needed to select the body origin, axis labels, actual
box dimensions, how the client computes per-axis p95, and how its clock is
synchronized. The sections above are the domain and wire contracts. They do not
connect a device.

## Agent association and capabilities

The existing conversational agent registry keeps the runtime key `steward` and
display name `老管家`. Its configured durable entity ID is `agent:butler`. These
identify one agent; neither is a robot ID. The canonical persistent relation is
the singular `assigned_to` edge, read as **embodiment assigned to agent**. One
agent may be named by any number of embodiments; each embodiment names at most
one agent. There is no reciprocal edge or second ownership fact.
Moving a body to another agent requires unassigning it first.

`capabilities` is a sorted, unique list of open-vocabulary, lowercase,
namespaced identifiers such as `mobility.move`, `vision.observe`, and
`audio.speak`. They describe configured physical channels, not model-facing
tools or permission to invoke an actuator. `AgentDefinition.allowed_tools`
continues to govern the model's software tools independently. Adding a physical
capability does not grant the agent a tool.

`EmbodimentWritingService` owns deterministic create, update, delete, assign,
and unassign operations. It validates the body contract and writes the existing
SurrealDB node/edge model. Assignment checks the agent node and uses one stable
edge identity per embodiment. The service is internal to Home Cortex; the
device-facing session and telemetry APIs cannot call its assignment methods.
At API startup, configured agent identities are materialized as `agent:` nodes
if absent. That step does not create an `assigned_to` edge, and startup does
not require any embodiment to be online. Existing graph agent records are
preserved. The startup catalog is a read of SurrealDB, including bodies that
have never connected in this process. `EmbodimentCatalog` provides lookups by
agent, name, and capability from those reads. It has no assignment writer.
Operators change durable assignments through the writing service or canonical
JSON ingest. The operator CLI is `python -m scripts.maintenance.embodiments`
with `create`, `update`, `delete`, `assign`, `unassign`, `get`, and `list`
commands; `create` and `update` take a JSON file containing a body record
without `agent_id`.

`EmbodimentConnections` keeps one process-local runtime session per embodiment.
The session has its own id, `online`, `connected_at`, `last_seen`, and the
capabilities advertised as available. It is not an embodiment record. A connect
for a known body opens that session. Connecting again while it is still online
replaces the session id and timestamps. Disconnect marks the same session
offline and clears advertised availability. A later connect opens a new session
for the same embodiment id. Session IDs include a process-random component so
an old token cannot match the first session after a server restart.
`last_seen` advances on heartbeat, capability
refresh, and accepted telemetry. Offline is an explicit disconnect; age alone
does not close a session.

A client presents a persistent embodiment id. Registration reads that body
and its `assigned_to` edge from SurrealDB, caches the read, and opens the
runtime session. The client sends only the capabilities available right now.
Those names must be a subset of the configured capabilities. Omitting a
configured name, such as `vision.observe`, means that channel is supported
and currently unavailable. The session does not store the agent assignment.
An unknown id, such as `embodiment:unknown-device`, is rejected with
`unknown_embodiment` and does not create a body or an assignment. Disconnect
leaves the SurrealDB record unchanged. Reconnect reads the same record,
opens a new runtime session, and the stored assignment is visible again.

## Read model

`GET /api/embodiments` and `GET /api/embodiments/{id}` build one view per
persistent body. The list is the SurrealDB embodiment table, so a body with
no session is still listed. The view is not written back.

SurrealDB supplies `id`, `name`, `embodiment_type`, geometry, local frame,
the assigned agent, and configured capabilities (`supported: true`). `linked`
is true when an `assigned_to` edge exists. The runtime supplies `connected`,
`connected_at`, `last_seen`, whether each supported capability is `available`,
and telemetry. `connected` is true only while a session is online. A missing
session is offline, not a missing embodiment. An existing body may be
unlinked and either online or offline. Selecting that body for a conversation
requires the persistent assignment to the conversation's agent and does not
require a session. A physical capability is executable only when the body
exists, is assigned to the current agent, is connected, persistently supports
the capability, and currently advertises it. Those failures stay distinct.

`is_currently_embodied` means an associated body has an online session,
regardless of whether its latest pose is valid. Telemetry submitted through
the client protocol requires that online session and its current session ID.
For HTTP writes after registration, send the returned ID in
`X-Embodiment-Session-ID`. A replaced client's old ID cannot submit telemetry,
refresh capabilities, heartbeat, or disconnect the new session.
`EmbodimentPresence` still stores the latest admitted sample. The engineering
simulated client exercises only this protocol:

```sh
python -m scripts.maintenance.embodiment_client demo
```

`register` with `--base-url`, `--api-key`, and `--embodiment` opens a session
on a running API. The client does not read sensors or create embodiment
records.

## Runtime presence

`EmbodimentPresence` keeps one current `PhysicalTelemetry` sample per registered
embodiment. The sample stays in process memory. A restart drops it; the client
publishes the next estimate, and until then lookup is unavailable. High-frequency
history is not written. There is no existing requirement for a pose log, and a
telemetry row is not a household fact.

Durable registration is separate from the sample. At startup the API loads
embodiment records and assignments from SurrealDB and registers their IDs.
Space IDs are read from `nodes/space.json`
when that file is present. A missing file registers nothing. Telemetry for an
unregistered embodiment or space is rejected. This does not create embodiment
records and does not write samples back to the graph.

Admission is ordered by `measured_at`, never by arrival time:

- a newer sample, including a newer `no_estimate`, becomes current;
- the same sample submitted again does not change current state;
- an older sample is ignored;
- a different sample with the same `measured_at` is rejected.

`derive_telemetry_status` reports `valid`, `unavailable`, `fresh`, and `stale`.
`valid` means the current sample is a pose. `unavailable` means there is no
current sample or the current sample is `no_estimate`. `fresh` and `stale`
compare measurement age with the observer window (`stale_after_s`, default 2
seconds). That window is not a `GOOD` / `DEGRADED` / `LOST` label. Position and
orientation `p95` limits remain on `estimate_within_limits`.

Clients submit and read the current sample at
`POST` and `GET /v1/embodiments/{embodiment_id}/telemetry`. The path ID must
match the sample. The body is the Ticket 1 telemetry object, including all six
transform degrees. Home Cortex does not accept raw sensors or a shortened pose.

## Deterministic occupancy and pre-hardware replay

`spatial.occupancy` derives the axis-aligned envelope of the eight nominal
oriented box vertices in a single space. `translation_p95_envelope` enlarges
that envelope by x/y/z p95 values in space coordinates. It keeps body geometry
and localization uncertainty separate. It does **not** incorporate orientation
p95 or claim a joint 95% occupied volume; applications needing a conservative
orientation-aware bound must add and validate that calculation later.

The engineering-only `scripts.probes.embodiment_replay` constructs known
ground-truth trajectories, adds seeded error to already-fused estimates, and
replays canonical messages through `EmbodimentSession` and `EmbodimentPresence`.
It measures position/orientation error and observed per-axis p95 coverage,
then exercises stale, no-estimate, recovery, disconnect, and reconnect states.
Its JSON inspector includes space axes, nominal body corners, both envelopes,
the body origin, a body-local camera pose mapped into the space, and recent
poses. No physical client implementation imports the simulator. The current
`home_gui` has no spatial view; a 2D inspector is a follow-up after the JSON
contract and live data lifecycle are settled.

## Presence and conversation selection

`GET /api/embodiments` lists persistent bodies in ID order. An optional
`agent_id=steward` filter returns bodies linked to that registered agent.
`GET /api/embodiments/{id}` adds box geometry, local frame, and the full latest
canonical telemetry sample. Both routes require the usual API or GUI session
authentication. The read projection reports `unlinked`, `linked_offline`, or
`linked_online` from the persistent association and runtime session. It does
not infer online state from telemetry. A linked online body with no estimate is
normal. The telemetry `available` flag requires an online session and a fresh,
valid sample; `valid` and `fresh` are reported separately. Runtime session IDs
are omitted from this browser-facing read API because they fence client writes.

`PATCH /conversations/{id}/active-embodiment` accepts exactly
`{"active_embodiment_id": "embodiment:..."}` or `null`. The owned conversation
must belong to a registered agent, and a selected body must already be linked
to that agent. The selected ID is stored once on the conversation and appears
in transcript and list responses; it does not change agent identity, open a
device session, or imply localization. Transcript responses also identify the
agent's persistent `agent_entity_id`; the existing `agent_id` remains the
runtime registry key. Offline selection remains valid and
ordinary chat continues. The same context is passed through the existing
agent execution flow for browser, OpenAI-compatible, and later voice callers.

`EmbodimentDirectory.action_availability(agent_id, embodiment_id, capability)`
is the deterministic pre-dispatch gate for future physical action executors.
It returns an `ActionAvailability` with `available` and a structured code after
checking existence, association, online session, configured capability, and
current capability advertisement in that order. It performs no physical action.
