# Embodiment and physical telemetry V1

`Embodiment` is a durable `embodiment:` identity. Its serialized record contains a
name, one box, an intrinsic `LocalFrame`, and optionally one `agent:` association.
The record ID survives client disconnects and reconnects. This contract does not
infer an agent from a device connection. Configured capabilities and the
association are persistent; active connection and temporarily available
capabilities are runtime state.

The box stores **full** length, width, and height in meters, plus a **required**
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
  "agent_id": "agent:butler",
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
the embodiment's singular `agent_id` field, read as **embodiment controlled by
agent**. One agent may be named by any number of embodiments; each embodiment
names at most one agent. There is no reciprocal edge or second ownership fact.
Moving a body to another agent requires unassigning it first.

`capabilities` is a sorted, unique list of open-vocabulary, lowercase,
namespaced identifiers such as `mobility.move`, `vision.observe`, and
`audio.speak`. They describe configured physical channels, not model-facing
tools or permission to invoke an actuator. `AgentDefinition.allowed_tools`
continues to govern the model's software tools independently. Adding a physical
capability does not grant the agent a tool.

`EmbodimentCatalog` reads the durable `nodes/embodiment.json` source, validates
association against the registered agents, and provides deterministic lookups
by agent, name, and capability. Its `associate` and `unassign` operations return
new snapshots. `save_node_file` atomically writes a snapshot to the durable
source; normal graph ingestion remains a separate operation. Source records are
validated by ingestion and retain the same `embodiment:` IDs across reconnects.

`EmbodimentConnections` is process-local, explicit connection state. Connecting
may report a temporary subset of configured capabilities; disconnecting drops
that runtime state without changing `agent_id`, geometry, or configured
capabilities. `is_currently_embodied` means an associated body has an active
connection in this state, regardless of whether its latest pose is valid.
Telemetry freshness is separately reported by `EmbodimentPresence`. A transport
must call `connect` and `disconnect` before these connection-based queries
reflect live hardware. No device adapter or body-selection planner is part of
this contract.

## Runtime presence

`EmbodimentPresence` keeps one current `PhysicalTelemetry` sample per registered
embodiment. The sample stays in process memory. A restart drops it; the client
publishes the next estimate, and until then lookup is unavailable. High-frequency
history is not written. There is no existing requirement for a pose log, and a
telemetry row is not a household fact.

Durable registration is separate from the sample. Embodiment records remain the
Ticket 1 documents. When `nodes/embodiment.json` is present, startup validates
those records and registers their IDs. Space IDs are read from `nodes/space.json`
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
