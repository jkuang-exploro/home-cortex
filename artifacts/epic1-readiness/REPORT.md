# Epic 1 pre-hardware readiness — 2026-09-28

## Demonstrated Home Cortex path

An engineering-only simulator creates one known `embodiment:microduck-01` in
`space:kitchen`, associated with the existing `agent:butler` (`steward` / 老管家).
It opens a runtime session, sends already-fused canonical telemetry through
`EmbodimentSession` and `EmbodimentPresence`, reads the latest state, transforms
the rigid body box and a body-local camera pose into kitchen coordinates, then
exercises stale, no-estimate, recovery, disconnect, and reconnect states. The
body ID and agent association remain unchanged. No model, GPU, database, camera,
or robot is required.

| Contract | Evidence |
|---|---|
| Embodiment and geometry | Stable `embodiment:` record with explicit box center and dimensions |
| Local frame | Right-handed signed axes; camera pose composes through the embodiment pose |
| Agent association | `steward` maps to `agent:butler`; one body has at most one `agent_id`, many bodies may share it |
| Capabilities | Configured dotted names are distinct from temporarily available session capabilities and model tools |
| Runtime session | Connect, replace, heartbeat, disconnect, reconnect; session ID fences writes from a replaced client |
| Telemetry ingestion | Valid six-DOF `{value,p95}` samples, timestamp ordering, explicit `no_estimate`, latest-state query |
| Spatial state | Eight oriented box corners, nominal axis-aligned envelope, translation-p95 envelope, body-local child pose |
| Queries | Deterministic catalog lookup by agent/name/capability, runtime embodied status, latest telemetry, JSON inspection |

The nominal box and the translation uncertainty envelope are separate outputs.
The latter does not incorporate orientation p95 and is not a joint 95% occupied
volume. The simulator uses one kitchen frame; it introduces no cross-space math.

## Reproducible measurements

Run from the repository root:

```sh
.venv/bin/python -m scripts.probes.embodiment_replay --scenario square --profile low --count 400 --seed 7 --output /tmp/epic1-replay.json
.venv/bin/python -m pytest -q
```

For the 400-sample square path with seed 7, the low-noise fixture had mean
position error **0.00956 m**, mean orientation component error **0.00955 rad**,
reported per-axis p95 **0.01176 m/rad**, and observed aggregate per-axis
coverage **95.21%**. Coverage varied by axis from **93.0% to 96.25%** in this
finite sample. The exact fixture had zero error and 100% coverage. With a
0.25 m / 0.20 rad injected bias, the calibrated fixture raised reported p95
to 0.2696 m / 0.2196 rad and achieved 97.625% aggregate coverage. An explicit
underreported negative fixture used the same bias with one tenth of that p95
and yielded 0% coverage. These are deterministic simulation measurements,
not physical accuracy evidence.

The full deterministic suite passed: **1,091 tests in 19.13 seconds**. The
replay tests cover stationary, straight, square, circle, rotation in place,
forward-and-rotate, and seeded random paths. Recording and replaying a JSON
tape produces the same report. The JSON inspector reports space axes and
optional anchors, body corners and heading, origin, p95, envelopes, local
camera pose, and recent trajectory. The chat-only `home_gui` has no spatial
view; a 2D top-down inspector is a follow-up.

## Codex boundary review

- **Domain boundaries:** the LLM, semantic planner, and household fact executor
  do not compute telemetry or localization. No raw sensor data crosses the
  boundary. A concrete issue was corrected: a replaced session could previously
  accept writes from its old client. HTTP writes now require the current
  `X-Embodiment-Session-ID`, whose process-random component also prevents
  reuse after a server restart.
- **Spatial duplication:** the occupancy helper reuses `BodyGeometry`,
  `LocalFrame`, canonical telemetry, and existing pose/basis transforms; it
  adds only the envelope calculation.
- **Client/server responsibility:** the development-only simulated client was
  moved from the production `agents.session` module into `scripts/maintenance`.
  Production code imports no engineering script.
- **Persistence:** only identity, geometry, association, and configured
  capabilities are durable. Sessions, temporary capabilities, and latest
  telemetry remain in memory. Replay tapes are optional engineering output.
- **MicroDuck assumptions:** its dimensions, name, and body-local camera are
  fixture values only. The core contracts contain no MicroDuck API or device
  algorithm.
- **Cross-space scope:** every replay sample belongs to `space:kitchen`; no
  door transition or household-global frame was added.

## Remaining work and explicit deferrals

A calibrated embodiment source record is not yet deployed, and a physical
client must map its fused localization output to the canonical six-DOF
telemetry contract and current session protocol. Real sensor interface,
camera calibration, IMU integration, VIO, SLAM, sensor fusion, physical
accuracy measurement, centimeter-level field validation, and motor/navigation
integration remain device-side or future work. Cross-space pose, room
transitions, a global coordinate system, and multi-agent body arbitration are
deferred. None of the synthetic coverage numbers validates those systems.
