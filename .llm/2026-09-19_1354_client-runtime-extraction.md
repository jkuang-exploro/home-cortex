Date: 2026-09-19 13:54 PDT
Type: refactor
Status: completed

## Objective

Extract laptop/camera execution from Home Cortex into an independent sibling
`home_cortex_client` project while retaining backend visual-evidence and spatial
contracts.

## Context

Vision execution is paused pending MicroDuck hardware. The repository still held
a useful Mac OpenCV/MJPEG edge runtime plus a dead-end Tapo authentication,
FFmpeg relay, and embedded player prototype.

## Findings

- The generic edge package had no backend imports and could be moved cleanly.
- Tapo camera, relay, and player code was isolated but depended on brittle
  device-specific behavior and was not worth transferring.
- `home_cortex.vision` contracts, ports, artifact storage, and spatial packages
  are backend/domain code and remain valid without any device runtime.
- No production observation HTTP adapter exists; the current structured seam is
  canonical JSON/JSONL parsing plus `ObservationIngestionService`.

## Decisions

- Move/adapt only the Mac/synthetic capture and MJPEG runtime.
- Delete the Tapo, FFmpeg relay, and old player prototype and their tests.
- Keep backend evidence contracts, ingestion ports, artifact storage, enrollment
  semantics, and spatial transforms/localization.
- Enforce the repository boundary with source-import, module-presence,
  dependency, and GUI dependency tests.

## Changes

- Created independent sibling project `../home_cortex_client` with a package,
  CLI, environment-backed device configuration, `.env.example`, camera optional
  dependency, lockfile, hardware-free tests, and boundary documentation.
- Removed backend `vision.edge`, `vision.camera`, `vision.relay`, embedded Vision
  web assets, and their device-specific tests.
- Removed backend OpenCV and cryptography extras; refreshed `uv.lock`, which also
  removed their transitive NumPy, CFFI, and pycparser packages.
- Updated root, backend, edge-interface, and frontend documentation for the
  three-project ownership model and paused MicroDuck state.

## Validation

- Client isolated suite: 10 passed, 1 opt-in physical-camera test skipped.
- Backend focused API/Vision/spatial suite: 191 passed.
- Backend import probe: API loads neither Vision nor `home_cortex_client`.
- Home GUI production build: 120 modules built successfully.
- Full backend suite: 931 passed, 1 failed. The sole failure is the pre-existing
  composition fingerprint reference to missing
  `benchmarks/composition/codex-approval.md`.
- `git diff --check`: passed.

## Remaining Issues

- Repair the unrelated composition fingerprint manifest in its already planned
  evaluation-correction work cycle.
- No production VisualObservation transport adapter exists. Do not implement it
  until Vision resumes with the MicroDuck boundary.

## Recommended Next Step

Return to the backend-functional-breakdown plan and take the highest-value
backend structural cleanup. Do not continue Tapo integration; keep Vision paused
until MicroDuck hardware defines the next client interface.
