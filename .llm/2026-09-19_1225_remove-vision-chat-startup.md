# Remove Vision from chat startup

Date: 2026-09-19
Type: refactor
Status: completed

## Objective

Remove the paused Vision prototype from the normal Home Cortex chat startup and deployment path while retaining its source as dormant reference for future MicroDuck integration.

## Context

The API imported Vision camera and relay modules, created a Tapo hub during application startup, mounted four `/vision` routes, exposed Vision settings, and included camera-specific dependencies in the default image. The GUI, proxy, and Compose configuration also exposed the feature even though the Vision epic is paused.

## Findings

- Importing `home_cortex.api` loaded the Vision package because camera and relay types were imported at module scope.
- The API lifespan always constructed and closed a Tapo camera hub.
- The normal image installed FFmpeg and the base Python dependency set included `cryptography` only for the paused camera path.
- The GUI still linked to Vision and proxied `/vision`; Compose and nginx still carried the corresponding configuration and route.
- The full test suite has a pre-existing composition fingerprint failure: `scripts/benchmarks/composition_eval.py` references the removed `benchmarks/composition/codex-approval.md` file.

## Decisions

- Keep `src/home_cortex/vision` intact as dormant implementation evidence until the MicroDuck interface is known.
- Remove every Vision touchpoint from default chat startup, routing, settings, GUI navigation, proxying, and container construction.
- Move `cryptography` to the `vision` and `dev` optional dependency groups so retained Vision tests remain runnable without adding it to production chat installations.
- Remove FFmpeg from the default image rather than maintaining unused camera runtime weight.
- Leave the unrelated composition fingerprint defect unchanged to keep this work cycle focused.

## Changes

- Removed Vision imports, hub lifecycle work, authentication helpers, stream plumbing, and routes from `src/home_cortex/api.py`.
- Removed Vision and Tapo settings and validation from `src/home_cortex/config.py`.
- Removed Vision environment variables and nginx routing from the default Docker stack.
- Removed FFmpeg from the Dockerfile and camera cryptography from base dependencies.
- Removed the GUI sidebar entry, translations, and development proxy.
- Replaced route integration tests with absence checks and retained standalone Vision unit tests.
- Updated active documentation to mark Vision as paused and explain that it is absent from the chat runtime.

## Validation

- Import probe confirmed that importing `home_cortex.api` loads no `home_cortex.vision` modules.
- Relevant API, configuration, architecture, and relay tests: 78 passed.
- GUI dependency install and production build completed successfully; 120 modules were built.
- Docker Compose configuration rendered without any `VISION_*` variables.
- Both `cortex-api` and `home-gui` images built successfully from the current source.
- Full unrestricted test suite: 965 passed, 1 skipped, 1 failed. The sole failure is the pre-existing missing composition fingerprint input described above; all retained socket-based Vision tests passed when local loopback access was available.
- `git diff --check` passed.

## Remaining Issues

- Repair the composition fingerprint manifest and its reference to the removed `benchmarks/composition/codex-approval.md` artifact in a separate work cycle.
- The dormant Vision package still occupies repository and optional test code space by design; decide whether to reuse or delete it after the MicroDuck integration boundary is available.

## Recommended Next Step

Continue the backend cleanup by separating route groups and application wiring from the large `api.py` module, using measured startup and request profiles to prioritize the next extraction.
