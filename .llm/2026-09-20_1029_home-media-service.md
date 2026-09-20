Date: 2026-09-20 10:29 PDT
Type: coding
Status: partial

## Objective

Create an independent, deterministic, read-only `home_media` service, integrate
it behind the existing authenticated nginx origin, and add a native `/media`
gallery to `home_gui` without changing the `home_cortex` Python package.

## Context

Source media is expected on the deployment host at `/opt/data/photo` and
`/opt/data/video`; neither path exists on this development Mac. The existing GUI
uses a signed HttpOnly `cortex_session` cookie validated by Cortex `/session`.
Docker Desktop was not running during final validation.

## Findings

- Nginx `auth_request` can reuse `/session` without sharing Python code or the API
  key with the media service.
- Starlette `FileResponse` streams files and implements the tested byte-range
  responses, including 206 and 416 behavior.
- The full suite retains its known failure because
  `benchmarks/composition/codex-approval.md` is absent.
- The deployment media roots were unavailable locally, so real inventory and
  production performance metrics could not be measured.

## Decisions

- Use independent FastAPI packaging, SQLite, Pillow/pillow-heif, ffprobe, and
  ffmpeg; do not import between `home_media` and `home_cortex`.
- Derive stable IDs from SHA-256 of media type, a NUL separator, and POSIX
  relative path; expose no absolute paths.
- Keep the service off published host ports and authenticate at nginx through an
  internal GET subrequest to Cortex `/session`.
- Start without scanning, use explicit locked refreshes, and generate oriented
  thumbnails and video posters lazily.

## Changes

- Added `home_media/` packaging, image, config, contracts, probing, incremental
  scanner, SQLite index, safe resolver, derivatives, API, docs, lockfile, and tests.
- Added Compose read-only source mounts, writable derived state, private exposure,
  and authenticated `/media-api/` nginx routing.
- Added the GUI `/media` route, navigation, month grouping, filters, pagination,
  refresh, photo/video viewers, keyboard controls, and empty/error states.
- Added deployment/boundary regression guards and root documentation. No
  `src/home_cortex` file changed.

## Validation

- Independent service suite: 16 passed.
- GUI production build: 122 modules transformed successfully without warnings.
- Compose config validation: passed, with expected unset local SurrealDB warnings.
- Deployment/boundary tests: 6 passed.
- Full repository suite: 951 passed, 1 known failure for the absent composition
  approval artifact.
- Python compile and `git diff --check`: passed.
- Container build could not run because the Docker daemon was unavailable.
- Real inventory, timings, cache/index sizes, page latency, and browser acceptance
  were not run because the deployment host and media roots were unavailable.

## Remaining Issues

- Build/deploy on `home-cortex-0` and complete the real-library acceptance metrics.
- Confirm nginx syntax in the running proxy; local Docker was unavailable.
- Repair the separately tracked composition approval/fingerprint artifact.

## Recommended Next Step

On `home-cortex-0`, ensure `/opt/data/home-media-cache` is writable, run Compose
build/up, sign in, open `/media`, refresh, and record inventory, scan/cache/index,
first-page latency, orientation, playback, seeking, refresh, and source integrity.
