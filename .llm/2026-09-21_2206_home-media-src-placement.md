Date: 2026-09-21 22:06 PDT
Type: refactor
Status: completed

## Objective

Place the independent media microservice project at `src/home_media/`, alongside
`src/home_cortex/` and `src/home_gui/`, while retaining the dashed Docker service
and DNS name `home-media`.

## Context

The preceding rename established `home-media/` at the repository root and
`home-media` in Compose. The requested source organization uses underscores for
filesystem project paths but dashes for Docker identity.

## Findings

- Moving the complete project preserves its conventional inner
  `src/home_media/` Python-package layout and independent `pyproject.toml`.
- Only the Compose build context, active documentation link/commands, and boundary
  test roots depend on the outer project location.
- The Docker service, Docker DNS target, Python import package, and environment
  variables require no rename in this change.

## Decisions

- Use `src/home_media/` as the independent project root.
- Retain `src/home_media/src/home_media/` as the standard Python source layout.
- Keep Compose/nginx identity `home-media`, import identity `home_media`, and
  environment prefix `HOME_MEDIA_`.

## Changes

- Moved the complete project from `home-media/` to `src/home_media/`.
- Updated the Compose build context to `../src/home_media`.
- Updated active root/service documentation and path-boundary tests.
- Added guards that reject legacy root-level `home-media/` and `home_media/`
  project directories.

## Validation

- Independent media suite from the relocated project: 16 passed, with two
  third-party deprecation warnings.
- Deployment and dependency-boundary suite: 6 passed.
- `docker compose -f docker/docker-compose.yml config --quiet`: passed with only
  expected unset local SurrealDB credential warnings.
- `uv lock --check --project src/home_media`: passed.
- `python3 -m compileall -q src/home_media/src`: passed.
- Full repository suite: 951 passed, 1 pre-existing failure for the missing
  `benchmarks/composition/codex-approval.md` artifact.
- `git diff --check`: passed.

## Remaining Issues

- Rebuild `home-media` on the deployment host because its Compose build context
  changed.
- The unrelated composition approval/fingerprint artifact remains absent.

## Recommended Next Step

Deploy `home-media` and recreate `proxy` on `home-cortex-0`, then verify internal
health through `http://home-media:8000/health` and the browser `/media` route.
