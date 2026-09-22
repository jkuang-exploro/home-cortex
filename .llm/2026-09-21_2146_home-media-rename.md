Date: 2026-09-21 21:46 PDT
Type: refactor
Status: completed

## Objective

Rename the media microservice and Docker Compose identity from `home_media` to
`home-media` consistently across deployment, source layout, documentation, and
tests.

## Context

The project directory and Compose service used underscores, while the Python
distribution was already named `home-media`. Python import identifiers and shell
environment variables cannot use hyphens.

## Findings

- Compose accepts `home-media` as a service key and Docker DNS name.
- The import package must remain `home_media`; `HOME_MEDIA_*` environment variable
  names must also remain unchanged.
- Local Docker remained unavailable, so an image-backed `nginx -t` could not run.

## Decisions

- Rename the top-level project to `home-media/`, the Compose service to
  `home-media`, and the nginx upstream/DNS target to `home-media`.
- Retain only technically required underscore forms inside Python imports/package
  paths and environment variables.
- Preserve historical work-log wording rather than rewriting earlier records.

## Changes

- Renamed `home_media/` to `home-media/`.
- Updated Compose build context, dependency, and service key.
- Updated nginx upstream, zone, server, and proxy target.
- Updated active documentation, GUI deployment diagnostics, and regression tests.
- Added guards rejecting the legacy top-level directory and Compose service key.

## Validation

- `docker compose -f docker/docker-compose.yml config --quiet`: passed with only
  expected warnings for unset local SurrealDB credentials.
- Focused deployment and boundary suite: 6 passed.
- GUI production build: 122 modules transformed successfully.
- `uv lock --check --project home-media`: passed.
- `python3 -m compileall -q home-media/src`: passed.
- Full repository suite: 951 passed, 1 pre-existing failure for missing
  `benchmarks/composition/codex-approval.md`.
- `git diff --check`: passed.

## Remaining Issues

- Recreate `home-media` and `proxy` on the deployment host so Docker replaces the
  legacy `cortex-home_media-1` container and nginx resolves the new DNS name.
- Run `nginx -t` in the deployed proxy; the local Docker daemon was unavailable.

## Recommended Next Step

On `home-cortex-0`, remove the obsolete service through Compose, deploy
`home-media` plus `proxy`, and verify the resulting container name and
`/media-api/health` response.
