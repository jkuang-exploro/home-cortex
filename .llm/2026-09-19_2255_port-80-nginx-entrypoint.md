Date: 2026-09-19 22:55 PDT
Type: coding
Status: completed

## Objective

Make nginx on host port 80 the only LAN-facing Home Cortex web entrypoint, keep
the GUI on internal Compose port 3000, and preserve API routing without adding
TLS or changing the `home-cortex-0` hostname.

## Context

Compose already published nginx on port 80 and did not publish the GUI, but the
three layers disagreed: Compose exposed GUI port 3000, the production GUI image
listened on 4173, and nginx contained duplicate `location /` blocks targeting
both `home-gui:4173` and an invalid direct `home_gui:3000` host. The API was
published to every host interface on port 8001.

## Findings

- The frontend uses same-origin `/session`, `/conversations`, `/agent`, and `/v1`
  paths, so no browser runtime URL change was required.
- Port 4173 remains only in Vite's explicit local preview configuration.
- There is no active Vision web page; tests deliberately assert that retired
  `/vision` routes are not mounted. Vision contract tests remain active.
- Static nginx upstream resolution made the entire proxy fail to start while the
  API container was temporarily absent.
- The first local Compose launch had no env file and initialized a new local
  SurrealDB volume with blank credentials. A later configured API therefore
  failed authentication. The volume was preserved; no database deletion or
  credential mutation was performed.

## Decisions

- Standardize the production GUI container, Compose exposure, and nginx upstream
  on port 3000.
- Bind the direct API maintenance/development port to `127.0.0.1:8001`, retaining
  local tooling without exposing a second LAN web entrypoint.
- Route GUI traffic through one catch-all location and retain/add the active API,
  admin, OpenAPI, and documentation routes.
- Use Docker's embedded DNS resolver plus resolvable upstream zones so GUI service
  does not depend on API availability during nginx startup.

## Changes

- Updated `docker/docker-compose.yml`, `docker/proxy/nginx.conf`, and the GUI
  Dockerfile for public port 80 and internal GUI port 3000.
- Preserved `/session`, `/conversations`, `/agent/`, `/v1/`, and `/health`; added
  proxy coverage for `/admin/`, `/docs`, `/redoc`, and `/openapi.json`.
- Updated active setup documentation to use `http://home-cortex-0/` and removed
  obsolete Open WebUI port-3000 commands.
- Added `tests/test_deployment_network.py` to guard port publication, internal
  targets, hostname, route coverage, lack of TLS, and user-facing documentation.

## Validation

- `docker compose config --quiet`: passed.
- Compose rebuild: GUI and API images built; the GUI production build transformed
  120 modules successfully.
- `nginx -t`: passed in the running proxy.
- Host proxy request with `home-cortex-0` Host resolution: HTTP 200.
- Safari loaded and rendered the Home Cortex GUI through local nginx.
- Proxy-to-`home-gui:3000` internal request: HTTP 200.
- Direct host request to port 3000: connection refused, as intended.
- Focused deployment/API/session/conversation/Vision suite: 93 passed.
- Full deterministic suite: 949 passed, 1 failed. The sole failure remains the
  pre-existing missing `benchmarks/composition/codex-approval.md` artifact.
- `git diff --check`: passed.
- Live `/health` returned 502 because the configured API cannot authenticate to
  the newly initialized local test database. Live login/chat were therefore not
  claimed. Static and mocked API regression tests passed.

## Remaining Issues

- Reconcile or explicitly reinitialize the local test SurrealDB volume credentials
  before performing live health, login, and chat smoke tests. Treat the volume as
  data and obtain explicit approval before deleting it.
- The actual hostname `home-cortex-0` resolves to another LAN host from this Mac;
  validation used local nginx with the intended Host header. Deploy these changes
  on that host for the final hostname-level browser check.
- Vision runtime and its web page remain intentionally paused/absent; only retained
  contract validation was exercised.

## Recommended Next Step

Deploy on `home-cortex-0` with its normal environment and persistent database,
then run `/health`, session login, model listing, and one chat turn through port 80.
