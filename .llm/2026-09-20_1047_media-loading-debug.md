Date: 2026-09-20 10:47 PDT
Type: debugging
Status: completed

## Objective

Diagnose and fix the `/media` page remaining indefinitely at "Loading your library…".

## Context

The deployed page was available in the user's authenticated Safari session. Local
Docker was still unavailable, and sandboxed command-line DNS could not resolve the
deployment hostname.

## Findings

- In the live authenticated browser, requesting
  `/media-api/items?type=all&limit=1&offset=0` rendered the Home Cortex chat SPA.
  The deployed nginx is therefore sending `/media-api/*` to the GUI catch-all
  instead of the `home_media` upstream.
- The frontend parsed the non-JSON 200 response as `{}`, accepted it as a media
  page, then failed while accessing the missing `items` value. The browser retained
  the last successful render, which was the loading state.
- The client also had no timeout, so a genuinely stalled proxy could produce the
  same permanent loading symptom.
- Deployment `docker ps` evidence confirmed the mismatch: `home_media` and
  `home-gui` were created about 50 minutes earlier, while `cortex-proxy-1` was
  created 13 hours earlier. The running nginx process therefore never loaded the
  new bind-mounted configuration.

## Decisions

- Treat JSON content type and the media-page response shape as runtime contracts.
- Abort normal media API requests after 15 seconds and show a visible failure;
  allow up to ten minutes for an explicit synchronous refresh.
- Return a specific diagnostic when nginx serves the web app at a media API URL.

## Changes

- Updated `src/home_gui/src/lib/media.ts` with request timeout handling, JSON
  content-type checks, error-envelope parsing, and list/refresh payload validation.
- Extended the frontend boundary guard for the timeout and invalid-response checks.
- No additional repository changes were needed after the container-age evidence;
  recreating the deployed proxy is the required operational correction.

## Validation

- Reproduced the incorrect live routing in Safari using the authenticated session.
- GUI production build: 122 modules transformed successfully.
- Focused deployment/boundary suite: 6 passed.
- `git diff --check`: passed.

## Remaining Issues

- The deployment must recreate `proxy`; the running proxy predates the
  `/media-api/` configuration.

## Recommended Next Step

On `home-cortex-0`, force-recreate `proxy`, confirm `nginx -T` contains the media
location, then verify the internal service health and reload `/media`.
