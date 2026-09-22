Date: 2026-09-21 22:00 PDT
Type: debugging
Status: completed

## Objective

Diagnose the IDE error marker on `src/home_gui/tsconfig.json`.

## Context

The marker appeared around the explicitly configured `svelte` and `vite/client`
type packages. The frontend dependency directory had previously been removed as
reproducible cleanup while disk space was constrained.

## Findings

- `src/home_gui/node_modules` was absent, so the TypeScript language service could
  not resolve the configured type packages.
- The TypeScript configuration itself is valid and requires no change.

## Decisions

- Restore the locked frontend dependencies and leave them installed for IDE type
  resolution.
- Do not remove or weaken the explicit type configuration merely to suppress the
  missing-package diagnostic.

## Changes

No repository source changes were made. `npm ci` restored the ignored,
reproducible `src/home_gui/node_modules` directory.

## Validation

- `src/home_gui/node_modules/.bin/tsc -p src/home_gui/tsconfig.json --noEmit`:
  passed with no diagnostics.
- `npm run build --prefix src/home_gui`: passed; 122 modules transformed.
- The tracked working tree was clean before this work-log entry was added.

## Remaining Issues

None identified. VS Code may need its TypeScript server restarted if it retains a
stale diagnostic after dependency restoration.

## Recommended Next Step

If the marker does not clear automatically, run “TypeScript: Restart TS Server”
from the VS Code command palette.
