# Epic 2 architecture review after the available prototype

## Decision

**HOLD B8+ — the requested vertical prototype is not present in this working
tree.** The implemented subset preserves the intended production boundary, but
the end-to-end workflow cannot yet be confirmed. Current executable scope stops
at:

```text
MacCameraSource or SyntheticCameraSource
  -> CameraFrame
  -> EdgeRuntime
  -> MJPEGStreamServer
```

The repository has canonical observation, clip, candidate, enrollment, artifact,
and ingestion contracts, plus in-memory contract tests. It does not contain a
detector or detector adapter, tracker, observation publisher, production
observation repository, clip worker, candidate/enrollment application service,
vision API routes, or frontend assets. No finding below assumes those missing
stages work.

## Findings

### Must fix before B8+

1. **Complete the first vertical slice and rerun this review.** The central review
   inputs—YOLO, tracking, observation delivery, clip creation, review, enrollment,
   and frontend integration—have no implementation to inspect. Contract tests do
   not substitute for boundary evidence from those components.
2. **Make retention operational before enabling continuous persistence.**
   `ObservationRepository` is currently a protocol and its only implementation is
   an in-memory test double. A production repository must apply an admission
   policy, configured expiry for unpinned observations, pinning for candidate and
   enrollment evidence, artifact garbage collection, and an ID/hash ledger that
   lasts through the producer retry horizon. Shipping an append-only SurrealDB
   implementation would create unbounded growth.
3. **Verify enrollment against the real household-write boundary.** The contracts
   correctly separate `HouseholdItemHypothesis` from `VisualEnrollment`, but no
   application service enforces existing-item resolution, authenticated human
   authority, one active enrollment, or merge/revocation transactions. The
   vertical test must prove that a missing item uses the existing `write_item` →
   `NamedItemWritingService` → `ItemWritingService` path and that enrollment never
   creates an item or a `located_in` edge.
4. **Implement and inspect the stable frontend DTO boundary.** There is no
   `/vision` page or `/v1/vision` API. The first page must consume the derived
   Home Cortex DTOs in `FRONTEND.md`; it must not import detector labels, tensor
   shapes, tracker internals, filesystem paths, RTSP URLs, or direct database
   operations.

### Should fix soon

1. **Fixed in this review: edge capture failure had no stable state.** A failed
   `CameraSource.read()` previously appeared only as `running=false`, which could
   not distinguish camera loss from a normal stop. `EdgeRuntime.health_json()` now
   exposes `source_status` and a bounded `camera_read_failed` code without leaking
   native exception text.
2. **Fixed in this review: stream startup failure leaked capture resources.** If
   the HTTP stream could not bind after capture started, the camera thread and
   source could remain open. `EdgeRuntime.start()` now calls its normal cleanup
   path before re-raising.
3. **Add source authentication when the observation transport is implemented.**
   Ticket 2 requires the authenticated producer credential to be bound to its
   declared `device_id`. This cannot be checked until an HTTP or message adapter
   exists.
4. **Keep detector output adaptation at edge egress.** No detector exists yet.
   Its implementation should emit `DetectorBelief` and normalized boxes before
   calling `ObservationSink`; model class IDs and result objects must end there.

### Acceptable V1 debt

1. The standard-library MJPEG server is suitable for the local Mac demo. It is
   edge-owned and does not make the semantic API a video proxy.
2. `ArtifactStore` is a local filesystem content store. It keeps bytes outside
   SurrealDB and can later sit behind `ArtifactSink`; reading a complete short
   clip into memory is acceptable for the first bounded prototype.
3. Observation and clip delivery use protocols and in-memory tests without a
   distributed queue. At-least-once, idempotent semantics are sufficient for V1.
4. Frontend polling, no automatic candidate merge, and no embedding recognition
   remain appropriate V1 choices.

## Review matrix

| Review question | Available evidence | Result |
|---|---|---|
| Edge leakage | `cv2` occurs only in lazy `_load_cv2()` inside `vision.edge.sources`; it remains in the optional `vision` dependency. `EdgeRuntime` accepts `CameraSource`. No YOLO or detector filesystem code exists. | Pass for current subset; detector stage unreviewed |
| Semantic leakage | Vision domain/port imports do not load `writing`, `db`, `retrieval`, or SurrealDB. No vision Python code invokes `write_item`, creates `item`, or changes `located_in`. | Pass for current subset; enrollment stage unreviewed |
| Artifact leakage | `ArtifactStore` writes content-addressed files; contracts carry `ArtifactReference`. There is no visual SurrealDB repository. | Pass for current subset; persistence adapter unreviewed |
| Persistence growth | Only a repository protocol and memory test double exist. Retention is documented but not executable. | Must fix before continuous ingest |
| Duplicate authority | Category, candidate hypotheses, item hypotheses, enrollment, candidate state, clip state, source identity, and producer profile are distinct canonical values. No competing implementation exists. | Contract passes; repository enforcement unreviewed |
| Frontend leakage | No HTML, CSS, JavaScript, vision routes, or frontend DTO assembler exists. | Cannot assess |
| Hardware migration | `CameraSource` isolates `MacCameraSource`; device/camera IDs are configurable and the core vision import does not load OpenCV. | Pass for capture/live subset; detector/publisher replacement unreviewed |

## Architecture confirmation

The available camera/live-stream and canonical-contract subset remains
detector-independent. Media bytes remain external, visual evidence remains
separate from household facts, and replacing the Mac camera source is an edge
adapter change.

The full production boundary is **not yet confirmed** because most of the stated
vertical workflow is absent. In particular, this review cannot substantiate
Mac-to-MicroDuck migration for detector/tracker/publisher behavior or stable
frontend consumption of actual backend responses.

## Bounded corrections made

- Added stable edge source health (`online`, `camera_disconnected`, `offline`) and
  `camera_read_failed` without exposing native exception details.
- Made edge stream startup failure close the capture thread and source.
- Added import/dependency guards proving canonical vision imports do not load
  OpenCV, Ultralytics, SurrealDB, household writing, retrieval, or edge modules.
- Added a guard proving importing the semantic API does not load the edge runtime.
- Added a dependency guard keeping OpenCV optional and Ultralytics out of the
  backend dependency set.

No detector, tracking, persistence, enrollment, artifact-delivery, frontend, or
embedding implementation was introduced during the review.

## Validation

- Full deterministic suite: **930 passed, 1 skipped**.
- Focused domain/port/architecture suite: **43 passed**.
- Edge suite: **8 passed, 1 skipped**; the skip is the opt-in physical Mac camera
  smoke test.
- `git diff --check`: passed.

The tests validate the available code and the new import guards. They do not
claim end-to-end visual correctness, YOLO accuracy, durable retention, successful
clip encoding, or enrollment behavior.

## Next recommended work

Do not start embedding recognition yet. First implement the missing vertical
slice through the existing seams:

1. edge detector adapter and tracker producing canonical `VisualObservation`;
2. retrying `ObservationSink` publisher and production ingestion adapter;
3. retention-aware observation/clip repositories and external artifact sink;
4. candidate promotion and transactional human enrollment service;
5. `/v1/vision` read/action DTOs and the minimal `/vision` page;
6. one fixture/replay-backed end-to-end test plus the Mac live demo;
7. rerun this architecture review against those concrete imports, records,
   network payloads, and write traces.

After that review confirms the boundary, the next Epic 2 step should be
embedding-based recognition and multi-view enrollment, using active enrollments
as the human authority described in `ENROLLMENT.md`.
