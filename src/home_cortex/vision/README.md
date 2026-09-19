# Paused Vision backend contracts

Vision implementation is paused pending MicroDuck hardware. Home Cortex starts
without a camera, detector, media process, model weights, or
`home_cortex_client`. This package retains only normalized backend evidence
contracts, validation, ingestion ports, and artifact storage abstractions.

Device capture and execution belong to the independent sibling project
`home_cortex_client`. The repositories communicate through serialized records;
neither imports the other's Python implementation.

## Ownership

| Responsibility | `home_cortex_client` | `home_cortex` |
|---|---|---|
| Camera capture, timestamps, raw frames | Owns | Never accesses devices |
| Detector, tracking, deduplication | Owns | Never receives native output |
| Observation ID and retry spool | Owns | Validates and ingests records |
| Rolling buffer and clip encoding | Owns | Validates clip metadata |
| Artifact bytes | Uploads | Stores/resolves through `ArtifactSink` |
| Candidate lifecycle and enrollment | May propose hypotheses | Owns durable state and human authority |
| Household reconciliation | Never mutates facts | Owns explicit mutations and history |

The backend owns these canonical concepts:

- `VisualObservation`: immutable evidence reported for one image region;
- `VisualCandidate`: durable evidence anchor for a possible physical instance;
- `VisualEnrollment`: auditable human confirmation linking a candidate to an
  existing household item;
- `EvidenceClip`: metadata for a pending, available, or failed clip;
- `ArtifactReference`: content-addressed external-media metadata;
- `ProducerProfile`: immutable pipeline/model provenance.

`VisualTrack` remains ephemeral producer state. A serialized `TrackReference`
may annotate an observation, but backend correctness never depends on finding a
live tracker.

## Semantic invariants

1. Detector tensors, pixel boxes, model class IDs, tracker objects, and embedding
   vectors do not cross the producer boundary.
2. Category belief, visual candidate, household item, observation, and enrollment
   are distinct values with distinct authority.
3. An observation is evidence. It does not create an item, assert a household
   location, or confirm identity.
4. Only an active human-created enrollment confirms candidate-to-item identity.
   Machine `item_hypotheses` remain scored evidence and cite their supporting
   enrollment records.
5. Missing spatial state is null/absent. It is never a dummy space or zero pose.
6. Spatial values reuse backend `RuntimePose`, `Position`, and uncertainty
   contracts. Every object-position estimate names its `space:` frame.
7. Raw media and embeddings never enter household graph records. References do
   not reveal filesystem paths, buckets, or device URLs.
8. Vision records remain outside the household semantic ontology. Reconciliation
   is a separate authenticated application operation.

## Current transport seam

The implemented backend seam is transport-neutral:

```text
serialized VisualObservation mapping
    -> parse_visual_observation
    -> ObservationIngestionService.ingest
    -> ObservationRepository.put_if_absent
    -> ObservationReceipt
```

JSONL replay uses `parse_observations_jsonl` and the identical ingestion service.
Delivery semantics are at-least-once with an idempotent decision keyed by stable
observation ID and canonical content hash:

- absent ID: `accepted`;
- same ID and content: `duplicate`;
- same ID with changed content: `conflict`.

No production Vision HTTP endpoint exists yet. When work resumes, a thin,
authenticated, bounded JSON/NDJSON adapter may call this service. Do not add a
message bus or import client classes into the backend.

`EvidenceClip` transitions are independently validated:

```text
pending -> available
pending -> failed
```

Artifact bytes must be committed before `available` metadata. Clip failure never
invalidates an observation.

## Enrollment authority

Enrollment links exactly one persistent candidate to one existing `item:` and
records the confirming person, time, and supporting observations. It never
creates the item. Missing items must first use the normal household mutation
boundary. At most one active enrollment may identify a candidate; merges and
revocations must preserve auditable history.

## Browser boundary

`home_gui` may eventually consume backend observation/evidence DTOs and a
browser-compatible stream advertisement. It must not import
`home_cortex_client`, backend persistence code, detector output, credentials,
filesystem paths, or raw RTSP endpoints. If MicroDuck requires a media gateway,
that gateway remains on the media path rather than inside semantic API execution.

No Vision route names, UI components, media gateway, detector, tracker, clip
worker, persistence repository, or MicroDuck adapter are committed while the
feature is paused. Git history retains the deleted Tapo, relay, and early UI
experiments; they are not production architecture.
