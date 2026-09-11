# Hosted space interpretation

Added declarative `hosted_space` and `host` reference concepts over existing
hosting relations. Space structure queries now have their own bilingual semantic
vocabulary, distinct from contents. Two redundant inventory demonstrations were
replaced by generic hosted-space list/count examples within the existing example
budget. No utterance-specific routing or executor repair was added.

The deterministic executor remains unchanged. Regression tests expand ontology
concepts and cover direct empty spaces, counts, nested space hosts, inverse hosts,
and explicit composition into contents, with collapse true/false/absent.
668 local tests passed. Composition fingerprints were refreshed for the ontology
change; benchmark cases and scoring were not changed.

GPU validation used qwen3.5:9b with an isolated package and explicit candidate
ontology, reading production facts without mutations. The four-question sequence
returned three fridge spaces (including the freezer), the existing grouped item
inventory, a space count of three, and the shelf's host. Independent English and
Chinese cabinet/desk/compartment probes selected the correct hosting direction;
English compartment counting used count, and shelf inventory retained contents.
These are targeted checks, not a broad accuracy measurement.

Final isolated fingerprints (sorted relative file names plus contents):
- Python package: d8ca14143978433adab642b1111d785f2f7a2d792cc0afdab648ca6dfa40a56a
- Candidate schemas: 8faf4cb2d73d8cbac9063c3266afddf0a57a2369473a4221d2a37c398ca0327a
- Source data JSON: aed59824250d3e1572166f32acc40ba9d09716d3d7f8755d5d040851eab1ee55

An initial probe accidentally fell back to the installed ontology and was rejected
as a validation setup error. The final probe explicitly loaded the candidate
ontology. An intermediate candidate also listed spaces for a count question;
the final list/count examples corrected that targeted regression.

## Live deployment

Rebuilt and deployed the API on 2026-09-11 UTC. Health returned HTTP 200.
Authenticated live requests returned all three fridge spaces (request
579dcebbc5e1422aa4f0e4a5d1eed077), including the freezer absent from inventory,
and the shelf's host (39fe2ca30c984b5d815dd67f2dd79d30).

Limitation: an independent live count question returned the correct space list
rather than its count (d0835040f1824a968fcb8222b5467cde), although counting passed
in the isolated multi-turn sequence. Count interpretation is not claimed stable.
The requested space enumeration and inclusion of empty spaces were verified live.
