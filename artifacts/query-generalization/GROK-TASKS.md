# Grok handoff: evidence and generic-query coverage

These tasks are prepared for Grok; they have not been dispatched through an agent
connection. Keep prompts, ontology, executor and frozen golds unchanged during
baseline measurement. No production writes or deployment.

## G1 — Deployed profile and capability inventory (P0)

Host: `jkuang@home-cortex-0` over Tailscale. Use applicable authorization for
read-only inspection; begin with code/config fingerprints and schema metadata,
not a household-wide record dump. Exclude credentials from every output.

Record source revision/file hashes, active ontology version/hash, model/digest,
Ollama version, num_ctx and effective generated capabilities/schema. Verify whether
adult AND minor is rejected by the actual serving process. Do not assume the
presence of an ontology-v2.yaml file means it is active.

For person, address, space, item and any room/product category, report available
semantic bindings, physical edge signatures, supported direction, ownership/scope,
cardinality, temporal behavior and missing capability mappings. Establish whether
room-to-home containment and item locations are actually represented. Report
metadata/aggregate evidence first; flag any need for additional private records.
Do not add guessed containment relationships.

Deliver REPORT.md plus a small capability/fingerprint summary and a gap matrix:
existing data + missing declaration / missing binding / missing data / genuinely
unexpressible operation.

## G2 — Synthetic execution counterexamples (P0)

Build fixtures, not question handlers. Reproduce scalar age + minor filtering,
then compare explicit each projection, exactly-one filtered result, zero results,
missing dates, ambiguous named roots and ambiguous final sets. Record the full
plan, result status, candidate IDs, row IDs and expected population.

Add multi-edge and >25-result cases for count/select; verify completeness and
whether filters on separate edges retain existing existential semantics. Add two
households containing same-named items to test scope before name resolution.
Do not change executor behavior or loosen scoring. Deliver failing tests/probes
for Codex to review, with the smallest reproducer for each issue.

## G3 — Frozen interpreter evaluation across domains (P1; after G1)

Use the existing frozen contract package and identical V1/V2 conditions. Run the
original transcript and held-out synthetic counterparts, but keep private records
out of the package. Score full plans and expected populations, not found/count
alone. Report wrong-but-legal interpretations separately from schema rejection,
missing knowledge and execution failures.

Include location, verified containment, item categories, members, age filters,
plural projections, nested relatives, ambiguous/missing locations and unsupported
questions. Do not feed sequence YAML through a loader that discards history.
Record first-pass success, retries, decoder errors, call counts and latency.
No prompt tuning, ontology edits or changing frozen labels during a scored run.

## G4 — Presentation fidelity matrix (P1)

Specify paired normal-language and detailed descriptions from validated plans and
synthetic results. Cover gender only, adult+gender, minor+gender, zero, intermediate
filters, edge ownership, exclusions, nested paths, missing data and incomplete
collections. Mark every effective condition and its attachment point.

Propose reusable grammar fragments/metadata, not one complete answer template per
question. Do not change numerical values or add facts. Codex reviews the display
contract and implements the shared composer.

## Return format

Separate observations, inference and proposed fixes. Include commands, synthetic
fixtures, source/schema/data fingerprints and compact summaries. Preserve all
counterexamples. Codex owns the scope/cardinality/compiler design and final change
review; do not spend tokens rewriting those boundaries independently.
