# Compact transport candidate — local review and Grok handoff

> **WITHDRAWN FROM SERVING.** The user reported universally negative responses
> after enabling compact transport. Ollama and OpenRouter now use expanded JSON
> as their sole serving format. The codec and archive remain offline research
> artifacts; do not deploy `candidate.tar.gz`. Earlier token/test results below
> do not demonstrate real-model correctness. A production-host synthetic
> reproduction now confirms invented filters and semantic validation failures;
> see `artifacts/compact-transport/production-diagnosis/REPORT.md`.

## Observations

The serving clients now use a versioned, schema-derived positional codec. Internal
semantic types, concept expansion, executor and resolution ownership are unchanged.
The offline profile uses synthetic fixture metadata and reusable authored examples;
no runtime household graph or production service was accessed.

`profile-summary.json` compares expanded and compact representations with identical
semantic examples (including explicit `eq` defaults). Token counts use
**cl100k_base as a proxy**, excluding provider chat templates and hidden grammar
compilation. They are not Ollama/Qwen token counts.

| Component | Expanded tokens | Compact tokens | Reduction |
|---|---:|---:|---:|
| Capabilities | 1,993 | 1,985 | 0.4% |
| Output schema | 1,820 | 1,782 | 2.1% |
| Assistant examples / output representations | 1,971 | 1,539 | 21.9% |
| Structured content including schema, before format instructions | 5,784 | 5,306 | 8.3% |
| Structured content including format instructions | 5,784 | 5,545 | 4.1% |
| Total message content | 7,462 | 7,261 | 2.7% |

Format-instruction overhead is included in total message content and the net
structured row; component-only reductions must not be presented as net savings.

All 37 synthetic demonstration outputs validate and round-trip on the complete
synthetic contract catalog. A representative age-filter output drops from 71 to
52 proxy tokens. Per-output bytes/tokens and fingerprints are in the summary.

Before editing, the static-test catalog capture measured 8,833 capability bytes,
8,263 schema bytes and 33,421 serialized message bytes. Its frozen comparison is
`baseline-summary.json`, tied to the original revision. That smaller catalog lacks
some demonstration properties; its parse failures are catalog rejections, not
production measurements. The complete synthetic profile above is the acceptance
reproducer. Both manifests record candidate source and schema hashes; the frozen
summary separately identifies the original baseline revision.

Local strict decoding measured roughly 0.35 ms versus roughly 0.003 ms for plain
JSON parsing. Validation/codec CPU overhead increased; there is no claim of a
serialization-speed win. A first field-alias-only prototype increased total
message tokens (7,511 to 7,941) and the representative output (71 to 78), so it was
replaced by positional required fields. Capability tabulation alone barely saves
tokens. The dominant demonstrated gain is model-output/example representation.

Validation: **595 deterministic tests passed**, including cross-process runs under
four hash seeds, typed IR round trips, complete concept examples, malformed output,
unknown vocabulary, per-attempt retry metrics, and existing conversation/isolation
regressions. No benchmark labels or scoring rules changed. Test model stubs were
updated to emit the advertised transport. Forbidden IDs now fail one boundary
earlier; the corresponding wire test expects schema rejection.

## Inference and limitations

Output generation may become faster, but local token counts do not establish that.
Overall prompt savings are modest because unchanged natural-language instructions
and capability explanations dominate. Neither production accuracy, actual token
savings, constrained-decoder support, retry rate nor end-to-end latency has been
established. This is a candidate for profiling, not production acceptance.

## Reproduce

From an isolated checkout/package with dependencies installed:

```sh
python -m pytest -q
PYTHONPATH=src python scripts/profile_semantic_transport.py --output /tmp/transport-summary.json
```

The profiling tokenizer is a development dependency (`pip install -e '.[dev]'`).
For the deployed tokenizer, supply `--tokenizer-json /path/to/tokenizer.json` and
install its optional `tokenizers` reader. The script records its hash. It performs
no model calls. The frozen pre-edit comparison used `--baseline-snapshot` on a
synthetic capture; reproduce the original builder from the revision named in the
summary if regenerating that capture. Do not substitute runtime `data/` records.

## Grok production handoff

The source archive contains the Python package, schemas, transport test/reproducer,
static test fixtures and synthetic contract fixture. Run its focused test with
`python -m pytest tests/test_semantic_transport.py -q`; the full 595-test run uses
the repository checkout. It contains no runtime `data/`, credentials or environment
files. The archive SHA-256 and member hashes are in `handoff-summary.json`.

Use the supplied candidate source archive/manifest with the existing isolated
frozen evaluation package. Keep its labels, history loader, schema/data package,
model digest, context size, seed and output-token limit fixed between the original
revision and candidate. Do not deploy over the serving installation to profile.

1. Record package, ontology, effective schema/dictionary, model/digest, Ollama
   version, `num_ctx` and fingerprints. Verify `prefixItems` array constraints on
   the actual provider before scored runs; report decoder incompatibility as a
   failure, without falling back to expanded JSON or changing the grammar.
2. Run paired warm and first-call samples. Score complete canonical plans and
   expected populations with frozen labels, preserving all user-turn history.
   Include scalar/each, both comparison operands, relationship properties,
   intermediate filters, exclusion, missing/ambiguous and discourse cases.
3. Report first-pass accuracy, retries, malformed/unknown-symbol/schema failures,
   call counts, actual prompt/output tokens, and median/p95 end-to-end latency.
   Sum per-attempt usage from `planner_diagnostics.transport.attempts`; top-level
   usage retains its existing final-attempt meaning. Separate wrong-but-legal
   interpretations from codec failures and execution failures.
4. Keep outputs synthetic or sanitized. Preserve all counterexamples. Report
   architecture changes, interpretation accuracy changes and evaluation corrections
   separately. Production acceptance remains blocked on this report, especially
   retry-rate non-regression and constrained-decoder fidelity.

No prompt tuning, ontology edits, scoring relaxation or semantic repair is part of
this candidate. Proposed follow-up decisions must use the production evidence.
The full wire specification is `docs/design/compact-semantic-transport.md`.
