# First-answer latency: warm streamed HTTP baseline

**Run:** 2026-09-26 PDT, isolated synthetic API package on `home-cortex-0`.
The package used the production Ollama service (`0.34.4`, `qwen3.5:9b`,
`Q4_K_M`, RTX 2060 SUPER 8 GiB, model and container image digests in
[the summary](2026-09-26-summary.json)); it did not
connect to the household database or replace the deployed API. The fixed
English/Chinese fixture and source, schema, prompt, and dataset fingerprints
are in the summary. One warmup iteration per case was excluded. These are
receipt times at a loopback HTTP client, not browser paint times.

| Cohort | n | First byte P50 / P95 | First answer content P50 / P95 | Complete P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Fact | 9 | 42 / 43 ms | 1,571 / 1,690 ms | 1,571 / 1,691 ms |
| Mutation preview | 3 | 43 / 43 ms | 1,888 / 1,891 ms | 1,888 / 1,892 ms |
| Multi-intent | 3 | 42 / 43 ms | 1,006 / 1,026 ms | 1,006 / 1,027 ms |
| Ordinary steward chat | 3 | 42 / 42 ms | 1,889 / 1,927 ms | 2,174 / 2,201 ms |
| Bare model chat | 3 | 43 / 43 ms | 204 / 205 ms | 373 / 375 ms |
| Discourse turns | 6 | 42 / 44 ms | 2,263 / 2,699 ms | 2,264 / 2,700 ms |
| Two simultaneous fact requests | 6 | 47 / 58 ms | 1,856 / 2,495 ms | 1,856 / 2,495 ms |

All 33 measured streams had nonempty content and no HTTP or SSE error.
This probe **does not score answer correctness**. In particular, the
multi-intent case entered the fact branch in all three measured repetitions;
the earlier partial-plan safety failure remains an open promotion gate.

The fact waterfall has a 1,568 ms median planner stage, including a 1,541 ms
model call. Ollama reports median 516 ms prefill and 884 ms generation across
those calls. Synthetic identity resolution, message append, executor, renderer,
and SSE setup are small by comparison. Ordinary steward chat makes a second
model call after the planner; all three measured cases did so. Three of six
discourse turns required more than one semantic model call. The immediate SSE
role frame is not counted as answer text.

The model was already resident at the requested 16,384 context. The new
warmup manager verified that residency in about 14 ms and sent **zero**
synthetic inference requests in this run. That is a residency-check time,
**not** cold startup or warmup benefit. No production model unload or restart
was performed. Three cold boots, first-user contention during warmup, browser
paint, and ingress/proxy timing remain unmeasured. The 300/800 ms stretch
target was not met for steward routes on this warm model; the current
structured interpretation call alone exceeds it.

The change implemented here adds background local-model warmup, residency
rechecks, a bearer-protected readiness route, and opt-in browser/server
timing. It does not change semantic planning or promote a different model.
Next: run cold boots in an isolated production-like model service, measure
the first request while warming, and compare faster model/runtime candidates
against the standard semantic and multi-intent safety suite before rollout.
