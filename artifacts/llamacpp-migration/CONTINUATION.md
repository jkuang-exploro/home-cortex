# llama.cpp parity follow-up — 2026-09-25 PDT

**Decision: keep Ollama in production.** The exact production GGUF can load on
a pinned llama.cpp CUDA branch, and its mutation results match Ollama case for
case. Both runtimes still produce three partial plans for mixed requests, so
the required absolute safety gate fails. The full standard comparison and
production promotion were stopped at that gate.

## Controlled model checks

The fixed seven mixed requests were run on `home-cortex-0` without changing
prompts, ontology, corpus, or mutation semantics:

| Model package and runtime | Mixed rejections | Partial plans |
| --- | ---: | ---: |
| Production Huihui abliterated 9B package, Ollama baseline | 4/7 | 3 |
| Same production GGUF blob, patched llama.cpp CUDA | 4/7 | 3 |
| Separate abliterated Q4_K_M GGUF imported into Ollama | 2/7 | 5 |
| Separate abliterated Q4_K_M GGUF, upstream llama.cpp | 3/7 | 4 |
| Existing base `qwen3.5:9b` Ollama package | 7/7 | 0 |
| Separate base Q4_K_M GGUF imported into Ollama | 6/7 | 1 |
| Same separate base Q4_K_M GGUF, upstream llama.cpp | 6/7 | 1 |

The Ollama imports used the Qwen3.5 renderer/parser and the same package
sampling parameters as the production model. The separate GGUFs have different
hashes and conversions, so their results do not isolate a runtime effect. The
base GGUF came from `bartowski/Qwen_Qwen3.5-9B-GGUF` revision
`b8d8d7cea4ac7388a497614c4ea3d720712b2475`, SHA-256
`9437f5bf0dd0c97800caaf902f41e6a6aa00223ab232f159eda41dcbbb492645`.
Its full llama.cpp mutation suite also had one malformed structured output.

## Exact production blob on llama.cpp

The upstream CUDA image still cannot load Ollama's exact Qwen3.5 GGUF. A
local image built from [draft loader PR #25334](https://github.com/ggml-org/llama.cpp/pull/25334),
commit `e4ee2d21a8ceae24aecdc7c3a229a48ad3b34ee5`, loaded it at 16,384
context with 99 GPU layers and one slot. The image ID was
`sha256:ce56116c1150900ef8378e81c7a30300a0aed29a6ffc8abb9b7a5784d3833ba3`.
The GGUF SHA-256 was
`afb54ad43a39f947407f5cabc59856348d70e072baa5c62d436332157c151bcd`
(6,594,462,816 bytes, Q4_K_M). GPU use after load was about 5,502 MiB of
8,192 MiB. This draft branch is an evaluation candidate, not a production
runtime selection.

The isolated, provenance-correct mutation run is `20260925-162253-2e15`.
It recorded 17/21 classification, 11/14 payload, 3/5 preview, 8/9 commit,
8/8 rejection, and 4/7 multi-intent correctness, with three partial plans
and no malformed output. The seven mixed outcomes are identical to the
production Ollama baseline `20260925-045530-22d5`. Its safety exit code was
3. The [run record](llama-exact-blob-valid-run.json) and
[summary](llama-exact-blob-valid-summary.json) include the active image,
commit, model checksum, and configuration. The source checkout was a detached
evaluation worktree with only the benchmark provenance correction applied;
prompt and mutation corpus fingerprints matched the baseline.

## Provenance correction and production state

The first follow-up diagnostic runs incorrectly recorded the default image
and omitted the model checksum because the harness read `.env` values rather
than the running container. `llamacpp_metadata` now inspects the container
serving the benchmark URL, its command and model mount, then hashes the active
GGUF. Those earlier run records were not retained as acceptance evidence.

After evaluation, `llama-server` was stopped and removed, Ollama was restarted,
the production checkout remained clean, `/health` returned OK, and an
authenticated `/v1/chat` request returned HTTP 200 with an answer. The default
Compose stack still uses Ollama.
