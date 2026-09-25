# llama.cpp migration evaluation — 2026-09-24 PDT

**Decision: do not promote.** Production on `home-cortex-0` remains on Ollama
0.34.4. The llama.cpp candidate failed the standard benchmark's multi-intent
safety gate. After the evaluation, the Ollama container was restarted, the
original Compose configuration was restored, and an authenticated Home Cortex
`/v1/chat` request returned HTTP 200 with an answer.

## Architecture evaluated

| Current production | Candidate |
| --- | --- |
| `home-cortex → Ollama provider → Ollama` | `home-cortex → OpenAI-compatible provider → llama-server → llama.cpp CUDA` |

The candidate provider shares one OpenAI-compatible HTTP transport with the
OpenRouter adapter. It leaves the semantic planner, prompts, ontology, and
executor unchanged. `docker-compose.llamacpp.yml` is an **opt-in candidate**;
the default Compose stack still includes Ollama.

## Runtime and model

| Field | Ollama baseline | llama.cpp candidate |
| --- | --- | --- |
| Run ID | `20260925-045530-22d5` | `20260925-051856-6cda` |
| Runtime | Ollama 0.34.4 | llama.cpp `b11151`, commit `bd4f514db14d87fded667787a7a963bfbaa98e89` |
| Container | `ollama/ollama:0.34.4` | `ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:014f721265464f38ccb247c1338d07d852c4bae7509a4b4734d07a2bbadc765c` |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | `qwen35-abliterated-9b-q4km` |
| Artifact | Ollama digest `92a443adb124f5e805bbdee23fdb38fcd22a7bf00a1016b53f764e741369c600` | `Huihui-Qwen3.5-9B-abliterated-Q4_K_M.gguf`, SHA-256 `bb30f918bfeb0b3141187f588baaca7faf554cb8565fc6c2869ec032d8f54134`, 5,627,044,800 bytes |
| Quantization | Q4_K_M | Q4_K_M |
| Context | 16,384 requested | 16,384 server context, one slot |

The candidate GGUF is from `Abiray/Huihui-Qwen3.5-9B-abliterated-GGUF` at
Hugging Face revision `dabf4e93cd78ecc2664fb2fbe33253707cee183d`.
The exact Ollama blob failed to load in upstream llama.cpp with
`qwen35.rope.dimension_sections` length 3 instead of 4. The candidate uses
a separate conversion of the same underlying abliterated 9B model; its weights
and chat template cannot be asserted identical to the Ollama blob. The
candidate server used the `chatml` template because the GGUF's bundled template
rejected the planner's interleaved system messages. The benchmark prompt,
corpus, and configuration fingerprints matched; both runs used the same host
and Git commit `66e33d1d7a5696c2482639bac35fb02497bbcbe1`, with dirty
working trees.

The Ollama GGUF loader mismatch is tracked in
[upstream llama.cpp PR #25334](https://github.com/ggml-org/llama.cpp/pull/25334).

## Standard benchmark

| Metric | Ollama | llama.cpp |
| --- | ---: | ---: |
| Plan correctness | 86/119 | 95/119 |
| Mutation classification | 17/21 | 16/21 |
| Mutation payload | 11/14 | 11/14 |
| Preview correctness | 3/5 | 3/5 |
| Commit correctness | 8/9 | 8/9 |
| Rejection correctness | 8/8 | 8/8 |
| Multi-intent handling | 4/7 | 3/7 |
| Partial multi-intent plans | 3 | 4 |
| Validation failures | 6 | 8 |
| Malformed structured output | 0 | 0 |
| P50 latency | 1.451 s | 1.079 s |
| P95 latency | 3.300 s | 2.314 s |

Both runs failed the absolute partial-plan safety gate. The candidate also
failed the comparison's multi-intent and partial-plan regression gates
(comparison exit code 3). Answer correctness was not scored by this suite.
The speed improvement does not outweigh the safety regression.

The llama.cpp model loaded in about 33.3 seconds. With `--n-gpu-layers 99`,
GPU memory in use was about 5,612 MiB of 8,192 MiB (about 2,173 MiB free);
the container used about 902 MiB host RAM during inference. One logged warm
request reported 71.55 prompt tokens/s on 12 uncached tokens and 55.63
generation tokens/s on 47 output tokens. Time to first token was not recorded.
These are samples, not distribution estimates. The Ollama baseline's first
request reported 13.59 seconds of model load time.

Small run records and summaries are next to this report. The full results and
per-case diagnostics remain on the GPU host under
`benchmarks/results/<run-id>/`; [comparison.txt](comparison.txt) records the
gate output.

## Validation and remaining work

The local deterministic suite passed (983 tests). A one-case structured-output
smoke passed through the llama.cpp planner. The full standard suite produced
no provider, timeout, or malformed-output failures. The candidate was **not**
deployed as the production runtime. The prior production API path was verified
after rollback.

Next, establish exact or stronger GGUF parity and fix the multi-intent safety
regression without changing semantic-layer behavior or relaxing validation.
Repeat the standard comparison before switching the default Compose stack.
