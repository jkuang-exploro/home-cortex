Date: 2026-09-24 22:28 PDT
Type: coding
Status: partial

## Objective

Migrate the default local inference runtime from Ollama to llama.cpp and deploy
after a comparable production benchmark.

## Context

Production host `home-cortex-0` ran Ollama 0.34.4 with
`huihui_ai/qwen3.5-abliterated:9b`. The user supplied a phased migration ticket
with a mandatory baseline, parity comparison, and semantic safety stop conditions.

## Findings

- Ollama baseline `20260925-045530-22d5`: plan 86/119, multi-intent 4/7,
  partial multi-intent plans 3, P50/P95 1.451/3.300 s. Its absolute safety
  gate already failed.
- The exact Ollama GGUF blob cannot load in pinned upstream llama.cpp build
  `b11151` because its Qwen3.5 rope metadata has 3 sections, while the loader
  expects 4. Upstream PR #25334 tracks the loader mismatch and other format
  differences. A separate Q4_K_M GGUF conversion of the same abliterated 9B
  model loaded on CUDA. Its bundled template rejected interleaved system
  messages; server-side ChatML handled the existing planner prompt.
- llama.cpp candidate `20260925-051856-6cda`: plan 95/119, multi-intent 3/7,
  partial plans 4, validation failures 8, P50/P95 1.079/2.314 s. The comparison
  exit code was 3 because multi-intent handling regressed and partial plans
  increased. Prompt, corpus, and config fingerprints matched. Model file hashes
  differed, limiting runtime-only attribution.

## Decisions

Do not promote or deploy the llama.cpp candidate. Keep the default Compose path
on Ollama and place llama.cpp in opt-in Compose files for further evaluation.
Preserve the shared OpenAI-compatible transport and benchmark provenance work
in the local working tree for review. Restore the production host's prior
Compose/source configuration and Ollama service.

## Changes

- Added a provider-neutral response IR, shared OpenAI-compatible HTTP transport,
  llama.cpp provider, configuration, and clean API transport errors.
- Added llama.cpp benchmark selection, readiness discovery, GGUF/image identity,
  historical Ollama compatibility, and runtime-aware comparison/reporting.
- Added opt-in `docker/docker-compose.llamacpp.yml`, GPU override, operations
  guide, and provider/deployment tests. Default Compose files remain unchanged.
- Stored small run records, summaries, comparison, and analysis in
  `artifacts/llamacpp-migration/`.

## Validation

- Local deterministic suite: `983 passed`.
- `git diff --check`: passed. Both default and candidate Compose configs parsed.
- Production candidate became healthy with 16,384 context, GPU use about
  5,612 MiB and about 2,173 MiB free; one structured planner smoke passed.
- Full standard Ollama baseline and llama.cpp parity runs completed; comparison
  failed the safety gate. See `artifacts/llamacpp-migration/REPORT.md`.
- After rollback, production Ollama/API containers were running, `/health`
  returned OK, and an authenticated Home Cortex chat returned HTTP 200.

## Remaining Issues

The candidate has a material multi-intent safety regression and uses a
different GGUF conversion. Time to first token was not measured. The normal
production stack still requires Ollama.

## Recommended Next Step

Investigate the four affected multi-intent cases and establish a GGUF/model
configuration with stronger parity without changing semantic-layer behavior.
Repeat the standard benchmark and deploy only after its safety gates pass.
