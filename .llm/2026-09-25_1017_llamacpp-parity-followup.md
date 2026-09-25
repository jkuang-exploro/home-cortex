Date: 2026-09-25 10:17 PDT
Type: benchmark
Status: partial

## Objective

Continue the Ollama-to-llama.cpp migration by isolating the mixed-intent
regression and testing exact production-model parity before deployment.

## Context

The prior upstream llama.cpp run used a separately converted Huihui GGUF and
failed the multi-intent comparison. Production used Ollama 0.34.4 and the
`huihui_ai/qwen3.5-abliterated:9b` package.

## Findings

- Importing the separate abliterated GGUF into Ollama gave 2/7 mixed-intent
  correctness, versus 3/7 through upstream llama.cpp; the current Ollama
  package baseline gave 4/7. The package conversion matters.
- A separately converted base Qwen3.5 9B GGUF gave 6/7 mixed-intent
  correctness through both Ollama and upstream llama.cpp. The latter also
  produced one malformed structured output in the full mutation suite.
- A CUDA image pinned to draft llama.cpp loader PR #25334 commit
  `e4ee2d21a8ceae24aecdc7c3a229a48ad3b34ee5` loaded the exact production
  GGUF. Its mutation run `20260925-162253-2e15` matched the Ollama baseline
  on all seven mixed cases: 4/7 correct and three partial plans. The absolute
  safety gate failed with exit code 3.
- Benchmark provenance initially misidentified the custom image because it
  read deployment defaults. The final run correctly captured the active
  container image, commit, model hash, context and GPU settings.

## Decisions

Do not promote llama.cpp. The absolute partial multi-intent plan gate still
fails, including with the exact production weights. Do not run a full standard
comparison after the required mutation gate has failed.

## Changes

- Changed `llamacpp_metadata` to discover the container serving the benchmark
  URL and resolve its mounted model file; added a regression test.
- Stored the valid run and summary plus analysis in
  `artifacts/llamacpp-migration/CONTINUATION.md`.
- No production code or Compose configuration was changed. The detached GPU
  evaluation worktree contains only the provenance correction and copied graph
  input data. The unrelated untracked repository audit was left untouched.
- Removed temporary Ollama imports and duplicate base GGUF copies created for
  the comparison; retained the pinned evaluation image and run records.

## Validation

- Local deterministic suite: 985 passed; `git diff --check` passed.
- Exact production GGUF loaded on the pinned CUDA branch with 16,384 context,
  99 GPU layers, and about 5,502 MiB GPU memory in use.
- Final benchmark record contains GGUF SHA-256
  `afb54ad43a39f947407f5cabc59856348d70e072baa5c62d436332157c151bcd`
  and the correct custom image ID; prompt and corpus fingerprints matched the
  baseline.
- Production Ollama was restored; `/health` returned OK and authenticated
  `/v1/chat` returned HTTP 200 with an answer.

## Remaining Issues

The existing production model itself fails the absolute mixed-intent safety
gate. The only exact-blob loader is an unmerged draft branch. The normal
production stack still requires Ollama.

## Recommended Next Step

Select or prepare a same-file GGUF/runtime combination that passes the
mixed-intent gate, then run the full standard comparison. Promote only after
that comparison and full application integration pass.
