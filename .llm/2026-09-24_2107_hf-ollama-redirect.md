Date: 2026-09-24 21:07 PDT
Type: investigation
Status: completed

## Objective

Diagnose `ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` failing with `blocked redirect to a different host` inside Docker Compose.

## Context

The user supplied the pull error. The tracked `docker/docker-compose.yml` pins `ollama/ollama:0.32.15`, while retained benchmark exports report that the GPU host ran Ollama 0.34.2; the active remote container version was not queried here.

## Findings

- Ollama issue #18526 reproduces the same Hugging Face Xet CDN cross-host redirect failure for an Unsloth Qwen3.8 GGUF pull on 0.34.2. It reports `--insecure` as a workaround.
- Ollama PR #18533 merged a redirect allowlist fix. The v0.34.3 `server/images.go` source includes the allowlist for `hf.co` and `huggingface.co` subdomains.
- The reported URL shows Ollama reached Hugging Face and rejected the CDN redirect during a HEAD request; the model ID and quant file exist in the Unsloth repository.

## Decisions

Recommend confirming the running container version. Use `--insecure` only as a one-time pull workaround if an immediate download is needed; otherwise upgrade the Ollama container to 0.34.3 or later and preserve the named model volume. Treat subsequent benchmark results as a different runtime cohort.

## Changes

No application, Docker, or benchmark files were changed.

## Validation

Read repository Compose definitions and upstream Ollama issue, merged PR, and v0.34.3 source. No remote container commands or network download were run.

## Remaining Issues

The active host image/version and network connectivity after the redirect are unverified.

## Recommended Next Step

On the GPU host, run `docker compose -p cortex exec ollama ollama --version` and retry the pull with the chosen workaround or upgraded image.
