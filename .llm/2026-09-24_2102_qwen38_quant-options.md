Date: 2026-09-24 21:02 PDT
Type: investigation
Status: completed

## Objective

Assess whether another Qwen3.8 27B quantization is worth testing for Home Cortex.

## Context

The retained `artifacts/qwen3.8-27b.yaml` full run used Q4_K_M, digest `22130167c4c2…`, 16,384 requested context, and Ollama 0.34.2. It scored 111/119 planner but 4/8 rejection, at 14.757 s p50 and 57.744 s p95. Historical hardware evidence only establishes an RTX 2060 SUPER with 8 GiB for earlier runs; the 27B export lacks hardware/offload telemetry.

## Findings

- The recorded digest matches Ollama's explicit `qwen3.8:27b-mtp-q4_K_M` tag. The current `qwen3.8:27b` listing uses a different digest, so the unqualified tag is unsafe for reproducing this run.
- Ollama publishes a separate standard `qwen3.8:27b-q4_K_M` tag at about 18 GB and Q8 at about 30 GB. Neither reduces memory use versus the tested Q4.
- Unsloth lists GGUF variants around 6.19 GB at UD-IQ1_S, 7.27 GB at UD-IQ2_XXS, 12 GB at UD-IQ3_S, 14.3 GB at UD-IQ4_XS, 16.5 GB at UD-Q4_K_M, and 19.8 GB at UD-Q5_K_M. These are file sizes, not total runtime VRAM.
- At 8 GiB and 16K context, the only plausible full-GPU variant is an extreme 1-bit quant, with significant quality uncertainty. On larger-memory GPUs, IQ3_S or Q5_K_M are more informative candidates for the size/accuracy tradeoff.

## Decisions

Recommend a same-size standard Q4 versus saved MTP Q4 diagnostic to separate packaging/MTP effects, and only test lower-bit 27B for speed if the available GPU memory justifies it. Preserve the rejection gate; higher quantization alone has no established ability to fix the four misses.

## Changes

No source or benchmark artifact changes were made.

## Validation

Read saved benchmark export and current publisher tag/model pages. No live model test or new benchmark was run.

## Remaining Issues

Exact 27B host GPU, RAM, offload split, and runtime memory use are absent from the export. Home Cortex accuracy of alternative quants is unknown.

## Recommended Next Step

Record host memory/offload telemetry and pin exact model digests before a targeted rejection/planner run on a second quantization.
