# Kinship after prior turns — 2026-09-08

## Confirmed reproduction

The reported sequence was reproduced on the formerly resident `qwen3.5:9b`:

```text
我父亲是谁          -> self / father
我岳父是谁          -> self / spouse / father_in_law       WRONG
我岳父是谁          -> self / spouse / father_in_law       WRONG
我老婆是谁          -> self / wife
我岳父的生日是哪天  -> self / father_in_law / birth_date
我岳父是谁          -> self / father_in_law
```

These are captured interpreter concept paths, not inferred from the rendered
answer. `father_in_law` already expands to `spouse -> parent(gender=male)`.
Adding `spouse` first therefore produces the observed extra traversal. The
ontology/executor is behaving consistently with the erroneous interpretation.
No private household data or production logs were needed for reproduction.

A synthetic deterministic test distinguishes the speaker's father from the
spouse's father and verifies birth-date selection. It also proves that explicit
`spouse + father_in_law` remains a valid, different request. Removing duplicate
relations downstream would break legitimate composition and was not done.

## Experiments and changing environment

The original baseline is the planner prompt at `bd161e0`; its exact source hash
matches the saved baseline payload. Candidate requests use identical schema,
clock, query histories and inference settings within each comparison, except for
the candidate's intentional instructions/examples/capability changes.

1. **9B initial attempt:** generic atomic-concept instructions plus maternal-
   grandfather examples did not fix the error. Expanded-subject checks stayed
   **39/48 -> 39/48**. The two repeated in-law errors and an extra spouse in a
   nested daughter/in-law case persisted in all three runs.
2. **Environment changed:** the user/Grok switched production to `qwen3.5:4b`;
   the original temporary container workspace was gone. No switch back to 9B was
   attempted. Grok also committed an anti-history-carryover instruction and
   father-in-law/wife examples in `d9f8759`.
3. **4B stacked candidate:** compared the original baseline with the then-current
   combination of Grok's work plus Codex's explicit `emit` capability description,
   generic paragraph and two mother-in-law examples. Scores were **54/57 ->
   51/57**. This experiment does not isolate which individual addition caused
   the regression. It is not acceptable as a validated improvement.
4. **Final local change:** remove only Codex's unsuccessful experimental
   paragraph/examples/`emit` metadata. Preserve Grok's instruction, examples and
   tests. Re-test that final variant against the same original baseline on 4B.

On 4B, even the original baseline passed the original conversation, the isolated
`我父亲是谁 -> 我岳父是谁` pair, and standalone `我岳父是谁` three times each.
That is not evidence that the failed prompt experiments fixed 9B. Model and
prompt effects are reported separately. The 9B second candidate was not measured
before the model switch, so its 9B behavior is unknown.

## Scoring and scope

Each 4B variant has 19 synthetic requests, repeated three times with alternating
variant order and one discarded warmup per variant/pass. The requests include
the supplied eight-turn sequence, a minimal two-turn pair and nine diagnostic
standalone/compositional cases. Chinese/English in-law synonyms, birth-date
selection and legitimate outer possession are included. The `heldout-*` IDs are
retained for reproducibility; after inspection these are diagnostic cases, not
an untouched generalization acceptance set.

`correct` means the full expanded `SemanticReference` matches the expected
reference (including anchor, every relation, filters and their order), the
response declares a fact request, and the request passes schema validation.
It is **not** a complete-plan, live-answer, or production-accuracy score. Other
operation/property distinctions are covered by the deterministic suite; these
numbers must not be relabeled as answer accuracy.

Direct interpreter requests use the real planner message builder and output
schema with synthetic capabilities, 8192 context, 24h keep-alive, temperature/
seed 0, thinking disabled and a 384-token ceiling. They exclude validation retry,
API/WebUI persistence, resolver/executor latency and private speaker bindings.
Only user history is forwarded, with the existing assistant boundary markers.
This reproduces the model-facing history shape, not a live browser session.

All model calls ran on the already resident model. Before/after snapshots record
Ollama version, model digest and context. Payload hashes and the final local
source/schema/fixture manifest are retained in the summaries. No deployment,
model unload, setting change or household data mutation was performed by this
investigation. Local deterministic suite: **464 passed**.

## Reproduction

In separate baseline and candidate source checkouts, prepare payloads using the
same current probe script and model name:

```sh
PYTHONPATH=src .venv/bin/python scripts/kinship_context_probe.py prepare \
  --model qwen3.5:4b --output /tmp/kinship-baseline.json
# Repeat from the candidate checkout, writing kinship-candidate.json.
```

Stage those two payload files and the script into an isolated GPU directory;
run only while that exact model is already resident at 8192:

```sh
python kinship_context_probe.py run --root /tmp/hc-kinship-final
```

Back in a local checkout:

```sh
PYTHONPATH=src .venv/bin/python scripts/kinship_context_probe.py score \
  --results /tmp/kinship-final-results.json --output /tmp/kinship-final-summary.json
```

Baseline payloads in this investigation were also checked against the saved
`bd161e0` static message prefix; dynamic messages came from the matched candidate
case. The model name was changed to 4B in **both** variants. Full request/response
files stay in `/tmp`; checked-in summaries contain scores and erroneous paths,
not complete prompts or household answers.

## Final 4B result and remaining limitation

| Variant | Exact expanded-subject checks | Mean interpreter wall |
|---|---:|---:|
| Original baseline | 54/57 | 1,043 ms |
| Grok changes preserved; Codex experiment removed | 54/57 | 1,049 ms |

The original eight-turn sequence is **24/24** in both variants. The minimal
father-then-in-law sequence is **6/6**, and standalone in-law is **3/3** in both.
The final variant therefore does not demonstrate a prompt-induced accuracy or
latency improvement over the original on 4B. No 9B improvement is claimed.

One diagnostic remains wrong in all three repetitions: `我的妻子的父亲是哪位？`.
The baseline emits `spouse -> father`, losing the wife's female constraint.
The preserved-Grok final variant emits `wife -> father_in_law`, adding an
extra spouse traversal and potentially returning a different person. Equal
aggregate scores do **not** mean these errors are equivalent or equally severe.
This remains an unresolved composition error; the original reported sequence
passing on 4B must not be generalized to all in-law phrasing.

The stacked candidate additionally misinterpreted `我母亲的外公是哪位？` as
`mother -> paternal_grandfather`; removing Codex's experimental changes restores
`mother -> maternal_grandfather` on this diagnostic. The final tree preserves
Grok's changes rather than silently reverting his commit. It does not claim that
those changes are universally beneficial. Further prompt work needs a fresh,
untouched composition set and the complete real-LLM contract suite before a
broad kinship-accuracy release claim.

For a continued live failure on 4B, the next comparison should capture the exact
model-facing request (with permission if it contains private conversation text)
and compare its model digest, capability/prompt hash and user-turn sequence with
this isolated reproduction. Do not change household graph facts to compensate
for an incorrect semantic path.
