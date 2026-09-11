# Item mutation routing

## Architecture

The steward previously exposed no write tool and sent every household utterance
through a fact-only compiler. Requests to remember an item could therefore become
contents queries. Simply enabling native tool calling also produced unsupported
claims of success without a tool call during isolated testing.

Mutation-enabled agents now first compile the current user turn with a small
structured mutation classifier. Non-mutations continue through the unchanged read
prompt, demonstrations, and read schema. The combined internal plan can carry a
named mutation intent, but cannot carry both a read and a write. Interpretation
and discourse replay never execute mutations. Only the current-turn agent handoff
calls the allowlisted `write_item` tool. Unplanned native write calls are blocked.

The name adapter resolves within server-configured household scope, allocates new
IDs server-side, and invokes the existing canonical transaction service. It does
not infer a container's internal destination, silently move a pre-existing item,
or expose IDs or raw graph state to the model. Creation at an already-recorded
location is a no-op. Preview remains non-persistent. Responses for planned writes
are rendered directly from tool outcomes, without a second freeform model answer.

Mutation-enabled ordinary reads incur one extra model call. Diagnostics include
that call and aggregate its reported token counts. Read-only agents and the fact
benchmark do not enable the classifier. Ollama and OpenRouter implement the same
classifier interface; real-model checks below used Ollama only.

## Deterministic validation

665 tests passed locally. Added coverage checks name-based create/preview/repeat/
move/delete, missing and ambiguous destinations, scope restriction, rejection of
model-supplied physical IDs, side-effect-free mutation interpretation/replay,
streaming dispatch once, deterministic preview wording, blocked unplanned writes,
mutually exclusive plans, and classifier call/token accounting.

## Real-model checks

All mutation tests ran in an isolated package on the GPU host, against invented
embedded SurrealDB data, using qwen3.5:9b and PYTHONHASHSEED=0. The target report
compiled item_name=香肠 and location_name=冰箱内部, returned APPLIED, and was visible
in a subsequent grouped contents query. Repetition returned NO_CHANGE; explicit
move and delete returned APPLIED. A preview left the item count unchanged. An
English recording request also applied. Query, quotation/translation, and
hypothetical controls made no mutation calls. Freeform hypothetical explanations
were not scored for prose accuracy; no broad accuracy rate is claimed.

The original four-turn containment sequence and additional compound-name probes
were separately rechecked read-only against production records. The final
candidate retained correct fridge grouping, kitchen scope, and shelf scope.
Earlier prompt-only and combined-prompt candidates were rejected during testing
because they misclassified items, skipped tool calls, or regressed read scope.

Read-check fingerprint (all Python files under isolated package root):
`215d80f94eb45737e8349eff9ebd8c58e90ce8a5d6eb6c422811d1736600dbce`.
Schema fingerprint: `b7a6fd4aa1ecefed58cb546146210c53354cc08bb0cd4e40b1f4a8664a86ccf6`.
Production source-data fingerprint: `aed59824250d3e1572166f32acc40ba9d09716d3d7f8755d5d040851eab1ee55`.
Invented mutation fixture fingerprint: `a0b65592d9d431f28c16f15710e263b85110ad2b09b3014a526169cbf41622e7`.
Hashes combine sorted relative paths and file contents, as recorded by the probes.

## Evaluation corrections

The internal plan gained named mutation fields. The experimental compact codec's
field dictionary therefore moved intentionally from version 1 to version 2.
Historical V1 counterexample bytes must now be rejected, not assigned new meanings.
Their tests re-encode the same captured semantics under V2 and retain the same
invalid-plan/unknown-property assertions. Production remains on expanded JSON;
this is an evaluation compatibility correction, not a claimed accuracy gain.

## Runtime data

The reported sausage command was tested in the isolated invented graph. It was
not committed into the user's production household as part of debugging. No
full graph ingestion or source-data rewrite is part of this change. Existing
snapshot-ingestion behavior remains unchanged; runtime mutations are database
state, not edits to the source JSON files.

## Deployment verification

Deployed the rebuilt API image to home-cortex-0 on 2026-09-10. Health returned
HTTP 200. An authenticated live HTTP preview returned “预览：香肠位于冰箱内部
（记录）；尚未保存。” (request fb55f2208086468fbfe2db53c4dbcf5b).
The production sausage alias count remained zero before and after the request.
A subsequent live fridge query retained grouped shelf/interior results with
cheese and bread/milk (request 79514933339b415491e2315f6811cfef).
No production commit mutation was performed for this verification.
