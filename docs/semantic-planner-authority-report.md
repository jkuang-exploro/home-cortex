# Semantic planner authority report

Date: 2026-09-04

This report closes the investigation in “Make the 9B Semantic Planner
Authoritative and Reduce Tier-0 Natural-Language Hardcoding.” The architecture
now treats the semantic planner as the open-world language interpreter and
Tier 0 as a removable latency optimization.

## Measurement method

- Planner-only dataset: 111 utterances, 26 canonical plans, and 9 capability
  categories in `benchmarks/semantic_planner_eval.yaml`.
- Required canonical and adversarial Chinese queries are included, but only six
  exact utterances are retained in Tier 0.
- The deterministic CI test uses an oracle semantic planner and executes all 111
  plans with Tier 0 disabled. It also verifies Tier-0/Planner semantic parity.
- The local real-model run used the available `qwen3:8b` model: 8.2B parameters,
  Q4_K_M quantization, temperature 0. The workstation did not have a separate
  9B model installed, so these numbers must not be represented as a 9B-model
  measurement.
- Real-model result: 109/111 (98.20%) after declared semantic-equivalence
  normalization; Planner P50 8,528.573 ms, P95 14,014.189 ms. The raw
  single-canonical-plan score was 108/111 (97.30%); one `display_name` identity
  plan was a declared valid alternative to `given_name`, leaving two filtering
  errors. A focused rerun resolved the ambiguous child-count case but retained
  the explicit-age over-planning error, so the report conservatively keeps the
  full-run score.
- Tier-0 JSON-graph baseline: 600 executions across all six retained utterances,
  P50 0.023 ms, P95 0.075 ms, and zero LLM calls.

## Required investigation answers

1. **Which successful queries depend exclusively on Tier 0?** None. Every
   retained exact Tier-0 utterance has a case in the Planner-only dataset. The
   deterministic parity suite reports 6/6 equivalent plans, and Tier 0 can be
   bypassed before parsing.

2. **Which still work with Tier 0 disabled?** All 111 benchmark cases execute
   through `SemanticFactPlanner -> SemanticFactRequest -> HouseholdFactEngine`
   in the deterministic suite, including all 17 ticket-minimum questions,
   marriage start/duration, birthday countdown, extrema, adult/minor counts,
   multi-hop in-law traversal, and speaker-relative daughter traversal.

3. **How accurately does the local planner map the benchmark?**
   98.20% semantic equivalence (109/111). The unadjudicated exact single-plan
   match was 97.30% (108/111).

4. **Which categories produce planner errors?** After semantic-equivalence
   adjudication, aggregation, entity reference, multi-hop kinship, property
   selection, relationship-property lookup, relationship traversal,
   speaker-relative reference, and temporal operation were 100%. Filtering was
   10/12 (83.33%) in the full run.

5. **What caused the failures?** One explicit “满十八岁” paraphrase caused the
   model to add unsupported pseudo-properties in addition to the declared
   `adult` predicate (`UNKNOWN_PROPERTY`). One unqualified “孩子” request was
   mapped to the speaker's graph children instead of current-household minors
   (`REFERENCE_MISMATCH`); it passed on the focused rerun, indicating model
   instability rather than missing capability. The accepted-plan executor
   exposed no missing IR primitive in this evaluation. Earlier
   development failures were traced to an oversized capability prompt, weak
   source/operation guidance, overly narrow evaluation labels, and one transient
   Ollama response error—not a need for more Python phrase handlers.

6. **Which Tier-0 parser functions were removed?** `_parse_reference`,
   `_starts_relation`, `_parse_comparison`, `_extract_property`,
   `_extract_birthday_intent`, `_strip_identity_syntax`, `_strip_count_syntax`,
   `_asks_identity`, `_asks_count`, `_asks_list`, `_mentions_household`,
   `_asks_household_children`, `_asks_residence_address`,
   `_household_age_extreme`, `_asks_marriage_date`, and `_self_residence`.
   The obsolete ontology `fast_path` marker and its unused property/reference
   prefix matching helpers were also removed.

7. **Which Tier-0 functions remain?** `TierZeroSemanticParser.parse`,
   `canonical_utterances`, generic text normalization, and the
   `_household_members` semantic-IR constructor. `parse` recognizes exactly
   `我是谁`, `Who am I?`, `你是谁`, `Who are you?`, `家里有几个人`, and
   `家里都有谁`; it has no open-ended phrase tables or specialized marriage,
   birthday, age, kinship, or filter handlers.

8. **Why retain them?** Identity requests are high-frequency session diagnostics
   with immutable `self`/`assistant` reference meaning. Household count/list are
   high-frequency dashboard operations with one stable declared
   `current_household -> member` plan. They save roughly four orders of magnitude
   of local CPU latency, are unambiguous exact forms, and remain independently
   covered by the planner. Any future Tier-0 addition requires measured traffic,
   latency value, unambiguous semantics, Planner-only coverage, and at least 95%
   category accuracy investigation first.

9. **Is marriage execution generic?** Yes. Marriage start compiles to
   `select(self -> spouse, property=start_date,
   property_source=relationship)`. Duration uses the same relationship property
   with the generic `duration` operation. There is no marriage-specific data
   access function.

10. **Are birthday/date operations generic?** Yes. Birthday lookup selects the
    semantic `birth_date`. Countdown uses `annual_occurrence` with `mode=days`;
    elapsed relationship time uses `duration`/`date_difference`. Date arithmetic
    stays in the deterministic executor, not the LLM or a birthday-specific
    query path.

11. **Does the planner emit physical fields or household IDs?** No accepted plan
    in the evaluation did. Its advertised payload contains semantic names rather
    than physical aliases such as `dob`, `first_name`, or `parent_of`; the strict
    output schema excludes `entity_id`. Model-originated IDs and unknown physical
    relations/properties are rejected before graph access with structured
    validation reasons.

12. **Planner P50/P95 latency?** 8,528.573/14,014.189 ms on the local CPU-only 8.2B
    Q4_K_M run. This is a deployment measurement, not an architectural constant.

13. **Tier-0 P50/P95 latency?** 0.023/0.075 ms over 600 JSON-graph executions.

14. **Accuracy with Tier 0 disabled?** 98.20% semantic equivalence for the local
    real-model full run; 100% semantic-plan and executor success with the
    deterministic CI planner. The distinction keeps model quality separate from
    downstream IR and executor independence.

## Diagnostics and safeguards

The planner receives a compact semantic capability catalog and compact strict
JSON schema. A plan exposes input summary, raw output, normalized request,
validation result/detail, attempt count, planner latency, executor latency,
tier, executor status, and final answer through the developer benchmark. Runtime
logs retain only privacy-safe validation status and counts. Structural output
may receive one focused retry; semantic validation failures never trigger a
Tier-0 fallback.

Stable rejection codes include `NOT_A_FACT`, `MALFORMED_OUTPUT`,
`UNSUPPORTED_OPERATION`, `UNKNOWN_PROPERTY`, `UNKNOWN_RELATION`,
`MODEL_ORIGINATED_ENTITY_ID`, and `INVALID_PLAN`.
