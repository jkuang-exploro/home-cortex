"""Strict semantic interpreter: utterance to SemanticPlan, never storage fields."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from .request_tracing import stage
from .semantic_ir import (
    AgentRequestContext,
    PlannerDiagnostics,
    PlannerValidationCode,
    SemanticFactRequest,
    SemanticPlan,
    SemanticPlannerFailure,
    SemanticPlannerOutcome,
)
from .semantic_schema import SemanticSchemaRegistry
from .text import latest_user_message

class SemanticFactPlanner:
    """Strict semantic interpreter that never sees storage field names."""

    def __init__(self, ollama: Any, schema: SemanticSchemaRegistry, *, enable_mutations: bool = False) -> None:
        self.ollama = ollama
        self.schema = schema
        self.enable_mutations = enable_mutations

    @stage("planner.total")
    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> SemanticPlannerOutcome:
        started = perf_counter()
        build_started = perf_counter()
        output_schema = self.schema.planner_output_schema()
        capabilities = self.schema.planner_capability_payload()
        input_summary = planner_input_summary(self.schema.capability_payload())
        prompt_build_ms = (perf_counter() - build_started) * 1000
        mutation_runtime = {}
        mutation_calls = 0
        mutation_ms = 0.0
        mutation_planner = getattr(self.ollama, "plan_item_mutation", None) if self.enable_mutations else None
        if mutation_planner is not None:
            mutation_started = perf_counter()
            decision, mutation_runtime = await mutation_planner(messages)
            mutation_ms = (perf_counter() - mutation_started) * 1000
            mutation_calls = 1
            if decision.requires_mutation:
                mutation_plan = SemanticPlan(requires_fact=False, mutation=decision.mutation)
                latency = (perf_counter() - started) * 1000
                return SemanticPlannerOutcome(mutation_plan, latency, PlannerDiagnostics(
                    input_summary=input_summary, output_raw=decision.model_dump(mode="json"),
                    normalized_plan=mutation_plan.model_dump(mode="json"),
                    validation_result="VALID", attempt_count=1, latency_ms=latency,
                    request_ms=mutation_ms, prompt_build_ms=prompt_build_ms,
                    **_planner_runtime_fields(mutation_runtime),
                ))
        payload: Mapping[str, Any] | None = None
        plan: SemanticPlan | None = None
        structural_error: Exception | None = None
        validation: PlannerValidationCode | None = None
        attempts = 0
        request_ms = mutation_ms
        validation_ms = 0.0
        runtime: Mapping[str, Any] = {}
        transport_attempts: list[dict[str, Any]] = []
        utterance = latest_user_message(messages)
        last_invalid_request: SemanticFactRequest | None = None
        for attempts in (1, 2):
            planner_messages: list[dict[str, Any]] = [
                {"role": "user", "content": str(message.get("content", ""))}
                for message in messages if message.get("role") == "user"
            ]
            if attempts == 2:
                previous = validation or _structural_validation_code(structural_error)
                hint = _identity_person_mismatch(utterance, last_invalid_request)
                location = _object_location_mismatch(utterance, last_invalid_request)
                grammar = _invalid_plan_retry_hint(self.schema, last_invalid_request)
                extra = " ".join(part for part in (hint, location, grammar) if part)
                planner_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response failed strict structural or "
                            f"semantic validation ({previous}). Recompile the "
                            "original meaning using only the advertised grammar; "
                            "check reference path, first vs second person, "
                            "property ownership, and operation requirements. "
                            + (f"{extra} " if extra else "")
                            + "Return exactly one structured value conforming to the "
                            "supplied output schema."
                        ),
                    }
                )
            request_started = perf_counter()
            try:
                payload = await self.ollama.plan_semantic_fact(
                    planner_messages,
                    capabilities,
                    output_schema,
                    household_now=_planner_clock(context.current_time),
                )
                runtime = getattr(self.ollama, "last_planner_runtime", {}) or {}
                validate_started = perf_counter()
                candidate = SemanticPlan.model_validate(
                    self.schema.expand_planner_concepts(payload)
                    if isinstance(payload, Mapping) else payload
                )
                validation = "VALID" if candidate.mutation is not None else "NOT_A_FACT"
                if candidate.request is not None and any(
                    reference.kind == "entity_id"
                    for reference in (candidate.request.subject, candidate.request.other, *candidate.request.exclude)
                    if reference is not None
                ):
                    validation = "MODEL_ORIGINATED_ENTITY_ID"
                elif candidate.request is not None:
                    validation = self.schema.validation_code(candidate.request)
                    if validation == "VALID":
                        person_error = _identity_person_mismatch(
                            utterance, candidate.request
                        )
                        location_error = _object_location_mismatch(
                            utterance, candidate.request
                        )
                        if person_error:
                            validation = "INVALID_PLAN"
                        elif location_error:
                            validation = "INVALID_PLAN"
                    if validation != "VALID":
                        last_invalid_request = candidate.request
                validation_ms += (perf_counter() - validate_started) * 1000
                if validation not in {"VALID", "NOT_A_FACT"}:
                    plan = None
                    structural_error = ValueError(validation)
                    continue
                plan = candidate
                structural_error = None
                break
            except (ValueError, TypeError) as error:
                runtime = getattr(self.ollama, "last_planner_runtime", {}) or runtime
                structural_error = error
            finally:
                attempt_runtime = getattr(self.ollama, "last_planner_runtime", {}) or {}
                if attempt_runtime.get("codec_version") is not None:
                    transport_attempts.append({
                        **_planner_runtime_fields(attempt_runtime),
                        "attempt": attempts,
                    })
                request_ms += (perf_counter() - request_started) * 1000
        latency_ms = (perf_counter() - started) * 1000
        attempts += mutation_calls
        if mutation_runtime:
            runtime = dict(runtime)
            for key in ("prompt_eval_count", "prompt_eval_duration_ms", "eval_count",
                        "eval_duration_ms", "load_duration_ms"):
                left, right = runtime.get(key), mutation_runtime.get(key)
                runtime[key] = left + right if left is not None and right is not None else None
        timing = {
            "prompt_build_ms": prompt_build_ms,
            "request_ms": request_ms,
            "validation_ms": validation_ms,
            **_planner_runtime_fields(runtime),
        }
        if transport_attempts:
            timing["transport"]["attempts"] = transport_attempts
        if plan is None:
            code = validation or _structural_validation_code(structural_error)
            raise SemanticPlannerFailure(
                PlannerDiagnostics(
                    input_summary=input_summary,
                    output_raw=payload,
                    validation_result=code,
                    failure_detail=(
                        type(structural_error).__name__
                        if structural_error is not None
                        else "planner returned no plan"
                    ),
                    attempt_count=attempts,
                    latency_ms=latency_ms,
                    **timing,
                )
            )
        assert validation is not None
        diagnostics = PlannerDiagnostics(
            input_summary=input_summary,
            output_raw=payload,
            normalized_plan=plan.model_dump(mode="json"),
            validation_result=validation,
            attempt_count=attempts,
            latency_ms=latency_ms,
            **timing,
        )
        return SemanticPlannerOutcome(plan, latency_ms, diagnostics)


def planner_input_summary(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize planner affordances without copying household fact values."""
    relation_properties = capabilities.get("semantic_relation_properties", {})
    semantic_properties = capabilities.get("semantic_properties", {})
    return {
        "references": list(capabilities.get("references", ())),
        "entity_types": list(capabilities.get("entity_types", ())),
        "relations": list(capabilities.get("semantic_relations", ())),
        "entity_properties": {
            str(entity_type): list(properties)
            for entity_type, properties in semantic_properties.items()
        }
        if isinstance(semantic_properties, Mapping)
        else {},
        "relationship_properties": {
            str(relation): list(properties)
            for relation, properties in relation_properties.items()
        }
        if isinstance(relation_properties, Mapping)
        else {},
        "operations": list(capabilities.get("operations", ())),
        "collection_predicates": list(
            capabilities.get("collection_predicates", ())
        ),
    }



_SECOND_PERSON_IDENTITY = re.compile(
    r"(?is)^(?:who are you\b|what(?:'s| is) your (?:name|role)\b)|"
    r"^(?:你|您)\s*(?:是\s*(?:谁|哪)|的名字|叫什么)"
)
_FIRST_PERSON_IDENTITY = re.compile(
    r"(?is)^(?:who am i\b|what(?:'s| is) my name\b)|"
    r"^我\s*(?:是\s*(?:谁|哪)|的名字|叫什么|的身份)"
)
_NAMED_OBJECT_LOCATION = re.compile(
    r"(?is)^(?:where(?:'s| is) (?:the )?(?!my\b).+|"
    r"(?!(?:我家|家里|咱家|我这个家)\s*).{1,24}在哪里)"
)


def _identity_person_hint(utterance: str) -> str | None:
    text = utterance.strip()
    if _SECOND_PERSON_IDENTITY.match(text):
        return (
            "The latest utterance addresses this helper in the second person. "
            "subject.kind must be assistant, path must be empty, property=null."
        )
    if _FIRST_PERSON_IDENTITY.match(text):
        return (
            "The latest utterance is first-person identity. "
            "subject.kind must be self, path must be empty, property=null."
        )
    return None


def _object_location_hint(utterance: str) -> str | None:
    if not _NAMED_OBJECT_LOCATION.match(utterance.strip()):
        return None
    return (
        "The latest utterance asks where a named object is. "
        "subject.kind=named_entity, entity_type=item, path concept "
        "location, resolve_reference, property=null. Not a person and "
        "not adult/minor."
    )


def _invalid_plan_retry_hint(
    schema: SemanticSchemaRegistry,
    request: SemanticFactRequest | None,
) -> str | None:
    """IR-based retry grammar; never matches on utterance text."""
    if request is None:
        return None
    notes: list[str] = []
    if schema.contract_error(request) == "CONTRADICTORY_PREDICATES":
        notes.append(
            "adult and minor cannot be combined. Gender is gender=male or "
            "gender=female, not those predicates."
        )
    hops = [step.relation for step in request.subject.path]
    if request.subject.kind == "self" and "member" in hops:
        notes.append(
            "self cannot traverse member; household people use "
            "current_household then member."
        )
    if "location" in hops and request.subject.kind == "current_household":
        notes.append(
            "item location uses named_entity entity_type=item then concept "
            "location; current_household cannot traverse location."
        )
    if (
        request.subject.kind == "named_entity"
        and "location" in hops
        and request.subject.entity_type != "item"
    ):
        notes.append(
            "location accepts items, not persons or spaces; named objects "
            "use entity_type=item."
        )
    named_predicates = {
        item.predicate for item in request.filters if item.predicate
    }
    if request.subject.kind == "named_entity" and named_predicates.intersection(
        {"adult", "minor"}
    ):
        notes.append(
            "named objects are not household members and do not take "
            "adult/minor. A named item's location is entity_type=item then "
            "concept location, resolve_reference, property=null."
        )
    if (
        request.subject.kind == "named_entity"
        and not hops
        and request.operation == "select"
        and request.property is None
    ):
        notes.append(
            "select with property=null needs a collection path. Item "
            "location uses named_entity entity_type=item then concept "
            "location; resolve_reference, property=null."
        )
    filter_properties = {
        item.property for item in request.filters if item.property
    }
    if "member" in hops and filter_properties.intersection({"space_type", "item_type"}):
        notes.append(
            "space_type and item_type are not household-member filters; "
            "rooms use current_household then concept room."
        )
    if request.subject.kind == "current_household" and not hops and (
        request.property in {"space_type", "item_type"}
        or filter_properties.intersection({"space_type", "item_type"})
    ):
        notes.append(
            "household rooms use current_household then concept room."
        )
    if request.operation == "same_entity" and (
        request.property is not None
        or request.other is None
        or request.filters
        or request.property_source != "entity"
    ):
        notes.append(
            "same_entity compares two resolved references; property=null; "
            "other is required; not a stored property and not resolve_reference."
        )
    return " ".join(notes) or None


def _object_location_mismatch(
    utterance: str, request: SemanticFactRequest | None
) -> str | None:
    """Reject object-location plans that name a person or omit location."""
    hint = _object_location_hint(utterance)
    if hint is None:
        return None
    if request is None:
        return hint
    hops = [step.relation for step in request.subject.path]
    if (
        request.subject.kind == "named_entity"
        and request.subject.entity_type in {None, "item"}
        and "location" in hops
        and request.property is None
        and not request.filters
    ):
        return None
    return hint


def _identity_person_mismatch(
    utterance: str, request: SemanticFactRequest | None
) -> str | None:
    """Reject first/second-person identity plans that name the wrong referent."""
    if request is None:
        return _identity_person_hint(utterance)
    if (
        request.operation != "resolve_reference"
        or request.property is not None
        or request.subject.path
    ):
        return None
    hint = _identity_person_hint(utterance)
    if hint is None:
        return None
    if _SECOND_PERSON_IDENTITY.match(utterance.strip()):
        return hint if request.subject.kind != "assistant" else None
    if _FIRST_PERSON_IDENTITY.match(utterance.strip()):
        return hint if request.subject.kind != "self" else None
    return None


def _planner_clock(moment: datetime) -> str:
    """Hour-precision clock so the planner prefix stays cacheable within an hour."""
    return moment.replace(minute=0, second=0, microsecond=0).isoformat()


def _planner_runtime_fields(runtime: Mapping[str, Any]) -> dict[str, Any]:
    def number(name: str) -> float | int:
        value = runtime.get(name, 0) or 0
        return value

    return {
        "prompt_eval_count": int(number("prompt_eval_count")),
        "prompt_eval_duration_ms": float(number("prompt_eval_duration_ms")),
        "eval_count": int(number("eval_count")),
        "eval_duration_ms": float(number("eval_duration_ms")),
        "load_duration_ms": float(number("load_duration_ms")),
        "transport": {key: runtime[key] for key in (
            "codec_version", "schema_fingerprint", "compact_output_bytes",
            "expanded_output_bytes", "transport_parse_success", "transport_parse_ms",
            "compact_prompt_bytes", "compact_schema_bytes", "expanded_schema_bytes",
            "field_dictionary_fingerprint",
        ) if key in runtime},
    }


def _structural_validation_code(
    error: Exception | None,
) -> PlannerValidationCode:
    if isinstance(error, ValidationError):
        if any(item.get("loc", ())[-1:] == ("operation",) for item in error.errors()):
            return "UNSUPPORTED_OPERATION"
    return "MALFORMED_OUTPUT"
