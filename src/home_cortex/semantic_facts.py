"""Semantic household facts: interpretation, resolution, execution, rendering.

Implementations live in focused modules:

- semantic_ir.py — request/result types
- semantic_schema.py — vocabulary mapping onto the catalog
- semantic_planner.py — LLM interpreter
- entity_resolver.py — speaker/name/path grounding
- household_fact_engine.py — deterministic execution
- fact_renderer.py — answer text
"""
from .semantic_ir import (
    AgentRequestContext,
    DiscourseContext,
    FactAnswer,
    FactContentGroup,
    FactEvidence,
    FactOperation,
    FactRelationshipEvidence,
    FactResult,
    FactRow,
    FactStatus,
    FactTimings,
    PlannerDiagnostics,
    PlannerValidationCode,
    ReferenceKind,
    ResolutionResult,
    ResolutionStatus,
    SemanticConceptUse,
    SemanticFactRequest,
    SemanticFilter,
    SemanticMutationIntent,
    SemanticPlan,
    SemanticPlannerFailure,
    SemanticPlannerOutcome,
    SemanticReference,
    SemanticRelationStep,
    _CONTEXT_ENTITY_TYPES,
    _FactFailure,
    _entity_type,
    _last_relation,
    _related_entity_id,
    _string_or_none,
    _unique_entities,
)
from .semantic_schema import SemanticSchemaRegistry
from .semantic_planner import (
    SemanticFactPlanner,
    planner_input_summary,
    _identity_person_hint,
    _identity_person_mismatch,
    _invalid_plan_retry_hint,
    _object_location_hint,
    _object_location_mismatch,
    _planner_clock,
)
from .entity_resolver import EntityResolver, _derived_date_matches
from .household_fact_engine import HouseholdFactEngine, SemanticFactService, _failure_stage
from .fact_renderer import FactRenderer

__all__ = [
    "AgentRequestContext",
    "DiscourseContext",
    "EntityResolver",
    "FactAnswer",
    "FactRenderer",
    "FactResult",
    "HouseholdFactEngine",
    "SemanticFactPlanner",
    "SemanticFactRequest",
    "SemanticFactService",
    "SemanticFilter",
    "SemanticMutationIntent",
    "SemanticPlan",
    "SemanticPlannerFailure",
    "SemanticReference",
    "SemanticRelationStep",
    "SemanticSchemaRegistry",
]
