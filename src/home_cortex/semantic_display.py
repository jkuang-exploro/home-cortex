"""Presentation of validated semantic IR; no utterances, graph reads or inference."""

from __future__ import annotations

import html
import json
import re
from typing import TYPE_CHECKING

from .semantic_ontology import SemanticOntology

if TYPE_CHECKING:
    from .semantic_facts import SemanticFactRequest, SemanticFilter, SemanticReference


def _text(value: str) -> str:
    """Keep labels/literals plain text when chat clients render Markdown/HTML."""
    return re.sub(r"([\\`*_{}\[\]()#!|])", r"\\\1", html.escape(value, quote=False)).replace("\n", " ").replace("\r", " ")


class SemanticDisplay:
    def __init__(self, ontology: SemanticOntology, locale: str) -> None:
        self.ontology = ontology
        self.locale = locale.replace("_", "-")
        self.zh = self.locale.startswith("zh")

    def label(self, labels: tuple[tuple[str, str], ...], fallback: str) -> str:
        translations = dict(labels)
        for locale in (self.locale, self.locale.split("-")[0], "en"):
            if locale in translations:
                return _text(translations[locale])
        return _text(fallback)

    def property(self, name: str) -> str:
        definition = self.ontology.properties.get(name)
        return self.label(definition.label if definition else (), name)

    def relation(self, name: str) -> str:
        # Only an unfiltered, one-hop concept can label a base relation.
        # Never collapse a filtered or multi-hop concept into a broader noun.
        concept = self.ontology.reference_concepts.get(name)
        labels = concept.label if (
            concept and len(concept.path) == 1
            and concept.path[0].relation == name and not concept.path[0].filters
        ) else ()
        return self.label(labels, name)

    def literal(self, property_name: str, value: object) -> str:
        definition = self.ontology.properties.get(property_name)
        labels = dict(definition.value_labels).get(value, ()) if definition and isinstance(value, str) else ()
        if labels:
            # A label changes presentation, never the executed canonical value.
            return self.label(labels, value)
        if isinstance(value, tuple):
            return "[" + ", ".join(self.literal(property_name, item) for item in value) + "]"
        return _text(json.dumps(value, ensure_ascii=False))

    def condition(self, item: SemanticFilter, anchor: str, *, collection: bool) -> str:
        if item.predicate:
            definition = self.ontology.collection_predicates.get(item.predicate)
            return self.label(definition.label if definition else (), item.predicate)
        name = item.property or ""
        owner = "实体" if self.zh else "entity"
        if item.source == "relation":
            # Collection filters each quantify over associated edges separately.
            # Traversal conditions instead apply together to the traversed edge.
            owner = ("至少一条关联关系" if collection else "本步关系") if self.zh else (
                "at least one associated relationship" if collection else "this relationship"
            )
        left = f"{owner}.{self.property(name)}"
        if item.value_from == "anchor":
            right = f"{'起点' if self.zh else 'anchor'}({anchor}).{self.property(item.value_property or name)}"
        else:
            right = self.literal(name, item.value)
        operator = {"eq": "=", "ne": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "in": "∈"}.get(item.operator)
        if operator:
            return f"{left} {operator} {right}"
        # Explicit operator names preserve exists(false/null) and half-open bounds.
        suffix = " [start inclusive, end exclusive]" if item.operator == "date_range" else ""
        if self.zh and suffix:
            suffix = " [含起点，不含终点]"
        return f"{left}: {item.operator}({right}){suffix}"

    def conditions(self, items: tuple[SemanticFilter, ...], anchor: str, *, collection: bool) -> str:
        return (" 且 " if self.zh else " AND ").join(
            self.condition(item, anchor, collection=collection) for item in items
        )

    def reference(self, reference: SemanticReference) -> str:
        roots = {
            "self": ("您", "you"),
            "assistant": ("助手", "assistant"),
            "current_household": ("当前家庭", "current household"),
            "unresolved": ("未明确的对象", "unresolved reference"),
        }
        if reference.kind == "named_entity":
            root = f"{'名称' if self.zh else 'named'} {self.literal('', reference.value)}"
            if reference.entity_type:
                root += f" ({_text(reference.entity_type)})"
        elif reference.kind == "entity_id":
            # Internal IDs are not display names. Do not expose record IDs.
            root = "指定实体" if self.zh else "specified entity"
        elif reference.kind == "discourse":
            cardinality = ("对象集合" if reference.cardinality == "collection" else "对象") if self.zh else reference.cardinality
            root = (f"前{reference.turn_offset}轮的{cardinality}" if self.zh else
                    f"{cardinality} from {reference.turn_offset} turn(s) earlier")
        else:
            pair = roots.get(reference.kind, (reference.kind, reference.kind))
            root = pair[0 if self.zh else 1]
        steps = [root]
        for index, step in enumerate(reference.path, 1):
            text = f"{index}. {self.relation(step.relation)}"
            if step.filters:
                text += " {" + self.conditions(step.filters, root, collection=False) + "}"
            steps.append(text)
        return " → ".join(steps)

    def describe(self, request: SemanticFactRequest) -> str:
        scope = self.reference(request.subject)
        if request.filters:
            scope += ("；结果筛选：" if self.zh else "; result filters: ") + self.conditions(
                request.filters, self.reference(request.subject.model_copy(update={"path": ()})),
                collection=not (request.projection == "each" and request.property_source == "relationship"),
            )
        if request.exclude:
            scope += ("；排除对象：" if self.zh else "; exclude entities: ") + " | ".join(
                "(" + self.reference(reference) + ")" for reference in request.exclude
            )
        if request.other:
            scope += ("；比较对象：" if self.zh else "; compare with: ") + "(" + self.reference(request.other) + ")"
        if request.property:
            owner = ("最终关系" if self.zh else "final relationship") if request.property_source == "relationship" else ("结果实体" if self.zh else "result entity")
            scope += f"; {owner}.{self.property(request.property)}"
        if request.projection == "each":
            if request.property_source == "relationship":
                scope += "；逐条关系记录的结果（关系条件作用于同一条记录）" if self.zh else "; result for each relationship record (relationship conditions bind the same record)"
            else:
                scope += "；逐个实体的结果" if self.zh else "; result for each entity"
        return ("查询范围：" if self.zh else "Query scope: ") + scope + ("。" if self.zh else ".")
