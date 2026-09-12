"""Render executed semantic requests into localized answer text."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .profiling import stage
from .display import resolve_display_name
from .semantic_display import SemanticDisplay
from .semantic_ir import (
    AgentRequestContext,
    FactResult,
    SemanticFactRequest,
    SemanticFilter,
    SemanticReference,
    SemanticRelationStep,
    _last_relation,
)
from .semantic_ontology import SemanticOntology

class FactRenderer:
    def __init__(
        self,
        ontology: SemanticOntology | None = None,
        *,
        detailed: bool = False,
    ) -> None:
        self.ontology = ontology or SemanticOntology.load_default()
        self.detailed = detailed

    @stage("renderer")
    def render(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        rendered = self._render_result(request, result, context)
        # Describe the expanded, validated IR once, including empty/partial results.
        # Unsupported plans have no executed scope to describe.
        if self.detailed and result.status != "semantic_plan_unsupported" and (
            request.subject.path or request.filters or request.exclude or request.other
            or request.property is not None
            or request.operation == "count" or request.projection == "each"
            or (request.operation == "select" and request.property is None)
        ):
            description = SemanticDisplay(self.ontology, context.locale or "en").describe(request)
            return f"{description}\n{rendered}"
        if not self.detailed and result.status == "found" and not (
            request.operation == "count"
            or (request.operation == "select" and request.property is None)
        ):
            qualifier = self._condition_qualifier(request, context.locale or "en")
            if qualifier:
                separator = "\n" if result.shape == "rows" else ""
                if separator:
                    qualifier = qualifier.lstrip()
                rendered = f"{rendered}{separator}{qualifier}"
        return rendered

    def _condition_qualifier(
        self,
        request: SemanticFactRequest,
        language: str,
    ) -> str:
        """Faithfully expose conditions not already carried by relation nouns."""
        display = SemanticDisplay(self.ontology, language)
        remaining: list[SemanticFilter] = list(request.filters)
        noun_relations = {"spouse", "child", "parent"}
        for step in request.subject.path:
            for item in step.filters:
                absorbed_gender = (
                    step.relation in noun_relations
                    and item.source == "entity"
                    and item.property == "gender"
                    and item.operator == "eq"
                    and item.value_from is None
                    and item.value in {"male", "female"}
                )
                if not absorbed_gender:
                    remaining.append(item)
        clauses: list[str] = []
        if remaining:
            clauses.append(
                display.conditions(
                    tuple(remaining),
                    display.reference(request.subject.model_copy(update={"path": ()})),
                    collection=request.projection != "scalar",
                )
            )
        if request.exclude:
            excluded = " | ".join(display.reference(item) for item in request.exclude)
            clauses.append(("排除 " if display.zh else "exclude ") + excluded)
        if not clauses:
            return ""
        joined = ("；" if display.zh else "; ").join(clauses)
        return f"（条件：{joined}）" if display.zh else f" (Conditions: {joined})"

    def _render_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        language = context.locale or "en"
        if (result.status == "found" and result.shape == "entities"
            and request.operation == "select" and request.property is None
            and result.content_groups):
            display = SemanticDisplay(self.ontology, language)
            simple = (
                request.subject.kind == "named_entity"
                and len(request.subject.path) == 1
                and not request.subject.path[0].filters
                and not request.filters and not request.exclude and request.other is None
            )
            if simple:
                owner = request.subject.value
                heading = f"{owner}里有：" if display.zh else f"Contents of {owner}:"
            else:
                heading = display.describe(request)
            lines = [
                f"{_name(group.space, language)}" + ("：" if display.zh else ": ")
                + ("、" if display.zh else ", ").join(_name(item, language) for item in group.entities)
                for group in result.content_groups
            ]
            return heading + "\n" + ("；\n".join(lines) + "。" if display.zh else "\n".join(lines))
        if result.status == "ambiguous" and "discourse_antecedent" in result.missing_requirements:
            return "前文包含多个对象，请说明您指的是哪一个。" if language.startswith("zh") else "That earlier turn refers to multiple entities; please clarify which one you mean."
        if result.status == "discourse_context_missing":
            return "请说明您指的是谁；对应的前文对象尚未明确。" if language.startswith("zh") else "Please clarify who you mean; that earlier turn has no resolved referent."
        if result.shape == "rows":
            if not result.rows:
                return "没有找到符合条件的记录。" if language.startswith("zh") else "No matching records."
            return "\n".join(
                f"{_name(row.entity, language)}: " + self._render_result(
                    request.model_copy(update={"projection": "scalar"}),
                    FactResult(row.status, row.value, row.evidence, row.missing_requirements, unit=row.unit), context
                ) for row in result.rows
            )
        if result.status == "found" and request.operation == "inspect":
            display = SemanticDisplay(self.ontology, language)
            values = {key: (_name({"name": value}, language) if key == "display_name" and value is not None else value)
                      for key, value in result.value.items()}
            lines = [f"{display.property(key)}：{display.literal(key, value) if value is not None else ('未记录' if display.zh else 'not recorded')}"
                     for key, value in values.items()]
            return "\n".join(lines)
        if result.status == "found" and request.operation == "date_add":
            return f"指定日期是{result.value}。" if language.startswith("zh") else f"The specified date is {result.value}."
        if not self.detailed and result.status == "found" and (
            request.operation == "count"
            or (request.operation == "select" and request.property is None)
        ):
            natural = self._collection_result(request, result, language)
            if natural is not None:
                return natural
        if language.startswith("zh"):
            return self._zh(request, result, context)
        return self._en(request, result, context)

    def _collection_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        language: str,
    ) -> str | None:
        display = SemanticDisplay(self.ontology, language)
        described = display.collection_noun(request.subject, request.filters)
        if described is None:
            return self._generic_collection_result(request, result, display)
        noun, remaining = described
        if request.exclude or request.other:
            return self._generic_collection_result(request, result, display)
        zh = language.startswith("zh")
        if request.subject.kind == "current_household":
            owner = "家里" if zh else "the household"
        elif request.subject.kind == "self":
            owner = "您" if zh else "you"
        elif request.subject.kind == "named_entity" and request.subject.value:
            owner = str(request.subject.value)
        else:
            return self._generic_collection_result(request, result, display)
        condition = display.conditions(
            remaining,
            display.reference(request.subject.model_copy(update={"path": ()})),
            collection=True,
        ) if remaining else ""
        if request.operation == "count":
            count = int(result.value)
            if zh:
                counter = (
                    "个"
                    if noun == "人"
                    else "位" if noun.endswith(("人", "男性", "女性")) else "个"
                )
                if count == 0:
                    target = f"符合{condition}的{noun}" if condition else noun
                    return f"{owner}没有{target}。"
                suffix = f"，筛选条件还包括：{condition}" if condition else ""
                return f"{owner}有{count}{counter}{noun}{suffix}。"
            plural = noun if count == 1 else {
                "person": "people",
            }.get(noun, noun + ("es" if noun.endswith("s") else "s"))
            qualifier = f" matching {condition}" if condition else ""
            verb = "is" if count == 1 else "are"
            return f"There {verb} {count} {plural}{qualifier} in {owner}."
        values = result.value if isinstance(result.value, list) else []
        if not values:
            if zh:
                target = f"符合{condition}的{noun}" if condition else noun
                return f"{owner}没有{target}。"
            target = f"{noun} matching {condition}" if condition else noun
            return f"There are no {target}s in {owner}."
        names = ("、" if zh else ", ").join(_name(item, language) for item in values)
        if zh:
            suffix = f"（筛选条件还包括：{condition}）" if condition else ""
            return f"{owner}的{noun}有：{names}{suffix}。"
        suffix = f" matching {condition}" if condition else ""
        return f"The {noun}s in {owner}{suffix} are {names}."

    def _generic_collection_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        display: SemanticDisplay,
    ) -> str:
        zh = display.zh
        scope = display.reference(request.subject)
        clauses: list[str] = []
        if request.filters:
            clauses.append(display.conditions(request.filters, scope, collection=True))
        if request.exclude:
            excluded = " | ".join(display.reference(item) for item in request.exclude)
            clauses.append(("排除 " if zh else "exclude ") + excluded)
        condition = ("；" if zh else "; ").join(clauses)
        if request.operation == "count":
            if zh:
                suffix = f"，条件为{condition}" if condition else ""
                return f"在{scope}中有{int(result.value)}条记录{suffix}。"
            suffix = f" matching {condition}" if condition else ""
            return f"There are {int(result.value)} records in {scope}{suffix}."
        values = result.value if isinstance(result.value, list) else []
        names = ("、" if zh else ", ").join(_name(item, "zh" if zh else "en") for item in values)
        if zh:
            suffix = f"，条件为{condition}" if condition else ""
            return f"在{scope}中找到的记录为：{names or '无'}{suffix}。"
        suffix = f" matching {condition}" if condition else ""
        return f"The records in {scope}{suffix} are {names or 'none'}."

    def _zh(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        if result.status == "caller_context_missing":
            return "我无法确认当前登录者的身份。"
        if result.status == "entity_not_found":
            return "家庭资料中没有找到对应的人或实体。"
        if result.status == "relationship_not_found":
            relation = _relation_label(
                request.subject,
                result.evidence.relationship,
            )
            return f"家庭资料中没有找到{relation}关系记录。"
        if result.status == "property_unavailable":
            label = _property_label(result.missing_requirements)
            return (
                "家庭资料中有对应记录，但目前没有记录"
                f"{_subject_possessive(request.subject)}{label}。"
            )
        if result.status == "relation_property_unavailable":
            if result.evidence.relationship == "spouse" and (
                "start_date" in result.missing_requirements
            ):
                return "家庭资料中有配偶关系记录，但目前没有记录结婚日期。"
            label = _property_label(result.missing_requirements)
            return f"家庭资料中有对应关系记录，但目前没有记录{label}。"
        if result.status == "filter_input_missing":
            return "目前缺少筛选所需资料，无法可靠完成筛选。"
        if result.status == "filter_unsupported":
            return "当前语义查询协议不支持这个筛选条件。"
        if result.status == "operator_unsupported":
            return "当前语义查询协议不支持这项计算。"
        if result.status == "semantic_plan_unsupported":
            return "老管家无法将这个请求转换为受支持的家庭事实查询。"
        if result.status == "ambiguous":
            names = "、".join(_name(item, "zh") for item in result.candidates)
            return f"找到多个符合条件的家庭成员：{names}。请说明您指哪一位。"
        if result.status == "computation_input_missing":
            if (
                request.operation in {"argmin", "argmax"}
                and request.property == "birth_date"
            ):
                return "目前缺少部分家庭成员的出生日期，因此无法可靠判断年龄排序。"
            label = _property_label(result.missing_requirements) or "所需资料"
            return f"目前没有足够的{label}来完成这项计算。"
        if result.status == "computation_impossible":
            label = _property_label(result.missing_requirements)
            suffix = f"，缺少{label}" if label else ""
            return f"家庭资料不足以完成这项计算{suffix}。"
        if result.status == "collection_incomplete":
            return "查询结果超过当前完整性上限，无法给出可靠的总数或完整列表。"
        if request.operation == "count":
            count = int(result.value)
            return f"符合条件的记录数：{count}。"
        if request.operation == "select" and request.property is None:
            values = result.value if isinstance(result.value, list) else []
            if not values:
                if request.filters or any(step.filters for step in request.subject.path):
                    return "没有找到符合筛选条件的记录。"
                return "没有找到符合查询范围的记录。"
            names = "、".join(_name(item, "zh") for item in values)
            return f"符合条件的记录：{names}。"
        if request.operation == "select":
            if (
                request.property_source == "relationship"
                and request.subject.path
                and request.subject.path[-1].relation == "spouse"
                and request.property == "start_date"
            ):
                return f"该配偶关系的开始日期是{result.value}。"
            if request.property == "birth_date" and request.property_source == "entity":
                return f"{_subject_possessive(request.subject)}出生日期是{result.value}。"
            if request.property == "full_address":
                return f"具体住址是{_format_address(result.value)}。"
            return f"查询到的值是{result.value}。"
        if request.operation in {"date_difference", "duration", "completed_years"}:
            unit = result.unit or ("years" if request.operation == "completed_years" else request.mode)
            if unit == "years" and request.property == "birth_date" and request.property_source == "entity":
                return f"{_subject_nominative(request.subject)}今年{result.value}岁。"
            label = {"years": "年", "months": "个月", "days": "天", "seconds": "秒"}[unit]
            if result.value >= 0:
                prefix = "已满" if unit in {"years", "months"} else "已过"
                return f"从记录的日期到现在{prefix}{result.value}{label}。"
            return f"该日期与现在的间隔为{result.value}{label}。"
        if request.operation == "annual_occurrence":
            if request.mode == "days":
                days = int(result.value)
                if days == 0:
                    return f"{_subject_possessive(request.subject)}生日就是今天。"
                return f"{_subject_possessive(request.subject)}生日还有{days}天。"
            return f"{_subject_possessive(request.subject)}下次生日是{result.value}。"
        if request.operation in {"argmin", "argmax"}:
            if request.other is not None and isinstance(result.value, Mapping):
                selected = _name(result.value.get("selected"), "zh")
                other = _name(result.value.get("other"), "zh")
                if request.property == "birth_date" and request.property_source == "entity":
                    if result.value.get("equal"):
                        return f"{selected}和{other}年龄相同。"
                    adjective = "大" if request.operation == "argmin" else "小"
                    return f"{selected}年龄比{other}{adjective}。"
                if result.value.get("equal"):
                    return f"{selected}和{other}的比较值相同。"
                return f"{'最小值' if request.operation == 'argmin' else '最大值'}对应对象：{selected}。"
            if request.property == "birth_date" and request.property_source == "entity":
                qualifier = "最年长" if request.operation == "argmin" else "最年轻"
                return f"查询范围内{qualifier}的是{_name(result.value, 'zh')}。"
            return f"符合极值条件的是{_name(result.value, 'zh')}。"
        if request.operation in {"sum", "average", "min", "max"}:
            return f"计算结果是{result.value}。"
        if request.operation in {
            "unit_conversion",
        }:
            return f"换算结果是{result.value} {result.unit or request.to_unit}。"
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"符合条件的是{_name(result.value, 'zh')}。"
        if request.operation == "same_entity":
            return "是。" if result.value else "不是。"
        if request.subject.kind == "assistant":
            return f"我是{context.assistant_display_name}。"
        name = _name(result.value, "zh")
        if request.subject.kind == "self" and not request.subject.path:
            return f"您是{name}。"
        return f"{_subject_nominative(request.subject)}是{name}。"

    def _en(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        if result.status != "found":
            return {
                "caller_context_missing": "I cannot verify the current signed-in user.",
                "entity_not_found": "I could not find the corresponding household entity.",
                "relationship_not_found": "No matching household relationship is recorded.",
                "property_unavailable": "The entity is recorded, but that semantic property is unavailable.",
                "relation_property_unavailable": (
                    "The relationship is recorded, but that semantic property is unavailable."
                ),
                "filter_input_missing": (
                    "Required evidence is unavailable for deterministic filtering."
                ),
                "filter_unsupported": "That semantic filter is not supported.",
                "operator_unsupported": "That semantic operator is not supported.",
                "semantic_plan_unsupported": (
                    "The request could not be expressed in the supported household query protocol."
                ),
                "ambiguous": "More than one household entity matches; please clarify which one.",
                "computation_input_missing": (
                    "A required semantic property is unavailable for this computation."
                ),
                "computation_impossible": "The available evidence is insufficient for that computation.",
                "collection_incomplete": "The result exceeds the completeness limit, so an exact total or complete list is unavailable.",
            }[result.status]
        if request.operation == "count":
            return f"The current count is {result.value}."
        if request.operation == "select" and request.property is None:
            if not result.value:
                if request.filters or any(step.filters for step in request.subject.path):
                    return "No records match the filters."
                return "No records match the query scope."
            return "Matching records: " + ", ".join(
                _name(item, "en") for item in result.value
            ) + "."
        if request.operation == "select":
            if request.property == "full_address":
                return f"The street address is {_format_address(result.value)}."
            return f"The requested value is {result.value}."
        if request.operation in {"date_difference", "duration", "completed_years"}:
            unit = result.unit or ("years" if request.operation == "completed_years" else request.mode)
            if unit == "years" and request.property == "birth_date" and request.property_source == "entity":
                return f"The age is {result.value} years."
            return f"The interval from the recorded date to now is {result.value} {unit}."
        if request.operation == "annual_occurrence":
            if request.mode == "days":
                days = int(result.value)
                return (
                    "The birthday is today."
                    if days == 0
                    else f"The birthday is in {days} days."
                )
            return f"The next birthday is {result.value}."
        if request.operation in {"argmin", "argmax"}:
            if request.other is not None and isinstance(result.value, Mapping):
                selected = _name(result.value.get("selected"), "en")
                other = _name(result.value.get("other"), "en")
                if request.property == "birth_date" and request.property_source == "entity":
                    if result.value.get("equal"):
                        return f"{selected} and {other} are the same age."
                    adjective = "older" if request.operation == "argmin" else "younger"
                    return f"{selected} is {adjective} than {other}."
                if result.value.get("equal"):
                    return f"{selected} and {other} have equal comparison values."
                return f"The {'minimum' if request.operation == 'argmin' else 'maximum'} belongs to {selected}."
            return f"The matching entity is {_name(result.value, 'en')}."
        if request.operation in {"sum", "average", "min", "max"}:
            return f"The computed result is {result.value}."
        if request.operation in {
            "unit_conversion",
        }:
            return f"The converted result is {result.value} {result.unit or request.to_unit}."
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"The matching result is {_name(result.value, 'en')}."
        if request.operation == "same_entity":
            return "Yes." if result.value else "No."
        if request.subject.kind == "assistant":
            return f"I am {context.assistant_display_name}, the Home Cortex household assistant."
        return f"The resolved person is {_name(result.value, 'en')}."




def _name(entity: Any, language: str) -> str:
    return resolve_display_name(entity, language) if isinstance(entity, Mapping) else str(entity)

def _relation_label(
    reference: SemanticReference,
    semantic_relation: str | None = None,
) -> str:
    return {
        "spouse": "配偶",
        "child": "亲子",
        "parent": "父母",
        "member": "家庭成员",
        "residence": "居住地",
        "location": "位置",
        "contents": "包含",
        "host": "承载位置",
        "hosted_space": "空间",
    }.get(semantic_relation or _last_relation(reference), "对应的")


def _subject_possessive(reference: SemanticReference) -> str:
    return f"{_subject_nominative(reference)}的"


def _subject_nominative(reference: SemanticReference) -> str:
    if not reference.path:
        if reference.kind == "self":
            return "您"
        if reference.kind == "named_entity" and reference.value:
            return reference.value
        return "对应实体"
    label = {
        "self": "您",
        "current_household": "家里",
        "named_entity": reference.value or "对应实体",
    }.get(reference.kind, "对应实体")
    for index, step in enumerate(reference.path):
        connector = "" if reference.kind == "self" and index == 0 else "的"
        label = f"{label}{connector}{_relation_noun(step)}"
    return label


def _relation_noun(step: SemanticRelationStep) -> str:
    gender = next(
        (item.value for item in step.filters if item.property == "gender"
         and item.source == "entity" and item.operator == "eq" and item.value_from is None
         and item.value in {"male", "female"}),
        None,
    )
    return {
        ("spouse", "female"): "妻子",
        ("spouse", "male"): "丈夫",
        ("spouse", None): "配偶",
        ("child", "male"): "儿子",
        ("child", "female"): "女儿",
        ("child", None): "孩子",
        ("parent", "male"): "父亲",
        ("parent", "female"): "母亲",
        ("parent", None): "父母",
        ("member", None): "家庭成员",
        ("residence", None): "住所",
        ("location", None): "所在位置",
        ("contents", None): "所含物品",
        ("host", None): "承载物",
        ("hosted_space", None): "空间",
    }.get((step.relation, gender), "关联实体")


def _property_label(properties: Sequence[str]) -> str:
    if not properties:
        return ""
    return {
        "birth_date": "出生日期",
        "display_name": "姓名",
        "given_name": "名字",
        "family_name": "姓氏",
        "form_of_address": "称呼",
        "gender": "性别",
        "household_role": "家庭角色",
        "start_date": "开始日期",
        "end_date": "结束日期",
        "adult": "成年人判断资料",
        "minor": "未成年人判断资料",
        "full_address": "具体住址",
    }.get(properties[0], "所需信息")


def _format_address(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    street = value.get("street")
    city = value.get("city")
    state = value.get("state")
    postal = value.get("zip") or value.get("postal_code")
    locality = ", ".join(str(item) for item in (city, state) if item)
    if postal:
        locality = f"{locality} {postal}".strip()
    return ", ".join(str(item) for item in (street, locality) if item)
