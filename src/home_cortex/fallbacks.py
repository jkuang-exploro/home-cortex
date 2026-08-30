"""Localized fallback replies shared by deterministic facts and the model loop."""

from __future__ import annotations


def grounding_fallback(language: str) -> str:
    if language == "zh":
        return "老管家目前无法从家庭资料中核实这项信息。"
    return "I could not verify that information from the home graph."


def no_records_fallback(language: str) -> str:
    if language == "zh":
        return "家庭资料中没有找到与这个问题匹配的信息。"
    return "The home graph does not contain matching information for that request."
