"""Allowlisted local arithmetic evaluated by simpleeval. Never uses eval() or exec()."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

import simpleeval

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 80
MAX_ABS_INT = 10**18
MAX_ABS_FLOAT = 1e15
MAX_POWER_ABS_EXPONENT = 12
MAX_FACTORIAL_ARGUMENT = 20

_EVAL_NODES = {ast.BinOp, ast.Call, ast.Constant, ast.Name, ast.UnaryOp}


class CalculationError(ValueError):
    """Raised when a valid expression cannot be evaluated numerically."""


def _finite_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculationError("Expression did not produce a number")
    if isinstance(value, int):
        if abs(value) > MAX_ABS_INT:
            raise CalculationError("Numeric result is out of range")
        return value
    if not math.isfinite(value) or abs(value) > MAX_ABS_FLOAT:
        raise CalculationError("Numeric result is out of range")
    return value


def _checked_pow(base: int | float, exponent: int | float) -> Any:
    # Reject before the power is computed so a huge exponent cannot allocate.
    if abs(exponent) > MAX_POWER_ABS_EXPONENT:
        raise CalculationError("Exponent is out of range")
    try:
        return operator.pow(base, exponent)
    except OverflowError as error:
        raise CalculationError("Numeric result is out of range") from error


def _checked_factorial(value: int | float) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalculationError("factorial requires a non-negative integer")
    if value < 0 or value > MAX_FACTORIAL_ARGUMENT:
        raise CalculationError("factorial argument is out of range")
    return math.factorial(value)


def _limit(function: Callable[..., Any]) -> Callable[..., int | float]:
    def wrapped(*args: Any) -> int | float:
        try:
            return _finite_number(function(*args))
        except ZeroDivisionError as error:
            raise CalculationError("Division by zero") from error
        except OverflowError as error:
            raise CalculationError("Numeric result is out of range") from error

    return wrapped


def _call(name: str, function: Callable[..., Any]) -> Callable[..., int | float]:
    def wrapped(*args: Any, **kwargs: Any) -> int | float:
        if kwargs:
            raise ValueError("Expression contains disallowed syntax")
        try:
            result = function(*args)
        except CalculationError:
            raise
        except TypeError as error:
            raise ValueError(f"Invalid arguments for {name}") from error
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise CalculationError("Mathematical evaluation failed") from error
        return _finite_number(result)

    return wrapped


_OPERATORS = {
    ast.Add: _limit(operator.add),
    ast.Sub: _limit(operator.sub),
    ast.Mult: _limit(operator.mul),
    ast.Div: _limit(operator.truediv),
    ast.FloorDiv: _limit(operator.floordiv),
    ast.Mod: _limit(operator.mod),
    ast.Pow: _limit(_checked_pow),
    ast.UAdd: _limit(operator.pos),
    ast.USub: _limit(operator.neg),
}
_FUNCTIONS = {
    name: _call(name, function)
    for name, function in {
        **{
            name: getattr(math, name)
            for name in (
                "acos asin atan atan2 ceil cos cosh degrees exp fabs floor hypot "
                "log log10 log2 radians sin sinh sqrt tan tanh trunc"
            ).split()
        },
        "abs": abs,
        "factorial": _checked_factorial,
        "max": max,
        "min": min,
        "pow": _checked_pow,
        "round": round,
        "sum": lambda *values: sum(values),
    }.items()
}
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    *_OPERATORS,
)


class _ArithmeticEval(simpleeval.SimpleEval):
    def __init__(self) -> None:
        super().__init__(
            operators=_OPERATORS,
            functions=_FUNCTIONS,
            names=_NAMES,
            allowed_attrs={},
        )
        self.nodes = {
            kind: handler for kind, handler in self.nodes.items() if kind in _EVAL_NODES
        }

    @staticmethod
    def _eval_constant(node: ast.Constant) -> int | float:
        return _finite_number(node.value)

    def _eval_name(self, node: ast.Name) -> int | float:
        try:
            return self.names[node.id]
        except KeyError:
            raise ValueError(f"Unknown name {node.id!r}") from None


def _reject_disallowed(tree: ast.AST) -> None:
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError("Expression contains disallowed syntax")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("Only numeric literals are allowed")


def evaluate_expression(expression: str) -> int | float:
    """Evaluate an allowlisted arithmetic expression to a number."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    source = expression.strip()
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression exceeds the maximum allowed length")
    try:
        tree = ast.parse(source, filename="<calculate>", mode="eval")
    except SyntaxError as error:
        raise ValueError("Expression is not valid arithmetic") from error
    _reject_disallowed(tree)
    try:
        value = _ArithmeticEval().eval(source, previously_parsed=tree.body)
    except simpleeval.FunctionNotDefined as error:
        raise ValueError(f"Function {error.func_name!r} is not allowed") from error
    except simpleeval.InvalidExpression as error:
        raise ValueError("Expression contains disallowed syntax") from error
    return _finite_number(value)
