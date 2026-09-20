"""Allowlisted local arithmetic. Never uses eval() or exec()."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 80
MAX_ABS_INT = 10**18
MAX_ABS_FLOAT = 1e15
MAX_POWER_ABS_EXPONENT = 12
MAX_FACTORIAL_ARGUMENT = 20

_ALLOWED_NODES = (
    ast.Add,
    ast.BinOp,
    ast.Call,
    ast.Constant,
    ast.Div,
    ast.Expression,
    ast.FloorDiv,
    ast.Load,
    ast.Mod,
    ast.Mult,
    ast.Name,
    ast.Pow,
    ast.Sub,
    ast.UAdd,
    ast.USub,
    ast.UnaryOp,
)

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


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


def _checked_pow(base: int | float, exponent: int | float) -> int | float:
    if abs(exponent) > MAX_POWER_ABS_EXPONENT:
        raise CalculationError("Exponent is out of range")
    try:
        return _finite_number(operator.pow(base, exponent))
    except OverflowError as error:
        raise CalculationError("Numeric result is out of range") from error


def _checked_factorial(value: int | float) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalculationError("factorial requires a non-negative integer")
    if value < 0 or value > MAX_FACTORIAL_ARGUMENT:
        raise CalculationError("factorial argument is out of range")
    return math.factorial(value)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "degrees": math.degrees,
    "exp": math.exp,
    "fabs": math.fabs,
    "factorial": _checked_factorial,
    "floor": math.floor,
    "hypot": math.hypot,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "max": max,
    "min": min,
    "pow": _checked_pow,
    "radians": math.radians,
    "round": round,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "sum": lambda *values: sum(values),
    "tan": math.tan,
    "tanh": math.tanh,
    "trunc": math.trunc,
}


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

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError("Expression contains disallowed syntax")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
        ):
            raise ValueError("Only numeric literals are allowed")

    return _finite_number(_eval_node(tree))


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return _literal_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _finite_number(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            if isinstance(node.op, ast.Pow):
                return _checked_pow(left, right)
            return _finite_number(_BIN_OPS[type(node.op)](left, right))
        except ZeroDivisionError as error:
            raise CalculationError("Division by zero") from error
        except OverflowError as error:
            raise CalculationError("Numeric result is out of range") from error
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        return _eval_call(node)
    raise ValueError("Expression contains disallowed syntax")


def _literal_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only numeric literals are allowed")
    return _finite_number(value)


def _eval_call(node: ast.Call) -> int | float:
    if not isinstance(node.func, ast.Name) or not isinstance(node.func.ctx, ast.Load):
        raise ValueError("Expression contains disallowed syntax")
    if node.keywords or getattr(node, "starargs", None) or getattr(node, "kwargs", None):
        raise ValueError("Keyword and starred arguments are not allowed")
    function = _FUNCTIONS.get(node.func.id)
    if function is None:
        raise ValueError(f"Function {node.func.id!r} is not allowed")
    arguments = [_eval_node(argument) for argument in node.args]
    try:
        result = function(*arguments)
    except CalculationError:
        raise
    except TypeError as error:
        raise ValueError(f"Invalid arguments for {node.func.id}") from error
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise CalculationError("Mathematical evaluation failed") from error
    if isinstance(result, bool):
        raise CalculationError("Expression did not produce a number")
    if isinstance(result, int):
        return _finite_number(result)
    if isinstance(result, float):
        return _finite_number(result)
    raise CalculationError("Expression did not produce a number")
