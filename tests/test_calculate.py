import pytest

from home_cortex.calculate import CalculationError, evaluate_expression


def test_operator_precedence_matches_standard_arithmetic() -> None:
    assert evaluate_expression("2 + 3 * 4") == 14


def test_ticket_example_divides_annual_amount_by_days() -> None:
    value = evaluate_expression("(4350 * 12) / 365")

    assert isinstance(value, float)
    assert value == pytest.approx(143.01369863013698)


def test_allowlisted_functions_and_constants_are_numeric() -> None:
    assert evaluate_expression("sqrt(9) + abs(-1)") == 4
    assert evaluate_expression("round(pi, 2)") == pytest.approx(3.14)
    assert evaluate_expression("min(8, 2, 5)") == 2
    assert evaluate_expression("log(e)") == pytest.approx(1.0)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "eval('2')",
        "exec('x=1')",
        "(lambda: 1)()",
        "os.system('echo')",
        "''.join(['1'])",
        "open('/etc/passwd')",
        "__builtins__",
        "[x for x in [1, 2]]",
        "True",
        "'2' + '2'",
    ],
)
def test_rejects_arbitrary_code_and_non_numeric_syntax(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate_expression(expression)


def test_division_by_zero_is_a_calculation_error() -> None:
    with pytest.raises(CalculationError, match="Division by zero"):
        evaluate_expression("1 / 0")


def test_out_of_range_exponent_is_rejected() -> None:
    with pytest.raises(CalculationError, match="out of range"):
        evaluate_expression("10 ** 20")


def test_unknown_function_is_rejected() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        evaluate_expression("getattr(1, 2)")
