"""Small, deterministic read-only tools; no generated code or host access."""

import ast
from datetime import datetime, timezone
from decimal import Decimal, DivisionByZero, InvalidOperation


class InvalidToolArguments(ValueError):
    """A model-proposed tool argument failed server validation."""


def _bounded_number(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > Decimal("1000000000000"):
        raise InvalidToolArguments("Number exceeds the calculator limit")
    return value


def evaluate_expression(arguments: dict) -> dict:
    if not isinstance(arguments, dict) or set(arguments) != {"expression"}:
        raise InvalidToolArguments("Expected one expression")
    expression = arguments["expression"]
    if not isinstance(expression, str) or not 1 <= len(expression) <= 120:
        raise InvalidToolArguments("Expression is outside the length limit")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise InvalidToolArguments("Invalid expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 25:
        raise InvalidToolArguments("Expression is too complex")

    def calculate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1_000_000_000:
                raise InvalidToolArguments("Operand exceeds the calculator limit")
            return _bounded_number(Decimal(str(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calculate(node.operand)
            return _bounded_number(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp):
            left, right = calculate(node.left), calculate(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.Pow) and right == right.to_integral_value() and 0 <= right <= 6:
                    result = left ** int(right)
                else:
                    raise InvalidToolArguments("Unsupported operation")
            except (DivisionByZero, InvalidOperation, OverflowError) as exc:
                raise InvalidToolArguments("Calculation is undefined") from exc
            return _bounded_number(result)
        raise InvalidToolArguments("Unsupported expression syntax")

    value = calculate(tree.body)
    return {"capability_id": "math.evaluate", "value": format(value.normalize(), "f")}


def current_utc(arguments: dict) -> dict:
    if arguments != {}:
        raise InvalidToolArguments("This tool takes no arguments")
    return {"capability_id": "time.current_utc",
            "value": datetime.now(timezone.utc).isoformat(timespec="seconds")}


READ_TOOL_HANDLERS = {
    "math.evaluate": evaluate_expression,
    "time.current_utc": current_utc,
}
