"""Deterministic, server-owned plans for the supported table operations."""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Literal

import pandas as pd


Operation = Literal["sum", "mean", "min", "max", "count", "row_count", "schema"]


@dataclass(frozen=True)
class TypedFilter:
    column: str
    operator: Literal["eq"]
    value: Any


@dataclass(frozen=True)
class ToolPlan:
    status: Literal["ready", "clarify"]
    asset_id: str | None = None
    operation: Operation | None = None
    column: str | None = None
    filters: tuple[TypedFilter, ...] = field(default_factory=tuple)
    message: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result["filters"] = [asdict(item) for item in self.filters]
        return result


def asset_id(asset: dict) -> str:
    return str(asset.get("asset_id") or f'{asset["source"]}::{asset["sheet"]}')


def _matches(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE))


def operation_for(query: str) -> Operation | None:
    if re.search(r"\b(median|join|group\s+by|ratio|percent|percentage|correlation|plot|chart|trend|standard\s+deviation|difference)\b", query, re.I):
        return None
    if re.search(r"\b(columns?|schema|fields?)\b", query, re.I):
        return "schema"
    if re.search(r"\b(how many|number of|count)\s+rows?\b|\brows?\s+count\b", query, re.I):
        return "row_count"
    for operation, pattern in (
        ("sum", r"\b(total|sum)\b"),
        ("mean", r"\b(average|mean)\b"),
        ("max", r"\b(max|maximum|highest)\b"),
        ("min", r"\b(min|minimum|lowest)\b"),
        ("count", r"\bcount\b"),
    ):
        if re.search(pattern, query, re.I):
            return operation
    return None


def is_table_request(query: str, assets: list[dict]) -> bool:
    mentioned = any(_matches(query, str(asset["source"])) for asset in assets)
    if mentioned:
        return True
    if re.search(r"\b(pdf|document|page)\b", query, re.I):
        return False
    return bool(operation_for(query) or re.search(r"\b(csv|spreadsheet|excel|table|sheet|median|join|group by)\b", query, re.I))


def _clarify(message: str) -> ToolPlan:
    return ToolPlan(status="clarify", message=message)


def _python_value(value):
    return value.item() if hasattr(value, "item") else value


def _filter(query: str, dataframe: pd.DataFrame) -> tuple[TypedFilter | None, str | None, str]:
    explicit = re.search(r"\bwhere\s+(.+?)\s*(?:\s=\s|\s+is\s+|\s+equals\s+)(.+?)\s*$", query, re.I)
    only = re.search(r"\bfor\s+(.+?)\s+only\b", query, re.I)
    if not explicit and not only:
        return None, None, query
    if explicit:
        column_text = explicit.group(1).strip(" '\"?")
        value_text = explicit.group(2).strip(" '\"?.")
        columns = [col for col in dataframe.columns if str(col).casefold() == column_text.casefold()]
        if len(columns) != 1:
            return None, f"Unknown filter column '{column_text}'. Choose a column from: {', '.join(map(str, dataframe.columns))}.", query
        candidates = columns
        remaining = query[:explicit.start()]
    else:
        value_text = only.group(1).strip(" '\"?.")
        candidates = [col for col in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
        remaining = query[:only.start()] + query[only.end():]
    matches = []
    for column in candidates:
        for value in dataframe[column].dropna().unique():
            if str(value).casefold() == value_text.casefold():
                matches.append((str(column), _python_value(value)))
                break
    if len(matches) != 1:
        return None, f"Filter value '{value_text}' is unknown or ambiguous. Specify a column and value.", query
    column, value = matches[0]
    return TypedFilter(column, "eq", value), None, remaining


def _unknown_column_words(query: str) -> list[str]:
    cleaned = re.sub(r"\b(total|sum|average|mean|max|maximum|highest|min|minimum|lowest|count|how|many|number|of|the|in|for|only|rows?|where|is|equals|what|are|show|me|please|a|an|by|sheet|table|csv|excel|spreadsheet|columns?|schema|fields?)\b", " ", query, flags=re.I)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)


def build_tool_plan(query: str, assets: list[dict]) -> ToolPlan:
    if not assets:
        return _clarify("No CSV or Excel table is available. Upload a table first.")
    mentioned = [asset for asset in assets if _matches(query, str(asset["source"]))]
    candidates = mentioned or assets
    sheet_mentions = [asset for asset in candidates if _matches(query, f'sheet {asset["sheet"]}')]
    if sheet_mentions:
        candidates = sheet_mentions
    if len(candidates) != 1:
        labels = ", ".join(sorted({f'{asset["source"]} ({asset["sheet"]})' for asset in candidates}))
        return _clarify(f"Which table should I use? Choose one: {labels}.")
    asset = candidates[0]
    dataframe = asset["dataframe"]
    if re.search(r"\b(median|join|group\s+by|ratio|percent|percentage|correlation|plot|chart|trend|standard\s+deviation|difference)\b", query, re.I):
        return _clarify("That table operation is not supported yet. Use sum, average, min, max, count, rows, or schema.")
    operation = operation_for(query)
    if operation is None:
        return _clarify("Specify a supported table operation: sum, average, min, max, count, rows, or schema.")
    typed_filter, problem, remaining = _filter(query, dataframe)
    if problem:
        return _clarify(problem)
    filters = (typed_filter,) if typed_filter else ()
    column = None
    if operation not in {"schema", "row_count"}:
        numeric = [str(col) for col in dataframe.select_dtypes(include="number").columns]
        matched = [col for col in dataframe.columns if _matches(remaining, str(col))]
        if len(matched) > 1:
            return _clarify("Choose one column for this calculation.")
        if matched:
            column = str(matched[0])
        elif operation != "count" and len(numeric) == 1:
            column = numeric[0]
        elif operation != "count":
            return _clarify(f"Choose a numeric column: {', '.join(numeric) or 'none available'}.")
        for candidate in (str(asset["source"]), f'sheet {asset["sheet"]}', column or ""):
            if candidate:
                remaining = re.sub(rf"(?<!\w){re.escape(candidate)}(?!\w)", " ", remaining, flags=re.I)
        if _unknown_column_words(remaining):
            return _clarify(f"Unknown column in request. Available columns: {', '.join(map(str, dataframe.columns))}.")
        if operation != "count" and column not in numeric:
            return _clarify(f"Column '{column}' is not numeric. Choose: {', '.join(numeric) or 'none available'}.")
    return ToolPlan("ready", asset_id(asset), operation, column, filters)
