"""Execute a validated, allowlisted Pandas operation without generated code."""

import math

import pandas as pd
from langchain_core.documents import Document

from agents.table_plan import asset_id, build_tool_plan


def _plain(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _execute(plan: dict, assets: list[dict]) -> dict:
    matching = [asset for asset in assets if asset_id(asset) == plan["asset_id"]]
    if len(matching) != 1:
        raise ValueError("Unknown or ambiguous table")
    asset = matching[0]
    frame = asset["dataframe"]
    selected = frame
    row_numbers = list(range(1, len(frame) + 1))
    for condition in plan["filters"]:
        column = condition["column"]
        if condition["operator"] != "eq" or column not in frame.columns:
            raise ValueError("Invalid table filter")
        mask = selected[column] == condition["value"]
        positions = [index for index, keep in zip(row_numbers, mask.fillna(False)) if bool(keep)]
        selected = selected.loc[mask.fillna(False)]
        row_numbers = positions
    operation = plan["operation"]
    column = plan["column"]
    if operation == "schema":
        value = {"columns": [str(item) for item in frame.columns], "row_count": len(selected)}
    elif operation == "row_count":
        value = len(selected)
    elif operation == "count" and column is None:
        value = len(selected)
    else:
        if column not in frame.columns:
            raise ValueError("Unknown table column")
        series = selected[column]
        if operation != "count" and not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Numeric column required")
        if operation == "sum":
            value = series.sum(min_count=1)
        elif operation == "mean":
            value = series.mean()
        elif operation == "min":
            value = series.min()
        elif operation == "max":
            value = series.max()
        elif operation == "count":
            value = series.count()
        else:
            raise ValueError("Unsupported table operation")
        value = _plain(value)
    return {"source": asset["source"], "sheet": asset["sheet"], "asset_id": asset_id(asset),
            "operation": operation, "column": column, "filters": plan["filters"],
            "selected_rows": row_numbers, "row_count": len(selected),
            "null_policy": "skip missing values", "value": value}


def _format_result(result: dict) -> str:
    operation = result["operation"]
    value = result["value"]
    if operation == "schema":
        answer = f"Columns: {', '.join(value['columns'])}\nRow count: {value['row_count']}"
    elif value is None:
        answer = "No non-null values matched the requested calculation."
    elif operation == "row_count" or (operation == "count" and result["column"] is None):
        answer = f"Row count: {value}"
    else:
        label = {"sum": "Total", "mean": "Average", "min": "Min", "max": "Max", "count": "Count"}[operation]
        answer = f"{label}:\n- {result['column']}: {value}"
    rows = ", ".join(str(number) for number in result["selected_rows"]) or "none"
    return (f"{answer}\nSource: {result['source']} ({result['sheet']})\n"
            f"Rows used: {rows}\nMissing values: skipped")


def analyze_dataframe(dataframe, query):
    """Compatibility helper for one table, with the same validated semantics."""
    assets = [{"source": "table", "sheet": "Data", "file_type": "csv", "dataframe": dataframe}]
    plan = build_tool_plan(query, assets).to_dict()
    if plan["status"] != "ready":
        return {}
    result = _execute(plan, assets)
    if result["operation"] == "schema":
        return result["value"]
    if result["operation"] == "row_count" or (result["operation"] == "count" and result["column"] is None):
        return {"row_count": result["value"]}
    key = {"sum": "total", "mean": "average", "min": "min", "max": "max", "count": "count"}[result["operation"]]
    return {key: {result["column"]: result["value"]}}


def dataframe_tool_agent(state):
    query = state["query"]
    assets = state.get("tabular_assets", [])
    canonical = build_tool_plan(query, assets).to_dict()
    provided = state.get("tool_plan")
    if provided is not None and provided != canonical:
        canonical = build_tool_plan(query, assets).to_dict()
        if canonical["status"] == "ready":
            canonical = {**canonical, "status": "clarify", "message": "The table plan changed. Retry the request."}
    state["tool_plan"] = canonical
    if canonical["status"] != "ready":
        state["response"] = canonical["message"]
        state["result_status"] = "needs_clarification"
        state["documents"] = []
        return state
    try:
        result = _execute(canonical, assets)
    except ValueError:
        state["tool_plan"] = {**canonical, "status": "clarify", "message": "The selected table or column is unavailable. Retry the request."}
        state["response"] = state["tool_plan"]["message"]
        state["result_status"] = "needs_clarification"
        state["documents"] = []
        return state
    response = _format_result(result)
    state["tool_result"] = result
    state["result_status"] = "answered"
    state["source"] = f'{result["source"]} ({result["sheet"]})'
    state["documents"] = [Document(page_content=response, metadata={
        "source": result["source"], "page": f'Sheet {result["sheet"]}', "sheet": result["sheet"],
        "rows": result["selected_rows"], "file_type": "table", "content_type": "dataframe_result",
    })]
    state["response"] = response
    return state
