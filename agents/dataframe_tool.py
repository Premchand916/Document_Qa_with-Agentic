from langchain_core.documents import Document


def _numeric_summary(dataframe):
    numeric_df = dataframe.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    return {
        "total": numeric_df.sum().to_dict(),
        "average": numeric_df.mean().round(2).to_dict(),
        "max": numeric_df.max().to_dict(),
        "min": numeric_df.min().to_dict(),
        "count": numeric_df.count().to_dict(),
    }


def analyze_dataframe(dataframe, query):
    query_lower = query.lower()
    numeric_df = dataframe.select_dtypes(include="number")

    if "column" in query_lower or "schema" in query_lower or "fields" in query_lower:
        return {
            "columns": [str(column) for column in dataframe.columns],
            "row_count": int(len(dataframe.index)),
        }

    if "total" in query_lower or "sum" in query_lower:
        return {"total": numeric_df.sum().to_dict() if not numeric_df.empty else {}}

    if "average" in query_lower or "mean" in query_lower:
        return {"average": numeric_df.mean().round(2).to_dict() if not numeric_df.empty else {}}

    if "max" in query_lower or "highest" in query_lower:
        return {"max": numeric_df.max().to_dict() if not numeric_df.empty else {}}

    if "min" in query_lower or "lowest" in query_lower:
        return {"min": numeric_df.min().to_dict() if not numeric_df.empty else {}}

    if "count" in query_lower or "rows" in query_lower:
        return {"row_count": int(len(dataframe.index))}

    return _numeric_summary(dataframe)


def _format_tool_result(result):
    if not result:
        return "No computation result was produced from the uploaded table."

    lines = []
    for key, value in result.items():
        if isinstance(value, dict):
            if not value:
                lines.append(f"{key.title()}: no numeric columns available.")
            else:
                lines.append(f"{key.title()}:")
                for metric, metric_value in value.items():
                    lines.append(f"- {metric}: {metric_value}")
        elif isinstance(value, list):
            lines.append(f"{key.title()}: " + ", ".join(str(item) for item in value))
        else:
            lines.append(f"{key.title()}: {value}")

    return "\n".join(lines)


def dataframe_tool_agent(state):
    query = state["query"]
    tabular_assets = state.get("tabular_assets", [])

    if not tabular_assets:
        state["response"] = "No CSV or Excel data is available for tabular analysis."
        state["documents"] = []
        return state

    primary_asset = tabular_assets[0]
    result = analyze_dataframe(primary_asset["dataframe"], query)
    response = _format_tool_result(result)

    state["source"] = f'{primary_asset["source"]} ({primary_asset["sheet"]})'
    state["documents"] = [
        Document(
            page_content=response,
            metadata={
                "source": primary_asset["source"],
                "page": f'Sheet {primary_asset["sheet"]}',
                "sheet": primary_asset["sheet"],
                "file_type": primary_asset["file_type"],
                "content_type": "dataframe_result",
            },
        )
    ]
    state["response"] = response

    return state
