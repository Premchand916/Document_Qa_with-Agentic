"""Read-only calculations bound to the evidence passed to grounded QA."""

import ast
from decimal import Decimal
import json
import re

from agents.evidence import prepare_evidence
from capabilities.read_tools import InvalidToolArguments
from runtime.tool_loop import ToolLoopRejected


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_CALCULATION_INTENT = re.compile(
    r"\b(?:add|sum|total|calculate|compute|subtract|difference|multiply|divide|average|"
    r"percentage|percent|ratio)\b|\d\s*[+*/-]\s*\d", re.IGNORECASE,
)
MAX_EVIDENCE_RECORDS = 6
MAX_EVIDENCE_TEXT = 700


def _evidence_bound_result(records: list[dict], capability_id: str,
                           arguments: dict, result: dict) -> dict:
    if capability_id != "math.evaluate":
        raise InvalidToolArguments("Grounded mode permits calculations only")
    tree = ast.parse(arguments["expression"], mode="eval")
    literals = [Decimal(str(node.value)) for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and type(node.value) in (int, float)]
    if not literals:
        raise InvalidToolArguments("Calculation has no evidence-bound numbers")
    matched = set()
    for literal in literals:
        for record in records:
            values = {Decimal(value) for value in _NUMBER.findall(record["text"])}
            if abs(literal) in values:
                matched.add(record["evidence_id"])
                break
        else:
            raise InvalidToolArguments("Calculation includes a number absent from evidence")
    return {**result, "evidence_ids": sorted(matched)}


def grounded_tools_agent(state: dict) -> dict:
    runtime = state.get("grounded_tool_runtime")
    if runtime is None or not _CALCULATION_INTENT.search(state.get("query", "")):
        return state
    documents = prepare_evidence(state)
    records = [{"evidence_id": document.metadata["evidence_id"],
                "source": str(document.metadata.get("source", "Unknown"))[:160],
                "text": document.page_content[:MAX_EVIDENCE_TEXT]}
               for document in documents[:MAX_EVIDENCE_RECORDS]]
    if not any(_NUMBER.search(record["text"]) for record in records):
        return state
    context = state["request_context"]
    query = json.dumps({"question": state["query"], "evidence": records},
                       ensure_ascii=False, separators=(",", ":"))
    try:
        run = runtime.run(
            query, state["provider"], context["workspace_id"], context["principal_id"],
            state["conversation_id"], state["turn_id"],
            attempt_id=context.get("_turn_attempt"),
            allowed_capability_ids=frozenset({"math.evaluate"}),
            result_validator=lambda capability_id, arguments, result: _evidence_bound_result(
                records, capability_id, arguments, result),
            stop_after_tool=True,
        )
    except ToolLoopRejected as exc:
        state["builtin_tool_status"] = {"status": "failed", "reason": exc.reason}
        return state
    state["builtin_tool_results"] = run["tool_results"]
    return state
