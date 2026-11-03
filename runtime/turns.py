"""Provider-neutral turn states; cancellation requests remain orthogonal until Step 23."""

from enum import Enum


class TurnStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


_TRANSITIONS = {
    TurnStatus.RUNNING: frozenset({TurnStatus.COMPLETE, TurnStatus.FAILED}),
    TurnStatus.COMPLETE: frozenset(),
    TurnStatus.FAILED: frozenset(),
}


def require_transition(current: TurnStatus, target: TurnStatus) -> None:
    if not isinstance(current, TurnStatus) or not isinstance(target, TurnStatus):
        raise ValueError("Turn transitions require declared statuses")
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid turn transition: {current.value} -> {target.value}")
