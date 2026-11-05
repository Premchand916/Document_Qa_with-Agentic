"""Provider-neutral terminal turn states."""

from enum import Enum


class TurnStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS = {
    TurnStatus.RUNNING: frozenset({TurnStatus.COMPLETE, TurnStatus.FAILED, TurnStatus.CANCELLED}),
    TurnStatus.COMPLETE: frozenset(),
    TurnStatus.FAILED: frozenset(),
    TurnStatus.CANCELLED: frozenset(),
}


def require_transition(current: TurnStatus, target: TurnStatus) -> None:
    if not isinstance(current, TurnStatus) or not isinstance(target, TurnStatus):
        raise ValueError("Turn transitions require declared statuses")
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid turn transition: {current.value} -> {target.value}")
