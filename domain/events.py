"""Safe, user-visible turn-event contracts; never hidden chain-of-thought."""

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any


class EventType(str, Enum):
    TURN_STARTED = "turn.started"
    TURN_CANCEL_REQUESTED = "turn.cancel_requested"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_CANCELLED = "turn.cancelled"
    TURN_PROGRESS = "turn.progress"
    MESSAGE_COMPLETED = "message.completed"
    CAPABILITY_SELECTED = "capability.selected"
    TOOL_REQUESTED = "tool.requested"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_RESOLVED = "approval.resolved"
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"


_SECRET_KEY = re.compile(
    r"^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|credential|"
    r"authorization|client[_-]?secret|private[_-]?key)$",
    re.IGNORECASE,
)


def validated_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Event payload must be an object")

    def inspect(value: Any, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("Event payload nesting exceeds the limit")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or _SECRET_KEY.fullmatch(key):
                    raise ValueError("Event payload contains a prohibited key")
                inspect(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inspect(child, depth + 1)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("Event payload contains an unsupported value")

    inspect(payload)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16_000:
        raise ValueError("Event payload exceeds the byte limit")
    return json.loads(encoded)


@dataclass(frozen=True)
class TurnEvent:
    event_id: str
    workspace_id: str
    principal_id: str
    conversation_id: str
    turn_id: str
    sequence: int
    event_type: EventType
    payload: dict[str, Any]
    created_at: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "created_at": self.created_at,
        }
