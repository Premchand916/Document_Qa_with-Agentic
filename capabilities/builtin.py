"""Definitions for proven capabilities that predate the registry."""

from capabilities.registry import CapabilityRegistry
from domain.capabilities import CapabilityDefinition, CapabilityKind, RiskClass


def build_default_registry() -> CapabilityRegistry:
    return CapabilityRegistry([
        CapabilityDefinition(
            capability_id="assistant.chat",
            kind=CapabilityKind.TOOL,
            version="1.0.0",
            description="Run the existing bounded document or general-chat workflow.",
            risk=RiskClass.READ,
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
        ),
        CapabilityDefinition(
            capability_id="math.evaluate",
            kind=CapabilityKind.TOOL,
            version="1.0.0",
            description="Evaluate bounded arithmetic without running code.",
            risk=RiskClass.READ,
            input_schema={"type": "object", "required": ["expression"],
                          "properties": {"expression": {"type": "string", "minLength": 1,
                                                        "maxLength": 120}},
                          "additionalProperties": False},
            output_schema={"type": "object", "required": ["value"],
                           "properties": {"value": {"type": "string"}}},
        ),
        CapabilityDefinition(
            capability_id="time.current_utc",
            kind=CapabilityKind.TOOL,
            version="1.0.0",
            description="Read the server's current UTC date and time.",
            risk=RiskClass.READ,
            input_schema={"type": "object", "required": [], "properties": {},
                          "additionalProperties": False},
            output_schema={"type": "object", "required": ["value"],
                           "properties": {"value": {"type": "string"}}},
        ),
        CapabilityDefinition(
            capability_id="memory.save_preference",
            kind=CapabilityKind.TOOL,
            version="1.0.0",
            description="Propose saving a user preference; explicit user approval is required.",
            risk=RiskClass.WRITE,
            required_scopes=("memory.write",),
            input_schema={"type": "object", "required": ["key", "value"],
                          "properties": {"key": {"type": "string", "minLength": 1,
                                                 "maxLength": 64},
                                         "value": {"type": "string", "minLength": 1,
                                                   "maxLength": 1000}},
                          "additionalProperties": False},
            output_schema={"type": "object", "required": ["memory_id"],
                           "properties": {"memory_id": {"type": "string"}}},
        ),
    ])
