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
    ])
