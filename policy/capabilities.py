"""Fail-closed capability filtering before model-visible contracts are built."""

from dataclasses import dataclass, field

from capabilities.registry import CapabilityRegistry
from domain.capabilities import CapabilityDefinition, RiskClass


@dataclass(frozen=True)
class CapabilityContext:
    workspace_id: str
    principal_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    allow_write: bool = False
    allow_consequential: bool = False
    allow_destructive: bool = False


@dataclass(frozen=True)
class CapabilityDecision:
    allowed: bool
    reason: str


class CapabilityPolicy:
    def evaluate(self, definition: CapabilityDefinition, context: CapabilityContext) -> CapabilityDecision:
        if not definition.enabled:
            return CapabilityDecision(False, "disabled")
        if not definition.available:
            return CapabilityDecision(False, "unavailable")
        missing = set(definition.required_scopes) - set(context.scopes)
        if missing:
            return CapabilityDecision(False, "missing_scope")
        if definition.risk is RiskClass.WRITE and not context.allow_write:
            return CapabilityDecision(False, "write_not_allowed")
        if definition.risk is RiskClass.CONSEQUENTIAL and not context.allow_consequential:
            return CapabilityDecision(False, "approval_required")
        if definition.risk is RiskClass.DESTRUCTIVE and not context.allow_destructive:
            return CapabilityDecision(False, "destructive_not_allowed")
        return CapabilityDecision(True, "allowed")

    def model_contracts(self, registry: CapabilityRegistry, context: CapabilityContext) -> list[dict]:
        return [definition.public_contract() for definition in registry.definitions()
                if self.evaluate(definition, context).allowed]
