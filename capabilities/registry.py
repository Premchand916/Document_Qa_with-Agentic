"""Deterministic in-process capability registry."""

from threading import RLock

from domain.capabilities import CapabilityDefinition


class CapabilityConflict(ValueError):
    """A capability ID was registered with another definition."""


class CapabilityRegistry:
    def __init__(self, definitions: list[CapabilityDefinition] | None = None):
        self._lock = RLock()
        self._definitions: dict[str, CapabilityDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: CapabilityDefinition) -> None:
        if not isinstance(definition, CapabilityDefinition):
            raise TypeError("Capability registry accepts CapabilityDefinition values")
        with self._lock:
            existing = self._definitions.get(definition.capability_id)
            if existing is not None and existing != definition:
                raise CapabilityConflict(f"Capability already registered: {definition.capability_id}")
            self._definitions[definition.capability_id] = definition

    def get(self, capability_id: str) -> CapabilityDefinition | None:
        with self._lock:
            return self._definitions.get(capability_id)

    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        with self._lock:
            return tuple(self._definitions[key] for key in sorted(self._definitions))
