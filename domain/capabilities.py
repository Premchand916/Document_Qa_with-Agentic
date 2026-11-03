"""Typed capability contracts. Definitions grant no authority by themselves."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Mapping


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class CapabilityKind(str, Enum):
    TOOL = "tool"
    SKILL = "skill"
    PLUGIN = "plugin"
    CONNECTOR = "connector"


class RiskClass(str, Enum):
    READ = "read"
    WRITE = "write"
    CONSEQUENTIAL = "consequential"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    kind: CapabilityKind
    version: str
    description: str
    risk: RiskClass = RiskClass.READ
    required_scopes: tuple[str, ...] = ()
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CapabilityKind) or not isinstance(self.risk, RiskClass):
            raise ValueError("Capability kind and risk must use their declared enums")
        if not isinstance(self.enabled, bool) or not isinstance(self.available, bool):
            raise ValueError("Capability availability flags must be boolean")
        if not _IDENTIFIER.fullmatch(self.capability_id):
            raise ValueError("Invalid capability identifier")
        if not _VERSION.fullmatch(self.version):
            raise ValueError("Invalid capability version")
        description = " ".join(self.description.strip().split())
        if not 1 <= len(description) <= 500:
            raise ValueError("Capability description must contain 1 to 500 characters")
        if not isinstance(self.required_scopes, tuple):
            raise ValueError("Capability scopes must be a tuple")
        scopes = tuple(sorted(set(self.required_scopes)))
        if any(not isinstance(scope, str) or not _IDENTIFIER.fullmatch(scope) for scope in scopes):
            raise ValueError("Invalid capability scope")
        input_schema = deepcopy(dict(self.input_schema))
        output_schema = deepcopy(dict(self.output_schema))
        encoded = json.dumps({"input": input_schema, "output": output_schema}, allow_nan=False)
        if len(encoded) > 32_000:
            raise ValueError("Capability schemas exceed the size limit")
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "required_scopes", scopes)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)

    def public_contract(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "kind": self.kind.value,
            "version": self.version,
            "description": self.description,
            "risk": self.risk.value,
            "required_scopes": list(self.required_scopes),
            "input_schema": deepcopy(dict(self.input_schema)),
            "output_schema": deepcopy(dict(self.output_schema)),
        }
