"""Versioned skill contracts. Skill text cannot grant runtime authority."""

from dataclasses import dataclass
import re


SKILL_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SKILL_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    version: str
    description: str
    triggers: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    resources: tuple[str, ...]
    enabled_by_default: bool

    def __post_init__(self) -> None:
        if not SKILL_ID.fullmatch(self.skill_id):
            raise ValueError("Invalid skill identifier")
        if not SKILL_VERSION.fullmatch(self.version):
            raise ValueError("Skill version must use semantic x.y.z form")
        description = " ".join(self.description.strip().split())
        if not 1 <= len(description) <= 500:
            raise ValueError("Skill description must contain 1 to 500 characters")
        if not isinstance(self.enabled_by_default, bool):
            raise ValueError("Skill default state must be boolean")
        if not 1 <= len(self.triggers) <= 32:
            raise ValueError("Skill must declare 1 to 32 triggers")
        triggers = tuple(sorted({" ".join(value.strip().lower().split())
                                 for value in self.triggers}))
        if any(not 2 <= len(value) <= 100 or not re.fullmatch(r"[\w .:/+-]+", value, re.UNICODE)
               for value in triggers):
            raise ValueError("Invalid skill trigger")
        capabilities = tuple(sorted(set(self.required_capabilities)))
        if any(not CAPABILITY_ID.fullmatch(value) for value in capabilities):
            raise ValueError("Invalid required capability")
        resources = tuple(sorted(set(self.resources)))
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "triggers", triggers)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "resources", resources)


@dataclass(frozen=True)
class SkillDescriptor:
    manifest: SkillManifest
    digest: str
    package_path: str

    def public_contract(self) -> dict:
        return {
            "skill_id": self.manifest.skill_id,
            "version": self.manifest.version,
            "description": self.manifest.description,
            "triggers": list(self.manifest.triggers),
            "required_capabilities": list(self.manifest.required_capabilities),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class LoadedSkill:
    descriptor: SkillDescriptor
    instructions: str
    resources: dict[str, str]


@dataclass(frozen=True)
class SkillActivation:
    skill_id: str
    version: str
    digest: str
    instructions: str
    capabilities: tuple[str, ...]

    def public_contract(self) -> dict:
        return {"skill_id": self.skill_id, "version": self.version,
                "digest": self.digest, "capabilities": list(self.capabilities)}
