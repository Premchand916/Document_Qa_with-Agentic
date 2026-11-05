"""Strict skill discovery, parsing, path validation, and digest verification."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from domain.skills import LoadedSkill, SkillDescriptor, SkillManifest


MAX_SKILLS = 100
MAX_SKILL_BYTES = 32_000
MAX_RESOURCE_BYTES = 64_000
MAX_PACKAGE_BYTES = 256_000
MAX_RESOURCES = 20
_FIELDS = frozenset({"name", "description", "metadata"})
_METADATA_FIELDS = frozenset({"version", "triggers", "required_capabilities",
                              "resources", "enabled_by_default"})
_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|override|bypass)\b.{0,40}\b(?:system|developer|security|policy|instructions?)\b|"
    r"\b(?:reveal|print|expose)\b.{0,30}\b(?:system prompt|hidden instructions?|credentials?|secrets?)\b",
    re.IGNORECASE | re.DOTALL,
)


class SkillError(Exception):
    """Base skill package error."""


class SkillIntegrityError(SkillError):
    """Package structure, content, or digest is unsafe or inconsistent."""


class SkillConflict(SkillIntegrityError):
    """Multiple packages claim one skill identity."""


class SkillNotFound(SkillError):
    """Requested skill is absent."""


def parse_skill_markdown(raw: str) -> tuple[SkillManifest, str]:
    if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
        raise SkillIntegrityError("SKILL.md requires bounded front matter")
    header, body = raw[4:].split("\n---\n", 1)
    values: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip() or line.startswith((" ", "\t")) or ":" not in line:
            raise SkillIntegrityError("Skill front matter must use one key-value pair per line")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in values or key not in _FIELDS or not value:
            raise SkillIntegrityError("Skill front matter contains invalid or duplicate fields")
        values[key] = value
    if set(values) != _FIELDS:
        raise SkillIntegrityError("Skill front matter fields are incomplete")
    try:
        metadata = json.loads(values["metadata"])
    except json.JSONDecodeError:
        raise SkillIntegrityError("Skill metadata must be one JSON object") from None
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
        raise SkillIntegrityError("Skill metadata fields are invalid")
    for field in ("triggers", "required_capabilities", "resources"):
        if (not isinstance(metadata[field], list)
                or any(not isinstance(item, str) or not item for item in metadata[field])):
            raise SkillIntegrityError(f"Skill metadata {field} must be a string list")
    if not isinstance(metadata["version"], str) or not isinstance(metadata["enabled_by_default"], bool):
        raise SkillIntegrityError("Skill metadata version/default state is invalid")
    instructions = body.strip()
    if not 1 <= len(instructions.encode("utf-8")) <= MAX_SKILL_BYTES:
        raise SkillIntegrityError("Skill instructions exceed size limits")
    if _OVERRIDE.search(instructions):
        raise SkillIntegrityError("Skill instructions contain an unsafe override")
    try:
        manifest = SkillManifest(
            skill_id=values["name"], version=metadata["version"],
            description=values["description"],
            triggers=tuple(metadata["triggers"]),
            required_capabilities=tuple(metadata["required_capabilities"]),
            resources=tuple(metadata["resources"]),
            enabled_by_default=metadata["enabled_by_default"],
        )
    except ValueError as exc:
        raise SkillIntegrityError(str(exc)) from None
    if len(manifest.resources) > MAX_RESOURCES:
        raise SkillIntegrityError("Skill declares too many resources")
    return manifest, instructions


def _safe_resource(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise SkillIntegrityError("Skill resource path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SkillIntegrityError("Skill resource path escapes its package")
    if path.parts[0] != "resources" or len(path.parts) < 2:
        raise SkillIntegrityError("Skill resources must live below resources/")
    return path


def _read_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SkillIntegrityError("Skill files must be regular non-symlink files")
    size = path.stat().st_size
    if size > limit:
        raise SkillIntegrityError("Skill file exceeds size limit")
    data = path.read_bytes()
    if len(data) != size:
        raise SkillIntegrityError("Skill file changed while reading")
    return data


def compute_package_digest(package: Path, manifest: SkillManifest) -> str:
    entries = [("SKILL.md", _read_regular(package / "SKILL.md", MAX_SKILL_BYTES))]
    total = len(entries[0][1])
    for resource in manifest.resources:
        relative = _safe_resource(resource)
        candidate = package
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise SkillIntegrityError("Skill resource path contains a symlink")
        data = _read_regular(candidate, MAX_RESOURCE_BYTES)
        total += len(data)
        entries.append((relative.as_posix(), data))
    if total > MAX_PACKAGE_BYTES:
        raise SkillIntegrityError("Skill package exceeds size limit")
    digest = hashlib.sha256()
    for name, data in sorted(entries):
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class SkillPackageLoader:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _descriptor(self, package: Path) -> SkillDescriptor:
        if package.is_symlink() or not package.is_dir():
            raise SkillIntegrityError("Skill packages must be regular directories")
        raw_bytes = _read_regular(package / "SKILL.md", MAX_SKILL_BYTES)
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise SkillIntegrityError("SKILL.md must be UTF-8") from None
        manifest, _ = parse_skill_markdown(raw)
        if package.name != manifest.skill_id:
            raise SkillIntegrityError("Skill directory must match manifest name")
        declared = {"SKILL.md", "skill.lock.json", *manifest.resources}
        for candidate in package.rglob("*"):
            if candidate.is_symlink():
                raise SkillIntegrityError("Skill package contains a symlink")
            if candidate.is_file() and candidate.relative_to(package).as_posix() not in declared:
                raise SkillIntegrityError("Skill package contains an undeclared file")
        lock_bytes = _read_regular(package / "skill.lock.json", 2048)
        try:
            lock = json.loads(lock_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SkillIntegrityError("Skill lock is invalid") from None
        if set(lock) != {"name", "version", "sha256"}:
            raise SkillIntegrityError("Skill lock fields are invalid")
        digest = compute_package_digest(package, manifest)
        if (lock["name"] != manifest.skill_id or lock["version"] != manifest.version
                or lock["sha256"] != digest or not re.fullmatch(r"[a-f0-9]{64}", digest)):
            raise SkillIntegrityError("Skill digest lock does not match package content")
        return SkillDescriptor(manifest, digest, str(package))

    def discover(self) -> tuple[SkillDescriptor, ...]:
        if not self.root.exists():
            return ()
        if self.root.is_symlink() or not self.root.is_dir():
            raise SkillIntegrityError("Skill root must be a regular directory")
        packages = sorted(path for path in self.root.iterdir() if not path.name.startswith("."))
        if len(packages) > MAX_SKILLS:
            raise SkillIntegrityError("Skill count exceeds limit")
        descriptors: dict[str, SkillDescriptor] = {}
        for package in packages:
            descriptor = self._descriptor(package)
            skill_id = descriptor.manifest.skill_id
            if skill_id in descriptors:
                raise SkillConflict(f"Duplicate skill: {skill_id}")
            descriptors[skill_id] = descriptor
        return tuple(descriptors[key] for key in sorted(descriptors))

    def descriptor(self, skill_id: str) -> SkillDescriptor:
        for descriptor in self.discover():
            if descriptor.manifest.skill_id == skill_id:
                return descriptor
        raise SkillNotFound()

    def load(self, skill_id: str) -> LoadedSkill:
        descriptor = self.descriptor(skill_id)
        package = Path(descriptor.package_path)
        raw = _read_regular(package / "SKILL.md", MAX_SKILL_BYTES).decode("utf-8")
        manifest, instructions = parse_skill_markdown(raw)
        digest = compute_package_digest(package, manifest)
        if digest != descriptor.digest:
            raise SkillIntegrityError("Skill changed during activation")
        resources = {}
        for resource in manifest.resources:
            data = _read_regular(package / resource, MAX_RESOURCE_BYTES)
            try:
                resources[resource] = data.decode("utf-8")
            except UnicodeDecodeError:
                raise SkillIntegrityError("Skill resources must be UTF-8") from None
        return LoadedSkill(descriptor, instructions, resources)
