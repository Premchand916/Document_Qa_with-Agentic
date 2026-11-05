"""Policy-bounded skill listing, matching, and explicit activation."""

import re

from capabilities.registry import CapabilityRegistry
from capabilities.skills.loader import SkillNotFound, SkillPackageLoader
from capabilities.skills.state import SkillStateRepository
from domain.events import EventType
from domain.skills import SkillActivation, SkillDescriptor
from policy.capabilities import CapabilityContext, CapabilityPolicy
from runtime.events.repository import TurnEventRepository


class SkillUnavailable(Exception):
    """Skill is disabled, changed, or lacks authorized dependencies."""


class SkillAmbiguous(Exception):
    """Several skills match with equal specificity."""


class SkillRuntime:
    MAX_ACTIVE_GUIDANCE_BYTES = 24_000

    def __init__(self, loader: SkillPackageLoader, states: SkillStateRepository,
                 events: TurnEventRepository, capabilities: CapabilityRegistry,
                 policy: CapabilityPolicy | None = None):
        self.loader = loader
        self.states = states
        self.events = events
        self.capabilities = capabilities
        self.policy = policy or CapabilityPolicy()

    def _authorized_capabilities(self, descriptor: SkillDescriptor,
                                 context: CapabilityContext) -> tuple[str, ...]:
        allowed = []
        for capability_id in descriptor.manifest.required_capabilities:
            definition = self.capabilities.get(capability_id)
            if definition is None or not self.policy.evaluate(definition, context).allowed:
                raise SkillUnavailable()
            allowed.append(capability_id)
        return tuple(allowed)

    def list_public(self, workspace: str, principal: str,
                    *, scopes: frozenset[str] = frozenset()) -> list[dict]:
        context = CapabilityContext(workspace, principal, scopes)
        result = []
        for descriptor in self.loader.discover():
            enabled, reason = self.states.status(workspace, descriptor)
            authorized = True
            try:
                self._authorized_capabilities(descriptor, context)
            except SkillUnavailable:
                authorized = False
            result.append({**descriptor.public_contract(), "enabled": enabled and authorized,
                           "state_reason": reason,
                           "dependencies_authorized": authorized})
        return result

    @staticmethod
    def match(query: str, descriptors: tuple[SkillDescriptor, ...]) -> SkillDescriptor | None:
        normalized = " ".join(query.casefold().split())
        candidates: list[tuple[tuple[int, int], SkillDescriptor]] = []
        for descriptor in descriptors:
            matched = [trigger for trigger in descriptor.manifest.triggers
                       if re.search(rf"(?<!\w){re.escape(trigger.casefold())}(?!\w)", normalized)]
            if matched:
                best = max(matched, key=lambda value: (len(value.split()), len(value)))
                candidates.append(((len(best.split()), len(best)), descriptor))
        if not candidates:
            return None
        score = max(item[0] for item in candidates)
        winners = [descriptor for item_score, descriptor in candidates if item_score == score]
        if len(winners) != 1:
            raise SkillAmbiguous()
        return winners[0]

    def activate(self, workspace: str, principal: str, conversation_id: str, turn_id: str,
                 skill_id: str, *, scopes: frozenset[str] = frozenset()) -> SkillActivation:
        descriptor = self.loader.descriptor(skill_id)
        enabled, _ = self.states.status(workspace, descriptor)
        if not enabled:
            raise SkillUnavailable()
        context = CapabilityContext(workspace, principal, scopes)
        capabilities = self._authorized_capabilities(descriptor, context)
        loaded = self.loader.load(skill_id)
        guidance = [loaded.instructions]
        for resource, content in sorted(loaded.resources.items()):
            guidance.append(f"Skill resource {resource}:\n{content}")
        instructions = "\n\n".join(guidance)
        if len(instructions.encode("utf-8")) > self.MAX_ACTIVE_GUIDANCE_BYTES:
            raise SkillUnavailable()
        activation = SkillActivation(
            skill_id, descriptor.manifest.version, descriptor.digest,
            instructions, capabilities,
        )
        self.events.append(
            workspace, principal, conversation_id, turn_id, EventType.CAPABILITY_SELECTED,
            {"kind": "skill", **activation.public_contract()},
            dedupe_key=f"skill.activated:{skill_id}:{descriptor.digest}",
        )
        return activation
