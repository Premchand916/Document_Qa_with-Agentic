"""Capability contracts are deterministic and filtered before model exposure."""

import unittest


class CapabilityRegistryTests(unittest.TestCase):
    @staticmethod
    def definition(capability_id="documents.search", **overrides):
        from domain.capabilities import CapabilityDefinition, CapabilityKind, RiskClass
        values = {
            "capability_id": capability_id,
            "kind": CapabilityKind.TOOL,
            "version": "1.0.0",
            "description": "Search authorized documents.",
            "risk": RiskClass.READ,
            "required_scopes": ("documents.read",),
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "output_schema": {"type": "object"},
        }
        values.update(overrides)
        return CapabilityDefinition(**values)

    def test_registry_is_deterministic_idempotent_and_rejects_conflicts(self):
        from capabilities.registry import CapabilityConflict, CapabilityRegistry
        first = self.definition("web.search")
        second = self.definition("documents.search")
        registry = CapabilityRegistry([first, second])
        registry.register(first)
        self.assertEqual([item.capability_id for item in registry.definitions()],
                         ["documents.search", "web.search"])
        with self.assertRaises(CapabilityConflict):
            registry.register(self.definition("web.search", version="2.0.0"))

    def test_policy_hides_unavailable_unauthorized_and_high_risk_capabilities(self):
        from capabilities.registry import CapabilityRegistry
        from domain.capabilities import RiskClass
        from policy.capabilities import CapabilityContext, CapabilityPolicy
        registry = CapabilityRegistry([
            self.definition("documents.search"),
            self.definition("documents.write", risk=RiskClass.WRITE,
                            required_scopes=("documents.write",)),
            self.definition("billing.charge", risk=RiskClass.CONSEQUENTIAL,
                            required_scopes=("billing.write",)),
            self.definition("hidden.search", available=False),
        ])
        context = CapabilityContext("alpha", "alice", frozenset({"documents.read"}))
        contracts = CapabilityPolicy().model_contracts(registry, context)
        self.assertEqual([contract["capability_id"] for contract in contracts],
                         ["documents.search"])
        self.assertNotIn("enabled", contracts[0])
        self.assertNotIn("available", contracts[0])

    def test_explicit_write_authority_exposes_only_matching_scope(self):
        from capabilities.registry import CapabilityRegistry
        from domain.capabilities import RiskClass
        from policy.capabilities import CapabilityContext, CapabilityPolicy
        definition = self.definition("documents.write", risk=RiskClass.WRITE,
                                     required_scopes=("documents.write",))
        registry = CapabilityRegistry([definition])
        denied = CapabilityContext("alpha", "alice", frozenset({"documents.write"}))
        allowed = CapabilityContext("alpha", "alice", frozenset({"documents.write"}),
                                    allow_write=True)
        self.assertEqual(CapabilityPolicy().model_contracts(registry, denied), [])
        self.assertEqual(CapabilityPolicy().model_contracts(registry, allowed)[0]["risk"], "write")

    def test_contract_validation_and_public_schema_copy(self):
        from domain.capabilities import CapabilityDefinition, CapabilityKind
        definition = self.definition()
        public = definition.public_contract()
        public["input_schema"]["properties"]["query"]["type"] = "integer"
        self.assertEqual(definition.input_schema["properties"]["query"]["type"], "string")
        with self.assertRaises(ValueError):
            self.definition("Unsafe ID")
        with self.assertRaises(ValueError):
            CapabilityDefinition("safe.id", "tool", "1", "description")
        with self.assertRaises(ValueError):
            self.definition(required_scopes=["documents.read"])
        with self.assertRaises(ValueError):
            self.definition(kind=CapabilityKind.TOOL, version="")

    def test_existing_chat_is_registered_and_turn_terminals_are_final(self):
        from capabilities.builtin import build_default_registry
        from runtime.turns import TurnStatus, require_transition
        registry = build_default_registry()
        self.assertEqual(registry.definitions()[0].capability_id, "assistant.chat")
        require_transition(TurnStatus.RUNNING, TurnStatus.COMPLETE)
        require_transition(TurnStatus.RUNNING, TurnStatus.FAILED)
        with self.assertRaises(ValueError):
            require_transition(TurnStatus.COMPLETE, TurnStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
