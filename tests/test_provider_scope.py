"""Request-local provider selection must not mutate process configuration."""
import os
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import tests  # Establish offline process settings before importing native modules.


class ProviderScopeTests(unittest.TestCase):
    def test_explicit_providers_select_independent_models(self):
        from agents.orchestrator_agent import invoke_orchestrator
        with patch("agents.orchestrator_agent._get_gemini_llm", return_value=SimpleNamespace(invoke=lambda p: "gemini")), \
                patch("agents.orchestrator_agent._get_ollama_llm", return_value=SimpleNamespace(invoke=lambda p: "ollama")):
            self.assertEqual(invoke_orchestrator("prompt", provider="gemini"), "gemini")
            self.assertEqual(invoke_orchestrator("prompt", provider="ollama"), "ollama")

    def test_fallback_does_not_change_environment(self):
        from agents.orchestrator_agent import invoke_orchestrator
        before = {key: os.environ.get(key) for key in ("LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_ACTIVE_MODEL")}
        def failed(prompt):
            raise RuntimeError("requires more system memory")
        with patch("agents.orchestrator_agent._get_ollama_settings", return_value=("large", "http://synthetic.invalid")), \
                patch("agents.orchestrator_agent._get_fallback_candidates", return_value=["small"]), \
                patch("agents.orchestrator_agent._get_ollama_llm", side_effect=[SimpleNamespace(invoke=failed), SimpleNamespace(invoke=lambda p: "fallback")]):
            self.assertEqual(invoke_orchestrator("prompt", provider="ollama"), "fallback")
        self.assertEqual({key: os.environ.get(key) for key in before}, before)
