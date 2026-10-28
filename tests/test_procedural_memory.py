"""Procedural guidance is code/operator-owned, versioned, and never auto-discovered."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import prompt_skill


class ProceduralMemoryTests(unittest.TestCase):
    def tearDown(self):
        prompt_skill.get_prompt_skill.cache_clear()

    def test_default_discovery_is_repository_only(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(prompt_skill, "ROOT_DIR", Path(folder)), \
                patch.dict(prompt_skill.os.environ, {}, clear=True):
            prompt_skill.get_prompt_skill.cache_clear()
            self.assertEqual(prompt_skill._candidate_skill_paths(), [Path(folder) / "prompt-skill.md"])
            self.assertEqual(prompt_skill.get_prompt_skill()["source"], "none")

    def test_explicit_operator_skill_has_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            skill = Path(folder) / "approved.skill.md"
            raw = "# Approved procedure\nUse grounded citations.\n"
            skill.write_text(raw)
            with patch.dict(prompt_skill.os.environ, {"PROMPT_SKILL_PATH": str(skill)}, clear=True), \
                    patch.object(prompt_skill, "ROOT_DIR", Path(folder) / "repo"):
                prompt_skill.get_prompt_skill.cache_clear()
                loaded = prompt_skill.get_prompt_skill()
            self.assertEqual(loaded["source"], "operator_configured")
            self.assertEqual(loaded["path"], str(skill))
            self.assertEqual(loaded["sha256"], hashlib.sha256(raw.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
