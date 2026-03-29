import os
import re
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

ROOT_DIR = Path(__file__).resolve().parents[1]


def _candidate_skill_paths():
    configured_path = os.getenv("PROMPT_SKILL_PATH", "").strip()
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    candidates.extend(
        [
            ROOT_DIR / "prompt-skill.md",
            Path.home() / "Downloads" / "SKILL (1).md",
            Path.home().drive and Path("d:/prompt-skill.skill"),
        ]
    )

    unique_candidates = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = Path(candidate)
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _strip_front_matter(text):
    if not text.startswith("---"):
        return text.strip()

    parts = text.split("---", 2)
    if len(parts) < 3:
        return text.strip()
    return parts[2].strip()


def _read_skill_file(path):
    if path.suffix.lower() == ".skill":
        with ZipFile(path) as archive:
            skill_entries = [
                entry for entry in archive.namelist()
                if entry.endswith("/SKILL.md") or entry.endswith("SKILL.md")
            ]
            if not skill_entries:
                raise FileNotFoundError(f"No SKILL.md found inside {path}.")
            with archive.open(skill_entries[0]) as handle:
                return handle.read().decode("utf-8", errors="ignore")

    return path.read_text(encoding="utf-8", errors="ignore")


@lru_cache(maxsize=1)
def get_prompt_skill():
    for candidate_path in _candidate_skill_paths():
        if candidate_path.exists():
            raw_text = _read_skill_file(candidate_path)
            body = _strip_front_matter(raw_text)
            return {
                "path": str(candidate_path),
                "body": body,
            }

    return {
        "path": "",
        "body": "",
    }


def _extract_section(body, tag_name):
    match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", body, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return re.sub(r"\s+\n", "\n", match.group(1)).strip()


def _build_execution_profile(body):
    sections = []
    for tag_name in ("core_identity", "non_negotiables", "execution_protocol", "response_contract"):
        section = _extract_section(body, tag_name)
        if section:
            sections.append(section)

    if not sections:
        return ""

    profile = "\n\n".join(sections)
    return re.sub(r"\n{3,}", "\n\n", profile).strip()


def _format_history(chat_history, limit=4):
    lines = []
    for message in (chat_history or [])[-limit:]:
        role = message.get("role", "user").strip().lower()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_prompted_query(query, state):
    skill = get_prompt_skill()
    body = skill.get("body", "")
    if not body:
        return query

    execution_profile = _build_execution_profile(body)
    uploaded_file_types = ", ".join(state.get("uploaded_file_types", [])) or "unknown"
    assets = []
    if state.get("vectorstore") is not None:
        assets.append("FAISS semantic retrieval over uploaded files")
    if state.get("tabular_assets"):
        assets.append("Parsed spreadsheet/CSV tables")
    if state.get("use_web_search"):
        assets.append("Current web search via Tavily")
    if not assets:
        assets.append("Uploaded files in the current workspace")

    history_text = _format_history(state.get("chat_history", []))

    sections = []
    if execution_profile:
        sections.append(f"Execution profile:\n{execution_profile}")

    sections.append(
        "\n".join(
            [
                "Task:",
                query,
                "",
                "Context:",
                f"Document intelligence workspace with uploaded file types: {uploaded_file_types}.",
                f"Answer mode: {state.get('answer_mode', 'Auto')}. Audience: {state.get('audience', 'General')}. Depth: {state.get('response_depth', 'Balanced')}.",
                f"Web search enabled: {'yes' if state.get('use_web_search') else 'no'}.",
                history_text or "No prior conversation context.",
                "",
                "Available assets:",
                ", ".join(assets),
                "",
                "Constraints:",
                "Use grounded evidence from uploaded files and available web results.",
                "Keep the response actionable, concise, and accurate.",
                "",
                "Desired output:",
                "A direct answer that follows the requested mode and audience settings.",
                "",
                "Success criteria:",
                "Helpful, accurate, and well-structured output with minimal unsupported assumptions.",
                "",
                "Optional preferences:",
                f"Prefer the style implied by answer mode '{state.get('answer_mode', 'Auto')}'.",
            ]
        )
    )

    return "\n\n".join(section.strip() for section in sections if section.strip()).strip()
