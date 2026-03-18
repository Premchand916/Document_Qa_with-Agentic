import re
from collections import defaultdict

from agents.orchestrator_agent import create_orchestrator


def extract_text(response):
    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if hasattr(item, "text") and item.text:
                    texts.append(item.text)
                elif isinstance(item, dict) and item.get("text"):
                    texts.append(item["text"])
                else:
                    texts.append(str(item))
            return " ".join(texts)

    return str(response)


def _clean_question(line):
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
    if not cleaned:
        return ""
    if not cleaned.endswith("?"):
        cleaned = f"{cleaned.rstrip('.')}?"
    return cleaned


def _fallback_questions(documents):
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown document")
        if source not in sources:
            sources.append(source)

    if len(sources) <= 1:
        source_name = sources[0] if sources else "your uploaded documents"
        return [
            f"What are the main themes covered in {source_name}?",
            "What important facts, figures, or dates are mentioned in the documents?",
            "What risks, blockers, or challenges are described in the documents?",
            "What recommendations or next steps do the documents suggest?",
            "Can you summarize the uploaded documents for a quick executive review?",
        ]

    joined_sources = ", ".join(sources[:3])
    if len(sources) > 3:
        joined_sources = f"{joined_sources}, and {len(sources) - 3} more"

    return [
        f"What are the common themes across these documents: {joined_sources}?",
        "How do the uploaded documents differ in their findings or recommendations?",
        "What important facts, figures, or dates appear across the document set?",
        "What risks, blockers, or open questions are mentioned across the uploaded documents?",
        "Can you give a combined executive summary of all uploaded documents?",
    ]


def _build_balanced_document_excerpt(documents, max_sources=8, chunks_per_source=2, max_chars=650):
    docs_by_source = defaultdict(list)

    for doc in documents:
        source = doc.metadata.get("source", "Unknown document")
        docs_by_source[source].append(doc)

    excerpt_sections = []
    for source in list(docs_by_source.keys())[:max_sources]:
        source_docs = docs_by_source[source][:chunks_per_source]
        snippet_parts = []
        for index, doc in enumerate(source_docs, start=1):
            text = re.sub(r"\s+", " ", doc.page_content).strip()
            if not text:
                continue
            snippet_parts.append(f"Excerpt {index}: {text[:max_chars]}")

        if snippet_parts:
            excerpt_sections.append(f"Document: {source}\n" + "\n".join(snippet_parts))

    return "\n\n".join(excerpt_sections)


def generate_questions(documents):
    if not documents:
        return []

    text = _build_balanced_document_excerpt(documents)
    llm = create_orchestrator()

    prompt = f"""
Generate 5 concise, high-value suggested questions based on the full uploaded document set.
Make sure the questions reflect all distinct documents provided, not just the first one.
If there are multiple documents, include comparison, synthesis, overlap, and contrast where useful.

Rules:
- Only return questions
- No explanations
- No numbering or bullets
- No JSON
- Each question on a new line

Document set excerpts:
{text}
"""

    try:
        response = llm.invoke(prompt)
        content = extract_text(response)
        generated = [_clean_question(line) for line in content.splitlines()]
        questions = []
        for question in generated:
            if question and question not in questions:
                questions.append(question)
    except Exception:
        questions = []

    for fallback_question in _fallback_questions(documents):
        if fallback_question not in questions:
            questions.append(fallback_question)

    return questions[:5]
