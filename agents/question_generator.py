import re
from collections import defaultdict

from agents.orchestrator_agent import create_orchestrator


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


def _parse_question_response(response):
    if hasattr(response, "content"):
        raw = response.content
    else:
        raw = response

    if isinstance(raw, list):
        parts = []
        for item in raw:
            if hasattr(item, "text") and item.text:
                parts.append(item.text)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        raw = "\n".join(parts)

    if isinstance(raw, dict):
        raw = raw.get("text", "")

    clean_questions = []
    for question in str(raw).split("\n"):
        cleaned = _clean_question(question.strip("- ").strip())
        if len(cleaned.strip()) > 10 and cleaned not in clean_questions:
            clean_questions.append(cleaned)

    return clean_questions[:5]


def generate_questions(llm_or_documents, context=None):
    if context is not None:
        llm = llm_or_documents
        prompt = f"""
Based on the following document, generate 5 meaningful questions.

Document:
{context}

Return ONLY a clean list of questions.
"""
        return _parse_question_response(llm.invoke(prompt))

    documents = llm_or_documents
    if not documents:
        return []

    text = _build_balanced_document_excerpt(documents)
    file_types = []
    for doc in documents:
        file_type = doc.metadata.get("file_type")
        if file_type and file_type.upper() not in file_types:
            file_types.append(file_type.upper())

    llm = create_orchestrator()
    prompt = f"""
Generate 5 concise, high-value suggested questions based on the full uploaded document set.
Make sure the questions reflect all distinct documents provided, not just the first one.
If there are multiple documents, include comparison, synthesis, overlap, and contrast where useful.
Uploaded file types: {", ".join(file_types) if file_types else "unknown"}

Rules:
- Return only questions
- No numbering
- No JSON
- No explanations

Document set excerpts:
{text}
"""

    try:
        questions = _parse_question_response(llm.invoke(prompt))
    except Exception:
        questions = []

    for fallback_question in _fallback_questions(documents):
        if fallback_question not in questions:
            questions.append(fallback_question)

    return questions[:5]
