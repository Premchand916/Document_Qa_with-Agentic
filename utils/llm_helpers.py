def normalize_llm_output(content):
    if isinstance(content, list):
        content = " ".join(str(x) for x in content)
    return str(content).strip()