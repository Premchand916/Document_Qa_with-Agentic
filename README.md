# Document_Qa_with-Agentic

Multi-format document intelligence workspace built with Streamlit, LangGraph, retrieval, and prompt-chained answering.

## Supported file types

- PDF
- PPTX
- XLSX
- XLSM
- CSV
- TSV
- TXT
- MD
- JSON

## What the app now supports

- Mixed-file upload in one workspace
- Cross-document suggested questions
- Prompt-chained answer planning
- Audience-aware and answer-mode-aware responses
- Source cards with 20-word previews
- Click-to-expand source evidence
- 72 curated prompt examples across major business use cases

## Prompt chain

The answer flow is:

1. Intent routing
2. Response planning
3. Retrieval and reranking
4. Final answer generation

This lets the app adapt to different user needs such as:

- executive brief
- analyst deep dive
- comparison matrix
- action plan
- risk review
- data highlights
- research synthesis

## Example user needs covered

- leadership summaries
- cross-file comparison
- deck review
- spreadsheet analysis
- customer research synthesis
- finance review
- operations review
- risk and compliance review
- product discovery
- strategy planning
- knowledge extraction

## Local run

```bash
streamlit run app/main.py
```
