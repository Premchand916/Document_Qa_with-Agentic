# 🧠 Document Intelligence Agent (LangGraph + RAG + Agents)

An advanced **multi-format AI-powered document intelligence system** built using **LangGraph, RAG, and Agentic AI workflows**.

This system enables users to upload documents (PDF, PPT, Excel, CSV, JSON, etc.) and interact with them using natural language — powered by **LLMs, vector search, and intelligent agents**.

---

## 🔥 Key Features

- 📄 Multi-format document support (PDF, PPTX, CSV, Excel, JSON, TXT)
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 Agentic AI using LangGraph
- 🔗 Multi-step reasoning & orchestration
- 💡 Auto-generated suggested questions
- ⚡ Fast semantic search using FAISS
- 📊 Structured + unstructured document handling
- 🧰 Tool-ready architecture (DataFrame computation ready)
- 🎯 Clean answers with minimal hallucination
- 📌 Versioned evidence IDs and validated citations
- 🔎 Bounded local OCR for images and scanned PDFs

---

## 🏗️ Architecture Overview

### 🔁 System Flow
User Query
   ↓
Response Planner (Agent)
   ↓
Retriever (FAISS Vector Search)
   ↓
(Optional) Tool Calling Layer
   ↓
QA Agent (LLM)
   ↓
Final Answer + Source
🧩 High-Level Architecture
            ┌──────────────────────┐
            │ FastAPI Browser UI   │
            └─────────┬────────────┘
                      ↓
            ┌──────────────────────┐
            │  LangGraph Workflow  │
            └─────────┬────────────┘
                      ↓
    ┌─────────────── Agents ────────────────┐
    │ Response Plan → Route → Retrieve → QA │
    └───────────────────────────────────────┘
                      ↓
            ┌──────────────────────┐
            │  Vector DB (FAISS)   │
            └──────────────────────┘
                      ↓
            ┌──────────────────────┐
            │ Embeddings (MiniLM)  │
            └──────────────────────┘
🧩 Core Components
📥 1. Document Ingestion

Supports multiple formats:

PDF → pdfplumber

PPTX → python-pptx

Excel/CSV → pandas

JSON/TXT/MD → native parsing

✂️ 2. Semantic Chunking

Adaptive chunking strategy

Preserves:

Tables

Slides

Structured content

Maintains semantic continuity using overlap

🔍 3. Embeddings & Vector Store

Model: sentence-transformers/all-MiniLM-L6-v2

Vector DB: FAISS

Enables fast similarity search

🤖 4. Agentic Workflow (LangGraph)
Agent	Responsibility
Response Planner	Choose bounded answer strategy
Planner Agent	Select validated document/table route
Retriever Agent	Fetch relevant chunks
QA Agent	Generate grounded answer
(Allowlisted) Table Tool	Perform validated calculations
💡 5. Suggested Questions Engine

Automatically generates meaningful questions

Helps users explore documents faster

Click → auto-query execution

🖥️ User Interface (FastAPI + browser)
Features:

📂 Multi-file upload

💬 Chat interface

💡 Suggested questions (clickable)

📄 Clean answer display

📌 Source highlighting

⚙️ Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Required for local image/scanned-PDF OCR
# macOS: brew install tesseract
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
```

🔑 Environment Setup

Create a .env file:

```dotenv
GOOGLE_API_KEY=your_api_key
HF_LOCAL_FILES_ONLY=true
EMBEDDING_BATCH_SIZE=32
DOCQA_AUTH_MODE=local
DOCQA_TURN_LEASE_SECONDS=60
DOCQA_CONVERSATION_RETENTION_DAYS=90
DOCQA_EPISODE_RETENTION_DAYS=180
DOCQA_CHAT_DEADLINE_SECONDS=120
DOCQA_INGESTION_LEASE_SECONDS=120
DOCQA_INGESTION_RETENTION_DAYS=7
DOCQA_INGESTION_MAX_PENDING=10
# Tesseract packs must be installed locally; join multiple identifiers with +
DOCQA_OCR_LANGUAGES=eng
DOCQA_OCR_PREPROCESS=true
DOCQA_OCR_AUTO_ROTATE=true
# Optional and explicit: permits image bytes to leave the machine for cloud OCR
DOCQA_ALLOW_CLOUD_OCR=false
```
▶️ Run the Application

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Open `http://127.0.0.1:8000`. Local mode is explicitly limited to loopback access.
Shared access requires a configured authentication proxy; see
[Step 1 access configuration and limits](docs/STEP1_RESULTS.md) and
[Step 2 workspace configuration, storage, and legacy-index migration](docs/STEP2_RESULTS.md).
`app/main.py` is a legacy Streamlit entry point and does not use these FastAPI boundaries.

📊 Example Use Cases

📄 Financial report analysis

📊 Validated CSV/Excel calculations

📚 Research document summarization

🏢 Enterprise knowledge assistant

📑 Policy & compliance document QA

⚡ Performance Optimizations

Embedding model caching

Retrieval caching

Reduced top-k search

Optimized prompt size

Minimal context passing

⚠️ Current Limitations

No full DataFrame computation engine (planned)

Limited tool-calling capabilities

No hybrid retrieval (BM25 + vector) yet

🔮 Future Enhancements

📊 DataFrame Agent (Excel intelligence)

🔁 Hybrid retrieval (BM25 + vector search)

⚡ Streaming responses (ChatGPT-like typing)

📈 Observability (Langfuse / tracing)

☁️ Cloud deployment (AWS / GCP)

🧠 Memory-enabled agents

🧪 Tech Stack

Frontend: Streamlit

Backend: Python

LLM: Gemini / OpenAI-compatible

Embeddings: HuggingFace

Vector DB: FAISS

Frameworks: LangChain, LangGraph

Data Processing: Pandas

🧠 What This Project Demonstrates

End-to-end RAG architecture

Agent-based system design

Prompt engineering & reasoning workflows

Vector search optimization

Multi-format document intelligence

📌 Why This Project Matters

This is not just a chatbot — it is a production-style GenAI system that demonstrates:

Grounded responses (reducing hallucination)

Multi-agent orchestration

Intelligent document analysis

Scalable AI architecture patterns

🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

📬 Contact

LinkedIn: https://www.linkedin.com/in/premchand24/

GitHub: https://github.com/Premchand916

⭐ Support

If you found this useful, give this repo a ⭐ and connect with me!
