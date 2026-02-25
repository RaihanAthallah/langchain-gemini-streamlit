# Dexa Medica FAQ Agent

Sistem RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan FAQ berdasarkan dokumen PDF Dexa Medica. Dibangun dengan LangGraph, pgvector, dan Gemini API.

---

## System Design

### Arsitektur

```
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: USER INTERFACE (Streamlit)                                       │
│  • Chat input  • Session history  • Ingest button  • Source expander       │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: FAQ AGENT (LangGraph)                                            │
│                                                                            │
│     ┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐   │
│     │  retrieve   │────▶│ route check  │────▶│ answer / rewrite /     │   │
│     │  chunks     │     │ (score≥0.68) │     │ no_answer               │   │
│     └──────┬──────┘     └──────┬───────┘     └───────────┬─────────────┘   │
│            │                  │                          │                 │
│            │                  │  score rendah            │  END            │
│            │                  └─────────────────────────┘                  │
│            │                            │                                  │
│            │                            ▼                                  │
│            │                  ┌─────────────────┐                          │
│            └─────────────────▶│ rewrite query   │ (max 1 retry → retrieve │
│                               │ (LLM)          │                           │
│                               └────────────────┘                           │
│                                                                            │
│  Memory: MemorySaver (context per thread_id)                               │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: SEMANTIC SEARCH                                                  │
│  • Embed query (Gemini)  • Cosine similarity (pgvector)  • Top-K chunks    │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: VECTOR STORE (PostgreSQL + pgvector)                             │
│  • faq_chunks: id, source_file, chunk_index, content, embedding(768)       │
└────────────────────────────────────────────────────────────────────────────┘
```

### Alur Kerja

1. **Ingestion (offline)**
   - PDF di `base-knowledge/` diekstrak dengan `pypdf`
   - Teks di-chunk dengan `RecursiveCharacterTextSplitter`
   - Setiap chunk di-embed dengan Gemini `text-embedding-004`
   - Data disimpan di tabel `faq_chunks` (pgvector)

2. **Query (online)**
   - User mengirim pertanyaan
   - **Retrieve**: Query di-embed → similarity search → top-k chunks
   - **Route**: Jika score tertinggi ≥ 0.68 → generate answer; jika tidak → rewrite query
   - **Rewrite**: LLM mengubah query untuk meningkatkan retrieval (max 1 retry)
   - **Answer**: LLM menjawab berdasarkan context chunks
   - **No Answer**: Jika tidak ada chunk relevan setelah retry → fallback message

3. **Memory**
   - `MemorySaver` menyimpan context per `thread_id`
   - Conversation history di-inject ke prompt rewrite dan generate

---

## Tools & Teknologi

| Kategori | Tool | Fungsi |
|----------|------|--------|
| **LLM** | Google Gemini API | Chat (`gemini-2.5-flash`), Embeddings (`gemini-embedding-001`) |
| **Vector DB** | PostgreSQL + pgvector | Penyimpanan embedding, similarity search (cosine) |
| **Orchestration** | LangGraph | State graph, retry flow, memory |
| **Document** | pypdf | Ekstraksi teks PDF |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | 900 chars, overlap 150 |
| **UI** | Streamlit | Chat interface, session state |
| **Tracing** | LangSmith | LangSmith tracing (opsional) |

---

## Struktur Project

```
ai-agents-part-1/
├── app/
│   ├── config.py          # Settings (env vars)
│   ├── db.py              # PostgreSQL connection
│   ├── document_processing.py  # PDF extract + chunk
│   ├── ingest.py          # Pipeline: PDF → embed → pgvector
│   ├── llm_clients.py     # Gemini chat + embeddings
│   ├── semantic_search.py # Query → embed → similarity search
│   ├── streamlit_app.py   # Chat UI
│   ├── vector_store.py    # pgvector schema, upsert, search
│   └── evaluate.py        # ROUGE evaluation (opsional)
├── agents/
│   ├── faq_agent.py       # LangGraph FAQ agent
│   ├── faq_tool.py        # LangChain tool wrapper
│   └── lab8_integration.py
├── base-knowledge/        # PDF FAQ documents
├── infra/
│   └── postgres-init/     # Init script (pgvector extension)
├── lab8_supervisor_app.py # Supervisor (DBQNA + RAG + FAQ)
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Cara Menjalankan

### Opsi 1: Docker (recommended)

```bash
# 1. Build dan jalankan semua service
docker compose up -d

# 2. Buka browser
# - App: http://localhost:8501
# - Adminer: http://localhost:8080 (opsional)

# 3. Di UI, klik "Ingest PDFs from base-knowledge" untuk load PDF ke vector store

# Stop
docker compose down
```

### Opsi 2: Lokal (tanpa Docker)

```bash
# 1. Jalankan PostgreSQL (Docker)
docker compose up -d postgres

# 2. Setup environment
cp .env.example .env
# Edit .env: isi GOOGLE_API_KEY dan variabel lain

# 3. Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# 4. Ingest PDF
python -m app.ingest

# 5. Jalankan Streamlit
streamlit run app/streamlit_app.py
```

### Opsi 3: Lab 8 Supervisor

```bash
streamlit run lab8_supervisor_app.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | - | API key Gemini |
| `GEMINI_EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Model embedding |
| `GEMINI_CHAT_MODEL` | No | `gemini-2.5-flash` | Model chat |
| `EMBEDDING_DIMENSION` | No | `768` | Dimensi embedding |
| `POSTGRES_HOST` | No | `localhost` | Host PostgreSQL (gunakan `postgres` di Docker) |
| `POSTGRES_PORT` | No | `5432` | Port PostgreSQL |
| `POSTGRES_DB` | No | `faq_agent` | Nama database |
| `POSTGRES_USER` | No | `faq_user` | User database |
| `POSTGRES_PASSWORD` | No | `faq_password` | Password database |
| `LANGSMITH_API_KEY` | No | - | LangSmith tracing (opsional) |

---

## Fitur

- **RAG**: Semantic search + LLM untuk jawaban berbasis dokumen
- **Query rewrite**: Retry dengan query yang di-rewrite jika retrieval kurang relevan
- **Memory**: Context percakapan per thread
- **Retry**: Auto-retry saat koneksi gagal (3x)
- **Logging**: Log untuk request, retrieval, dan answer
- **Source**: Tampilkan sumber chunk (file, chunk index, score)
