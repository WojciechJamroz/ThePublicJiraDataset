# Architecture Overview

This document describes the high-level design and module breakdown of the Jira RAG Bootstrap project.

## Components

### 1. Configuration (`src/config.py`)
- Loads environment variables via `python-dotenv`.
- Exposes constants for MongoDB, FAISS index files, embedding model, and Gemini settings.
- Detects compute device (`cpu` or `cuda`).

### 2. Data Access Layer (`src/db.py`)
- `connect_to_db()`: Establishes a MongoDB connection.
- `get_jira_issues()`: Streams batches of issues (summary, description, issuetype) for efficient indexing.

### 3. Embeddings (`src/embed.py`)
- `load_embedding_model()`: Loads `SentenceTransformer` on specified device.
- `generate_embeddings()`: Encodes issue text into float32 vectors, applies prefixing based on model type.

### 4. Indexing (`src/indexer.py`)
- `init_faiss_index()`: Initializes FAISS `IndexFlatIP` (CPU/GPU).
- `load_index_and_metadata()`: Loads persistent index, metadata, and progress for resuming.
- `save_index_and_metadata()`: Persists index and metadata JSON.
- `update_progress()`: Writes progress checkpoint.

### 5. Search (`src/search.py`)
- `search_faiss()`: Encodes a query, performs a top-k similarity search, formats results list.

### 6. Prompt Generation (`src/prompt.py`)
- `generate_rag_prompt()`: Builds a structured prompt including new ticket and historical context for LLM.

### 7. CLI (`src/cli.py` + `main.py`)
- Provides subcommands:
  - `index`: Build or resume FAISS index.
  - `query`: Search index and optionally generate/send RAG prompt to Google Gemini.
- Supports verbose (`--debug`) logging via `src/utils/logging_config.py`.

## Workflow

1. **Index**: Fetch issues → generate embeddings → add to FAISS → save index/metadata.
2. **Query**: Load index/metadata → embed query → retrieve top-k issues → build prompt → (optional) call Gemini API.

## Dependencies

- Data: MongoDB
- Embeddings: `sentence-transformers` (E5 models)
- Indexing: FAISS
- LLM calls: `google-generativeai` for Gemini

