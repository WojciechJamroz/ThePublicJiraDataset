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

### 8. Test Data Generation (`generate_real_testset.py`)
- Fetches a subset of data from MongoDB for creating test/benchmark datasets.
- Allows configuration of the number of records to fetch and skip.
- Outputs data in JSONL format (`real_data_rag_testset.jsonl`).

### 9. Benchmarking (`run_rag_benchmark.py`)
- Runs the RAG pipeline on a predefined test set (e.g., `real_data_rag_testset.jsonl`).
- Compares predicted issue types against expected types.
- Generates various output files for analysis:
  - `rag_benchmark_results.jsonl`: Detailed results per test case.
  - `rag_benchmark_results.csv`: CSV version of the results.
  - `rag_benchmark_prompts.jsonl`: Prompts used for LLM queries.
  - `confusion_matrix.png`: Visual confusion matrix.
- Prints summary statistics (overall accuracy, per-type accuracy, text-based confusion matrix).

### 10. LLM Abstraction (`src/llm.py`)
- Provides a unified interface for calling either Google Gemini or a local Hugging Face LLM.
- Uses chat template formatting for chat-based models (e.g., Qwen, Llama, Mistral) when available.
- CLI and benchmarking scripts can select backend via CLI flags or environment variables.

## Workflows

### 1. Data Preparation (Optional)
- Run `python generate_real_testset.py` to create `real_data_rag_testset.jsonl` from MongoDB for benchmarking or testing.

### 2. Indexing
- Execute `python main.py index`.
- **Process**: Connect to MongoDB → Load/Initialize FAISS index & metadata → Fetch Jira issues in batches → Generate embeddings for issues → Add embeddings to FAISS index → Save updated index, metadata, and progress.

### 3. Querying & RAG
- Execute `python main.py query --text "<new_issue_summary>" [--rag]`.
- **Process**: Load FAISS index & metadata → Load embedding model → Embed query text → Retrieve top-k similar issues from FAISS.
- **If `--rag` is used**: Generate RAG prompt using query and retrieved issues → Send prompt to Gemini API (if configured) → Print LLM response.

### 4. Benchmarking
- Ensure `real_data_rag_testset.jsonl` exists (see Data Preparation).
- Run `python run_rag_benchmark.py`.
- **Process**: For each item in test set → Perform RAG-augmented querying → Extract predicted issuetype → Compare with expected issuetype → Aggregate results → Save detailed results (JSONL, CSV), prompts, and confusion matrix image.

## Dependencies

- Data: MongoDB
- Embeddings: `sentence-transformers` (E5 models)
- Indexing: FAISS
- LLM calls: `google-generativeai` for Gemini, or `transformers` for local Hugging Face models (Qwen, Llama, etc.)

