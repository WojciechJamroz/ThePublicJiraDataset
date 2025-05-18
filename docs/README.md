# Jira RAG Bootstrap

This repository provides a Retrieval-Augmented Generation (RAG) pipeline for classifying new Jira tickets by retrieving historically similar issues.

Features:
- **MongoDB ingestion:** Batch-fetch Jira issues from a local or remote MongoDB instance.
- **Embeddings generation:** Use `sentence-transformers` to encode issue text into vectors.
- **FAISS indexing:** Build and persist a FAISS index for fast similarity search (CPU or GPU).
- **CLI interface:** `index` and `query` subcommands with optional RAG prompt generation.
- **Configurable via `.env`:** All settings (DB, model, file paths, API keys) are environment-driven.

Directory structure:
```
├── config/               # Environment variable examples
├── src/                  # Python package source
│   ├── cli.py            # Entry-point CLI logic
│   ├── db.py             # MongoDB connection & data loader
│   ├── embed.py          # Embedding model and functions
│   ├── indexer.py        # FAISS index init/load/save
│   ├── search.py         # Search wrapper for FAISS
│   ├── prompt.py         # RAG prompt builder
│   └── utils/            # Shared utilities (logging config)
├── docs/                 # Documentation (this folder)
├── main.py               # Root launcher for CLI
└── requirements.txt      # Python dependencies
```

Please refer to **USAGE.md** for setup and commands.
