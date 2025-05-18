# Jira RAG Bootstrap

![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A Retrieval-Augmented Generation (RAG) pipeline to classify new Jira tickets by finding and leveraging historically similar issues.

This project provides:

- **MongoDB ingestion**: Batch-fetch Jira issues from MongoDB.
- **Embeddings**: Encode ticket text via `sentence-transformers`.
- **FAISS indexing**: Fast similarity search (CPU/GPU).
- **CLI**: Simple `index` and `query` commands.
- **RAG prompts**: Generate context-rich prompts for LLMs like Google Gemini.
- **Benchmarking**: Evaluate RAG pipeline performance with detailed results and visualizations (e.g., confusion matrix).

## Getting Started

See the detailed documentation in the `docs/` folder:

- [Installation](docs/INSTALL.md)
- [Usage](docs/USAGE.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset Information](docs/DATASET.md)
- [Contributing](docs/CONTRIBUTING.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
