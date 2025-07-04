<!-- filepath: g:\ThePublicJiraDataset\docs\USAGE.md -->
# Usage

This project provides a command-line interface (CLI) for indexing Jira data and querying it using RAG.

## Prerequisites

- Ensure you have installed the necessary dependencies (see [Installation](INSTALL.md)).
- Configure your environment variables, especially `MONGO_URI` and `GEMINI_API_KEY` (refer to `src/config.py` for all required variables).

## Generating the Test Set

Before running benchmarks or if you need a specific subset of your Jira data for testing, you can generate a test set file.

The script `generate_real_testset.py` fetches data from your MongoDB instance and saves it to `real_data_rag_testset.jsonl`.

To run it:

```bash
python generate_real_testset.py
```

You can configure the number of records to fetch and skip within the script by modifying `NUM_RECORDS_TO_FETCH` and `SKIP_RECORDS` variables.

## Indexing Jira Data

To build or update the FAISS index with your Jira data from MongoDB, use the `index` command:

```bash
python main.py index
```

This command will:
- Connect to your MongoDB instance.
- Load the sentence transformer model.
- Load an existing FAISS index and metadata if found, or create a new one.
- Fetch Jira issues in batches (continuing from where it left off if an existing index is loaded).
- Generate embeddings for the fetched issues.
- Add the embeddings to the FAISS index.
- Save the updated index and metadata (`jira_index.faiss` and `jira_index_metadata.json`).
- Keep track of progress in `jira_index_progress.json`.

## Querying Similar Issues

Once your data is indexed, you can find similar historical issues for a new ticket summary using the `query` command:

```bash
python main.py query --text "Your new ticket summary here"
```

This will output a list of the most similar issues found in the index, along with their similarity scores.

### RAG-Augmented Querying

To also generate a prompt for a Large Language Model (LLM) like Google Gemini, which includes the context of these similar issues, add the `--rag` flag:

```bash
python main.py query --text "Your new ticket summary here" --rag
```

This will:
1.  Perform the similarity search.
2.  Print the top similar issues.
3.  Generate a RAG prompt incorporating the new ticket summary and the retrieved similar issues.
4.  Print the generated prompt.
5.  If `GEMINI_API_KEY` is configured, it will send the prompt to the Gemini API and print the LLM's response.

### Debug Mode

You can enable debug logging for more verbose output by adding the `--debug` flag before the command:

```bash
python main.py --debug index
python main.py --debug query --text "Your new ticket summary here"
```

## Using Local LLMs (Hugging Face Transformers)

In addition to Google Gemini, you can use a local Large Language Model (LLM) via Hugging Face Transformers for RAG-augmented querying and benchmarking. This is useful for running open-source models like Qwen3-30B-A3B-FP8 or Llama 2/3 on your own hardware.

### Querying with a Local LLM

To use a local LLM instead of Gemini, add the `--llm-backend local` flag and specify the model name with `--local-model-name`:

```powershell
python main.py query --text "Your new ticket summary here" --rag --llm-backend local --local-model-name Qwen/Qwen3-30B-A3B-FP8
```

- The model will be loaded from Hugging Face Hub or your local cache.
- The prompt will be formatted using the model's chat template if available (for best results with chat models).

### Environment Variables for Benchmarking

For the benchmark script (`run_rag_benchmark.py`), you can set the backend and model via environment variables:

```powershell
$env:LLM_BACKEND = "local"
$env:LOCAL_MODEL_NAME = "Qwen/Qwen3-30B-A3B-FP8"
python run_rag_benchmark.py
```

If you want to use Gemini, set `LLM_BACKEND` to `gemini` and ensure `GEMINI_API_KEY` is set.

See the [INSTALL](INSTALL.md) and [BENCHMARKING](BENCHMARKING.md) docs for more details.

## Troubleshooting

- If you encounter issues, check the logs for error messages.
- Ensure your MongoDB instance is running and accessible.
- Verify that your environment variables are correctly set.