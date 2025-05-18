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

To index your Jira data, use the `index_data.py` script. This will read data from your MongoDB instance and create the necessary indexes for efficient querying.

Run the script as follows:

```bash
python index_data.py
```

## Querying Jira Data

Once your data is indexed, you can query it using the `query_data.py` script. This script allows you to perform various queries on your indexed Jira data.

Example usage:

```bash
python query_data.py --query "your_query_here"
```

Refer to the script's help message for more query options.

## Troubleshooting

- If you encounter issues, check the logs for error messages.
- Ensure your MongoDB instance is running and accessible.
- Verify that your environment variables are correctly set.