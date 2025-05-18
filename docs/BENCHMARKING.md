# Benchmarking

This project includes a benchmarking suite to evaluate the performance of the RAG pipeline in classifying Jira issue types.

## How it Works

The benchmark script (`run_rag_benchmark.py`) processes a test set of Jira issues (defined in `real_data_rag_testset.jsonl`). For each issue, it:

1.  Uses the RAG pipeline to predict the `issuetype`.
2.  Compares the predicted `issuetype` with the `expected_issuetype` from the test set.
3.  Records the results, including the summary, description, expected type, predicted type, whether the prediction was correct, and the raw LLM output.

## Running the Benchmark

To run the benchmark, execute the following command from the project root:

```bash
python run_rag_benchmark.py
```

Ensure your environment is configured correctly, especially the `GEMINI_API_KEY`.

## Outputs

The benchmark script generates several output files in the project root:

*   **`rag_benchmark_results.jsonl`**: Detailed results for each test case in JSON Lines format. Each line is a JSON object containing:
    *   `summary`: The issue summary.
    *   `description`: The issue description.
    *   `expected`: The expected issue type.
    *   `predicted`: The predicted issue type.
    *   `is_correct`: Boolean indicating if the prediction was correct.
    *   `llm_output`: The full response from the language model.
*   **`rag_benchmark_results.csv`**: The same detailed results in CSV format for easier analysis in spreadsheet software.
*   **`rag_benchmark_prompts.jsonl`**: The actual prompts generated and sent to the LLM for each test case.
*   **`confusion_matrix.png`**: A PNG image visualizing the confusion matrix, showing the distribution of correct and incorrect predictions across different issue types.

## Interpreting Results

After running the benchmark, you can analyze the output files:

*   **Overall Accuracy**: The script prints the overall accuracy to the console.
*   **Accuracy by Issue Type**: The console output also includes a breakdown of accuracy for each specific issue type.
*   **Confusion Matrix (Text)**: A text-based confusion matrix is printed to the console.
*   **Confusion Matrix (Image)**: Open `confusion_matrix.png` to see a visual representation. This helps identify which issue types are often confused with others.
*   **Detailed CSV/JSONL**: For a deep dive into individual errors or patterns, inspect `rag_benchmark_results.csv` or `rag_benchmark_results.jsonl`. This can help in understanding why certain predictions were incorrect by looking at the `llm_output` and the input `summary`/`description`.

By regularly running the benchmark and analyzing its outputs, you can track the impact of changes to the embedding models, prompting strategies, or the LLM itself.
