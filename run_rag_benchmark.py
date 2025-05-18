import json
from src.prompt import generate_rag_prompt
from typing import List, Dict, Any
import re
import os
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
import google.generativeai as genai
from src.embed import load_embedding_model
from src.indexer import load_index_and_metadata
from src.search import search_faiss
from src.config import DEVICE
import logging
import asyncio # Added asyncio
import csv # Added csv module
from collections import Counter # Added for summary
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO)

INPUT_FILE = "real_data_rag_testset.jsonl"
OUTPUT_JSONL_FILE = "rag_benchmark_results.jsonl" # Renamed for clarity
OUTPUT_CSV_FILE = "rag_benchmark_results.csv" # Added CSV output file
OUTPUT_CONFUSION_MATRIX_FILE = "confusion_matrix.png" # Added for plot
PROMPTS_FILE = "rag_benchmark_prompts.jsonl"

async def query_command(text: str, embedding_model, index, metadata, gemini_model) -> tuple[str, str] | None:
    # FAISS search (CPU-bound, kept synchronous for now, or could be run in executor)
    results = search_faiss(text, embedding_model, index, metadata)
    print(f"Top similar issues for query \\'{text}\\':")
    for r in results:
        print(f"{r['issue_key']} (score={r['similarity']:.4f}): {r['text']} ({r['issuetype']})")
    
    prompt = generate_rag_prompt(text, results)
    print("\\n=== RAG Prompt ===\\n")
    print(prompt)

    # Send prompt to Gemini API asynchronously
    try:
        resp = await gemini_model.generate_content_async(prompt) # Changed to async call
        print("\\n=== Gemini API Response ===\\n")
        print(resp.text)
        return resp.text, prompt
    except Exception as e:
        logging.error(f"Error calling Gemini API for query \\'{text}\\': {e}")
        print(f"[Error] Could not get response from Gemini API for query \\'{text}\\'.\\nCheck your GEMINI_API_KEY and network connection.")
        return None

def extract_issuetype_from_llm_output(llm_output: str) -> str:
    if not llm_output or not llm_output.strip():
        return "Unknown"
    for line in llm_output.splitlines():
        if "issuetype:" in line.lower():
            parts = line.split(":", 1)
            if len(parts) > 1:
                value = parts[1].strip().rstrip(" .:-")
                if value and value.lower() != "issuetype":
                    return value
    return "Unknown"


def plot_confusion_matrix(confusion_matrix_data, all_types, output_filename):
    """Generates and saves a confusion matrix plot."""
    matrix_size = len(all_types)
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    type_to_index = {t: i for i, t in enumerate(all_types)}

    for (expected, predicted), count in confusion_matrix_data.items():
        if expected in type_to_index and predicted in type_to_index:
            matrix[type_to_index[expected]][type_to_index[predicted]] = count

    plt.figure(figsize=(max(10, matrix_size), max(8, matrix_size))) # Adjust size dynamically
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=all_types, yticklabels=all_types,
                annot_kws={"size": 8 if matrix_size > 10 else 10}) # Adjust font size
    plt.xlabel("Predicted Issue Type")
    plt.ylabel("Expected Issue Type")
    plt.title("Confusion Matrix for Issue Type Prediction")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(output_filename)
        print(f"Confusion matrix plot saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving confusion matrix plot: {e}")

async def run_benchmark():
    correct = 0
    total = 0
    benchmark_results = [] # Renamed from results to avoid conflict
    generated_prompts = [] # Renamed from prompts
    
    # For detailed summary
    expected_counts = Counter()
    correct_by_type = Counter()
    predicted_by_type = Counter()
    confusion_matrix = Counter()


    print(f"Running RAG benchmark on {INPUT_FILE}...\n")

    # Load models and index once
    index, metadata = load_index_and_metadata(use_gpu=(DEVICE=='cuda'))
    if index is None or not metadata:
        logging.error("Index not initialized. Please run 'index' first.")
        return

    embedding_model, _ = load_embedding_model(device=DEVICE)
    
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set in environment")
        print("[Error] GEMINI_API_KEY not set. Cannot run benchmark.")
        return
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    tasks = []
    source_items = [] # To store original items for later processing

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            source_items.append(item)
            summary = item["summary"]
            # Create a task for each item
            tasks.append(query_command(summary, embedding_model, index, metadata, gemini_model))

    if not tasks:
        print("No items to benchmark.")
        return
        
    # Run all tasks concurrently
    llm_query_results = await asyncio.gather(*tasks)

    for idx, llm_result_tuple in enumerate(llm_query_results):
        item = source_items[idx]
        summary = item["summary"]
        description = item.get("description", "")
        expected = item["expected_issuetype"]
        
        current_test_idx = idx + 1 # For 1-based indexing in logs

        print(f"[Test {current_test_idx}] Summary: {summary}")
        print(f"[Test {current_test_idx}] Description: {description}")
        print(f"[Test {current_test_idx}] Expected: {expected}")

        expected_counts[expected] += 1 # Count expected types

        if llm_result_tuple is None:
            print(f"[Test {current_test_idx}] LLM query failed. Skipping.")
            predicted_issuetype = "ERROR_QUERY_FAILED"
            llm_output_text = "ERROR: Query command failed or returned None"
            is_correct = False
        else:
            llm_output_text, prompt_text = llm_result_tuple
            generated_prompts.append(prompt_text)
            print(f"[Test {current_test_idx}] LLM Output: {llm_output_text}")
            predicted_issuetype = extract_issuetype_from_llm_output(llm_output_text)
            print(f"[Test {current_test_idx}] Predicted: {predicted_issuetype}")
            is_correct = (predicted_issuetype.lower() == expected.lower())
        
        predicted_by_type[predicted_issuetype] +=1 # Count predicted types
        confusion_matrix[(expected, predicted_issuetype)] += 1 # For confusion matrix

        print(f"[Test {current_test_idx}] {'CORRECT' if is_correct else 'INCORRECT'}\n{'-'*40}")
        
        benchmark_results.append({
            "summary": summary,
            "description": description,
            "expected": expected,
            "predicted": predicted_issuetype,
            "is_correct": is_correct,
            "llm_output": llm_output_text,
        })
        if is_correct:
            correct += 1
            correct_by_type[expected] += 1 # Count correct by type
        total += 1
            
    accuracy = correct / total if total else 0.0
    
    # Save to JSONL file
    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as out: # Updated filename
        for r_item in benchmark_results: # Renamed loop variable
            out.write(json.dumps(r_item, ensure_ascii=False) + "\\n")

    # Save to CSV file
    if benchmark_results:
        keys = benchmark_results[0].keys()
        with open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as out_csv: # Added CSV writer
            writer = csv.DictWriter(out_csv, fieldnames=keys)
            writer.writeheader()
            writer.writerows(benchmark_results)
        print(f"Results also saved to {OUTPUT_CSV_FILE}")

    with open(PROMPTS_FILE, "w", encoding="utf-8") as pf:
        for p_item in generated_prompts: # Renamed loop variable
            pf.write(p_item + "\\n") # Assuming prompts are full strings with newlines if needed
            
    print(f"\n--- Benchmark Summary ---")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to {OUTPUT_JSONL_FILE}") # Updated filename
    print(f"Prompts saved to {PROMPTS_FILE}")

    print("\n--- Accuracy by Issue Type ---")
    sorted_expected_types = sorted(expected_counts.keys())
    for issue_type in sorted_expected_types:
        count = expected_counts[issue_type]
        correct_count = correct_by_type[issue_type]
        type_accuracy = correct_count / count if count else 0.0
        print(f"  {issue_type}: {type_accuracy:.2%} ({correct_count}/{count})")

    print("\n--- Confusion Matrix (Expected -> Predicted) ---")
    # Collect all unique types for header/row labels
    all_types = sorted(list(set(expected_counts.keys()) | set(predicted_by_type.keys())))
    
    # Print header
    header = "Expected \\ Predicted | " + " | ".join(f"{t[:10]:<10}" for t in all_types) # Truncate long names
    print(header)
    print("-" * len(header))

    for expected_type in sorted_expected_types:
        row_str = f"{expected_type[:18]:<18} | " # Truncate long names
        for predicted_type in all_types:
            count = confusion_matrix[(expected_type, predicted_type)]
            row_str += f"{str(count):<10} | "
        print(row_str)
    print("-" * len(header))
    print("\nBenchmark complete.")

    # Generate and save confusion matrix plot
    if confusion_matrix and all_types:
        plot_confusion_matrix(confusion_matrix, all_types, OUTPUT_CONFUSION_MATRIX_FILE)


if __name__ == "__main__":
    asyncio.run(run_benchmark()) # Changed to asyncio.run
