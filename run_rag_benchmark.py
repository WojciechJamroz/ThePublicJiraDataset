import json
from src.prompt import generate_rag_prompt
from typing import List, Dict, Any, Tuple, Optional, Counter as TypingCounter # Renamed to avoid conflict
import re
import os
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from sentence_transformers import SentenceTransformer # For embedding_model type
import faiss # For index type
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
from src.llm import LLMWrapper  # Add this import

logging.basicConfig(level=logging.INFO)

INPUT_FILE: str = "real_data_rag_testset.jsonl"
OUTPUT_JSONL_FILE: str = "rag_benchmark_results.jsonl" # Renamed for clarity
OUTPUT_CSV_FILE: str = "rag_benchmark_results.csv" # Added CSV output file
OUTPUT_CONFUSION_MATRIX_FILE: str = "confusion_matrix.png" # Added for plot
PROMPTS_FILE: str = "rag_benchmark_prompts.jsonl"

async def query_command(text: str, embedding_model: SentenceTransformer, index: faiss.Index, metadata: List[Dict[str, Any]], llm: LLMWrapper) -> Optional[Tuple[str, str]]:
    # FAISS search (CPU-bound, kept synchronous for now, or could be run in executor)
    results: List[Dict[str, Any]] = search_faiss(text, embedding_model, index, metadata)
    print(f"Top similar issues for query '{text}':")
    for r in results:
        print(f"{r['issue_key']} (score={r['similarity']:.4f}): {r['text']} ({r['issuetype']})")
    
    prompt = generate_rag_prompt(text, results)
    print("\n=== RAG Prompt ===\n")
    print(prompt)

    # Send prompt to LLM (Gemini or local)
    try:
        # Use sync call for both Gemini and local for now
        resp_text = llm.generate(prompt)
        print("\n=== LLM Response ===\n")
        print(resp_text)
        return resp_text, prompt
    except Exception as e:
        logging.error(f"Error calling LLM for query '{text}': {e}")
        print(f"[Error] Could not get response from LLM for query '{text}'.\n{e}")
        return None

def extract_issuetype_from_llm_output(llm_output: Optional[str]) -> str:
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


def plot_confusion_matrix(confusion_matrix_data: TypingCounter[Tuple[str, str]], all_types: List[str], output_filename: str) -> None:
    """Generates and saves a confusion matrix plot."""
    matrix_size: int = len(all_types)
    matrix: np.ndarray = np.zeros((matrix_size, matrix_size), dtype=int)

    type_to_index: Dict[str, int] = {t: i for i, t in enumerate(all_types)}

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
    correct: int = 0
    total: int = 0
    benchmark_results: List[Dict[str, Any]] = [] # Renamed from results to avoid conflict
    generated_prompts: List[str] = [] # Renamed from prompts
    
    # For detailed summary
    expected_counts: TypingCounter[str] = Counter()
    correct_by_type: TypingCounter[str] = Counter()
    predicted_by_type: TypingCounter[str] = Counter()
    confusion_matrix: TypingCounter[Tuple[str, str]] = Counter()


    print(f"Running RAG benchmark on {INPUT_FILE}...\n")

    # Load models and index once
    index_data: Tuple[Optional[faiss.Index], List[Dict[str, Any]]] = load_index_and_metadata(use_gpu=(DEVICE=='cuda'))
    index: Optional[faiss.Index] = index_data[0]
    metadata: List[Dict[str, Any]] = index_data[1]

    if index is None or not metadata:
        logging.error("Index not initialized. Please run 'index' first.")
        return

    embedding_model_data: Tuple[SentenceTransformer, Any] = load_embedding_model(device=DEVICE) # Assuming second part is tokenizer or similar, type Any for now
    embedding_model: SentenceTransformer = embedding_model_data[0]
    
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set in environment")
        print("[Error] GEMINI_API_KEY not set. Cannot run benchmark.")
        return
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model: GenerativeModel = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # LLM backend selection
    llm_backend = os.getenv("LLM_BACKEND", "gemini")
    local_model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-30B-A3B-FP8")
    llm = LLMWrapper(backend=llm_backend, local_model_name=local_model_name)

    tasks: List[asyncio.Task] = []
    source_items: List[Dict[str, Any]] = [] # To store original items for later processing

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item: Dict[str, Any] = json.loads(line)
            source_items.append(item)
            summary: str = item["summary"]
            # Create a task for each item
            if embedding_model and index and metadata and llm: # Ensure models are loaded
                tasks.append(asyncio.create_task(query_command(summary, embedding_model, index, metadata, llm)))
            else:
                logging.error("A model or index is None, cannot create task.") # Should not happen if checks above pass

    if not tasks:
        print("No items to benchmark.")
        return
        
    # Run all tasks concurrently
    llm_query_results: List[Optional[Tuple[str, str]]] = await asyncio.gather(*tasks)

    for idx, llm_result_tuple in enumerate(llm_query_results):
        item = source_items[idx]
        summary = item["summary"]
        description: str = item.get("description", "")
        expected: str = item["expected_issuetype"]
        
        current_test_idx: int = idx + 1 # For 1-based indexing in logs

        print(f"[Test {current_test_idx}] Summary: {summary}")
        print(f"[Test {current_test_idx}] Description: {description}")
        print(f"[Test {current_test_idx}] Expected: {expected}")

        expected_counts[expected] += 1 # Count expected types
        
        llm_output_text: str
        prompt_text: Optional[str] = None # Initialize prompt_text

        if llm_result_tuple is None:
            print(f"[Test {current_test_idx}] LLM query failed. Skipping.")
            predicted_issuetype: str = "ERROR_QUERY_FAILED"
            llm_output_text = "ERROR: Query command failed or returned None"
            is_correct: bool = False
        else:
            llm_output_text, prompt_text = llm_result_tuple
            if prompt_text: # Ensure prompt_text is not None before appending
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
            
    accuracy: float = correct / total if total else 0.0
    
    # Save to JSONL file
    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as out: # Updated filename
        for r_item in benchmark_results: # Renamed loop variable
            out.write(json.dumps(r_item, ensure_ascii=False) + "\\n")

    # Save to CSV file
    if benchmark_results:
        keys: List[str] = list(benchmark_results[0].keys())
        with open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as out_csv: # Added CSV writer
            writer: csv.DictWriter = csv.DictWriter(out_csv, fieldnames=keys)
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
    sorted_expected_types: List[str] = sorted(expected_counts.keys())
    for issue_type in sorted_expected_types:
        count: int = expected_counts[issue_type]
        correct_count: int = correct_by_type[issue_type]
        type_accuracy: float = correct_count / count if count else 0.0
        print(f"  {issue_type}: {type_accuracy:.2%} ({correct_count}/{count})")

    print("\n--- Confusion Matrix (Expected -> Predicted) ---")
    # Collect all unique types for header/row labels
    all_types: List[str] = sorted(list(set(expected_counts.keys()) | set(predicted_by_type.keys())))
    
    # Print header
    header: str = "Expected \\ Predicted | " + " | ".join(f"{t[:10]:<10}" for t in all_types) # Truncate long names
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
