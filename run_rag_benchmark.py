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

logging.basicConfig(level=logging.INFO)

INPUT_FILE = "real_data_rag_testset.jsonl"
OUTPUT_FILE = "rag_benchmark_results.jsonl"
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


async def run_benchmark():
    correct = 0
    total = 0
    benchmark_results = [] # Renamed from results to avoid conflict
    generated_prompts = [] # Renamed from prompts

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
        total += 1
            
    accuracy = correct / total if total else 0.0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r_item in benchmark_results: # Renamed loop variable
            out.write(json.dumps(r_item, ensure_ascii=False) + "\\n")
    with open(PROMPTS_FILE, "w", encoding="utf-8") as pf:
        for p_item in generated_prompts: # Renamed loop variable
            pf.write(p_item + "\\n") # Assuming prompts are full strings with newlines if needed
            
    print(f"\nBenchmark complete. Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Prompts saved to {PROMPTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run_benchmark()) # Changed to asyncio.run
