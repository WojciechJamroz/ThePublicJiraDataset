import argparse
import logging
import sys  # add for exit
import numpy as np  # add numpy import
from typing import Any
from .utils.logging_config import configure_logging
from .config import DEVICE, GEMINI_API_KEY, EMBEDDING_MODEL, GEMINI_MODEL_NAME
from .db import connect_to_db, get_jira_issues
from .embed import load_embedding_model, generate_embeddings
from .indexer import init_faiss_index, load_index_and_metadata, save_index_and_metadata, update_progress
from .search import search_faiss
from .prompt import generate_rag_prompt
import google.generativeai as genai  # add Gemini API client

def index_command(args: Any) -> None:
    db = connect_to_db()
    if db is None:
        return

    # Load model and index
    model, dim = load_embedding_model(device=DEVICE)
    index, metadata = load_index_and_metadata(use_gpu=(DEVICE=='cuda'))
    if index is None:
        index = init_faiss_index(dim, use_gpu=(DEVICE=='cuda'))
        metadata = []

    # Determine starting offset
    start = len(metadata)
    total_indexed = start

    for batch in get_jira_issues(db, initial_skip=start):
        if not batch:
            break
        embeds = generate_embeddings(batch, model)
        if not embeds:
            continue

        vecs = [e['embedding'] for e in embeds]
        meta_batch = [{'issue_key': e['issue_key'], 'text': e['text'], 'issuetype': e['issuetype']} for e in embeds]

        index.add(np.array(vecs, dtype='float32'))
        metadata.extend(meta_batch)
        total_indexed = index.ntotal

        # persist
        save_index_and_metadata(index, metadata)
        update_progress(total_indexed)
        logging.info("Indexed %d items so far.", total_indexed)

    logging.info("Indexing complete. Total items: %d", total_indexed)


def query_command(args: Any) -> None:
    # load index/metadata
    index, metadata = load_index_and_metadata(use_gpu=(DEVICE=='cuda'))
    if index is None or not metadata:
        logging.error("Index not initialized. Please run 'index' first.")
        return

    # load model
    model, _ = load_embedding_model(device=DEVICE)
    results = search_faiss(args.text, model, index, metadata)
    print("Top similar issues:")
    for r in results:
        print(f"{r['issue_key']} (score={r['similarity']:.4f}): {r['text']} ({r['issuetype']})")

    if args.rag:
        prompt = generate_rag_prompt(args.text, results)
        print("\n=== RAG Prompt ===\n")
        print(prompt)
        # Send prompt to Gemini API
        try:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment")
            genai.configure(api_key=GEMINI_API_KEY)
            gem_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            resp = gem_model.generate_content(prompt)
            print("\n=== Gemini API Response ===\n")
            print(resp.text)
        except Exception as e:
            logging.error("Error calling Gemini API: %s", e)
            print("[Error] Could not get response from Gemini API.\nCheck your GEMINI_API_KEY and network connection.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jira RAG indexing and querying tool")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args_pre, _ = parser.parse_known_args()
    # Configure logging based on debug flag
    configure_logging(logging.DEBUG if args_pre.debug else logging.INFO)
    # Now parse full args including subcommands
    parser = argparse.ArgumentParser(description="Jira RAG indexing and querying tool")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    sub = parser.add_subparsers(dest='cmd')

    # subcommands
    p_index = sub.add_parser('index', help='Index Jira issues into FAISS')
    p_index.set_defaults(func=index_command)

    p_query = sub.add_parser('query', help='Query similar issues')
    p_query.add_argument('--text', required=True, help='New ticket summary to query')
    p_query.add_argument('--rag', action='store_true', help='Also generate a RAG prompt')
    p_query.set_defaults(func=query_command)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)
    # Dispatch
    args.func(args)

if __name__ == '__main__':
    main()
