import json
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import torch
import os # Added for path checking
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Determine device for embeddings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device for embeddings: {DEVICE}")
if DEVICE == 'cuda':
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device detected: {gpu_name}, CUDA runtime: {torch.version.cuda}")
    except Exception as e:
        logger.error(f"Failed to query CUDA device: {e}")
else:
    logger.warning("CUDA not available or PyTorch is CPU-only. Install the CUDA-enabled torch package to leverage your RTX4080 (e.g., pip install torch --index-url https://download.pytorch.org/whl/cu118)")

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI if different
DB_NAME = "JiraRepos"  # As per the dataset's mongodump
COLLECTION_NAME = "Apache"  # Use your actual Jira collection (e.g., 'Apache' or 'JiraEcosystem')
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large-instruct'  # Example embedding model
TASK_DESCRIPTION = "Given a new Jira ticket summary, retrieve relevant historical tickets for triage classification"

FAISS_INDEX_FILE = "jira_index.faiss"
INDEX_METADATA_FILE = "jira_index_metadata.json"
INDEX_PROGRESS_FILE = "jira_index_progress.json" # Stores number of indexed items

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Recommended: Load API key from environment variable

# LLM_MODEL_LOCAL = "distilbert-base-uncased-finetuned-sst-2-english" # Example local model
# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # Replace with your key

# --- 1. Connect to MongoDB ---
def connect_to_db(uri, db_name):
    """Connects to MongoDB and returns the database object."""
    try:
        client = MongoClient(uri)
        client.admin.command('ping') # Verify connection
        logger.info("Successfully connected to MongoDB!")
        return client[db_name]
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None

# --- 2. Load Jira Data ---
def get_jira_issues(db, collection_name, batch_size=1000, initial_skip=0): # Added initial_skip
    """Fetches Jira issues from the specified collection in batches and yields them, starting from initial_skip."""
    if not db:
        logger.error("Database connection not available for get_jira_issues.")
        return

    collection = db[collection_name]
    skip = initial_skip # Start skipping from here
    total_fetched_this_run = 0
    logger.info(f"Starting to fetch Jira issues from offset {initial_skip}.")
    while True:
        # Fetch a batch: project only the nested summary, description, and issuetype name
        # Materialize the cursor to a list to check its length and process
        raw_batch_cursor = collection.find(
            {},
            {"fields.summary": 1, "fields.description": 1, "key": 1, "fields.issuetype.name": 1, "_id": 0}
        ).skip(skip).limit(batch_size)
        
        raw_batch = list(raw_batch_cursor) # Convert cursor to list

        if not raw_batch:
            logger.info(f"No more issues to fetch from {collection_name} (current skip: {skip}). Total fetched in this run (after initial_skip): {total_fetched_this_run} issues.")
            break # No more documents

        issues_batch = []
        for doc in raw_batch:
            fld = doc.get("fields", {})
            issues_batch.append({
                "key": doc.get("key"),
                "summary": fld.get("summary", ""),
                "description": fld.get("description", ""),
                "issuetype": fld.get("issuetype", {}).get("name", "") # Extract issuetype name
            })
        
        logger.info(f"Fetched batch of {len(issues_batch)} issues from {collection_name} (current skip={skip}, initial_skip={initial_skip}).")
        total_fetched_this_run += len(issues_batch)
        yield issues_batch # Yield the processed batch
        
        skip += batch_size
        # Optimization: if last batch fetched was smaller than batch_size, it's the end of the collection
        if len(raw_batch) < batch_size:
            logger.info(f"Last batch fetched ({len(raw_batch)}) was smaller than batch_size ({batch_size}). Assuming end of collection. Total fetched in this run (after initial_skip): {total_fetched_this_run}.")
            break

# --- 3. Generate Embeddings ---
def generate_embeddings(issues, model, model_name):
    """Generates embeddings for Jira issues using a pre‐loaded model."""
    # Prepare texts, keys, and issuetypes
    raw_texts = [f"{i.get('summary','')} {i.get('description','')}" for i in issues]
    keys = [i.get('key') for i in issues]
    issuetypes = [i.get('issuetype', '') for i in issues] # Extract issuetype

    if "-instruct" in model_name.lower():
        texts_to_embed = raw_texts
        logger.info("Using instruct model: embedding documents as is.")
    else:
        texts_to_embed = [f"passage: {text}" for text in raw_texts]
        logger.info("Using base E5 model: prepending 'passage:' to documents.")

    logger.info(f"Encoding {len(texts_to_embed)} documents in batches...")
    try:
        embeddings_list = model.encode(texts_to_embed, batch_size=32, convert_to_tensor=False, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"Batch encoding failed: {e}")
        return []

    embeddings_data = []
    # Use raw_texts for storing the original text, not the potentially prefixed one
    for key, text, emb, issuetype_val in zip(keys, raw_texts, embeddings_list, issuetypes):
        embeddings_data.append({
            "issue_key": key,
            "text": text,  # Store original text
            "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb), # Storing as list for JSON
            "issuetype": issuetype_val # Store issuetype
        })
    logger.info(f"Generated {len(embeddings_data)} embeddings.")
    return embeddings_data

# --- 4. Store Embeddings (Example: JSON file, consider a vector database for larger datasets) ---
# This function is no longer primarily used for the main index embeddings, as they are processed in batches.
# It can be kept for other purposes or removed if not needed.
def store_embeddings(embeddings_data, filename="jira_embeddings.json"):
    """Stores the given embeddings data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(embeddings_data, f, indent=4)
        logger.info(f"Embeddings stored in {filename}")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")

def build_faiss_index(embeddings_data, dim, index_path="jira_index.faiss"):
    """Builds a FAISS index from all embeddings_data and saves it.
    NOTE: This function processes all embeddings in memory. For large datasets,
    use the batched indexing approach in the main script.
    """
    if not embeddings_data:
        logger.error("No embeddings data provided to build_faiss_index.")
        return None
    vectors = np.array([item['embedding'] for item in embeddings_data]).astype('float32')
    index = faiss.IndexFlatIP(dim) 
    index.add(vectors)
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index with {index.ntotal} vectors built and saved to {index_path}")
    return index


def search_faiss(query_text, model, index, index_metadata_list, model_name_str, top_n=3):
    """Searches the FAISS index for top_n similar issues to the query_text."""
    
    # model_name_str = model.config.name_or_path # Get model name from the loaded model instance - This was causing an error
    
    if "-instruct" in model_name_str.lower():
        # Encode query with instruction as per model card and normalize
        instruct_query = f"Instruct: {TASK_DESCRIPTION}\\nQuery: {query_text}"
        logger.info(f"Using instruct model: formatting query with TASK_DESCRIPTION: {instruct_query[:100]}...")
    else:
        # Encode query with "query: " prefix for base E5 models
        instruct_query = f"query: {query_text}"
        logger.info(f"Using base E5 model: formatting query with 'query: ' prefix: {instruct_query[:100]}...")
        
    q_emb = model.encode(instruct_query, convert_to_tensor=False, normalize_embeddings=True)
    q_emb = np.array(q_emb).astype('float32').reshape(1, -1)
    # Query embedding is now normalized by SentenceTransformer
    # faiss.normalize_L2(q_emb)
    # Search
    distances, indices = index.search(q_emb, top_n)
    results = []
    for sim, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(index_metadata_list): # Check index bounds
            item = index_metadata_list[idx]
            results.append({
                'issue_key': item['issue_key'],
                'similarity': float(sim),
                'text': item['text'],
                'issuetype': item.get('issuetype', '') 
            })
        else:
            logger.warning(f"Invalid index {idx} from FAISS search results. Max index: {len(index_metadata_list)-1}")
    return results

def find_similar_issues(*args, **kwargs):
    """Deprecated stub; use search_faiss for similarity search via FAISS index."""
    raise NotImplementedError("Use search_faiss() with FAISS index for efficient similarity queries.")

def generate_rag_prompt(query_ticket_summary, similar_issues):
    """Generates a prompt for the LLM based on the query and similar issues."""
    prompt = "Classify the following new Jira ticket by suggesting the most appropriate 'issuetype'.\\n\\n"
    prompt += f"**New Ticket to Classify:**\\n"
    prompt += f"Summary: '{query_ticket_summary}'\\n\\n"

    prompt += "**Historical Context:**\\n"
    if similar_issues:
        prompt += "Here are some historically similar tickets that might be relevant:\\n"
        for i, issue in enumerate(similar_issues):
            prompt += f"\\n--- Similar Ticket {i+1} (Key: {issue['issue_key']}, Similarity: {issue['similarity']:.4f}) ---\\n"
            ticket_text = issue['text']
            prompt += f"Text: {ticket_text}\\n"
            if 'issuetype' in issue and issue['issuetype']:
                 prompt += f"Original Issue Type: {issue['issuetype']}\\n"
        prompt += "\\n--- End of Similar Tickets ---\\n"
    else:
        prompt += "No similar historical tickets were found.\\n"

    prompt += "\\n**Your Task:**\\n"
    prompt += "1. Analyze the summary of the new Jira ticket provided above.\\n"
    prompt += "2. Review the historical context (if any similar tickets were provided), paying attention to how the new ticket relates to or contrasts with them, especially concerning their 'issuetype'.\\n"
    prompt += "3. Based on your analysis, suggest a single, most appropriate 'issuetype' for the new ticket.\\n"
    prompt += "Consider the following common issuetypes for Apache: Improvement, Task, Sub-task, New Feature, Bug, Epic, Test, Wish, New JIRA Project, RTC, TCK Challenge, Question, Temp, Brainstorming, Umbrella, Story, Technical task, Dependency upgrade, Suitable Name Search, Documentation, Planned Work, New Confluence Wiki, New Git Repo, Github Integration, New TLP , New TLP - Common Tasks, SVN->GIT Migration, Blog - New Blog Request, Blogs - New Blog User Account Request, Blogs - Access to Existing Blog, New Bugzilla Project, SVN->GIT Mirroring, IT Help, Access, Request, Project, Proposal, GitBox Request, Dependency, Requirement, Comment, Choose from below ..., Outage, Office Hours, Pending Review, Board Vote, Director Vote, Technical Debt.\\n"
    prompt += "Provide your suggested issuetype and a brief reasoning for your choice, justifying why it's the best fit.\\n"
    return prompt

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Connect to DB
    db = connect_to_db(MONGO_URI, DB_NAME)

    if db:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {embedding_dim}")
 
        MONGO_BATCH_SIZE = 100000  # Adjust as needed for your system's memory and DB performance
                                 # Smaller for less RAM during embedding generation per batch
                                 # Larger for fewer DB calls.

        faiss_index = None
        index_metadata = []
        num_indexed_previously = 0
        faiss_res = None

        if DEVICE == 'cuda':
            try:
                faiss_res = faiss.StandardGpuResources()
                logger.info("FAISS GPU resources initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize FAISS GPU resources: {e}. Falling back to CPU for FAISS.")
                DEVICE_FAISS = 'cpu' # Fallback for FAISS specifically
            else:
                DEVICE_FAISS = 'cuda'
        else:
            DEVICE_FAISS = 'cpu'
        logger.info(f"FAISS will use: {DEVICE_FAISS}")


        # Load progress if available
        if os.path.exists(INDEX_PROGRESS_FILE):
            try:
                with open(INDEX_PROGRESS_FILE, 'r') as f:
                    progress_data = json.load(f)
                    num_indexed_previously = progress_data.get("num_indexed", 0)
                logger.info(f"Loaded progress: {num_indexed_previously} items previously indexed.")

                if num_indexed_previously > 0:
                    if os.path.exists(INDEX_METADATA_FILE):
                        with open(INDEX_METADATA_FILE, 'r') as f:
                            index_metadata = json.load(f)
                        logger.info(f"Loaded {len(index_metadata)} metadata entries.")
                    else:
                        logger.warning(f"Progress file exists but metadata file {INDEX_METADATA_FILE} not found. May need to re-index.")
                        num_indexed_previously = 0 # Force re-index if metadata is missing

                    if os.path.exists(FAISS_INDEX_FILE):
                        logger.info(f"Loading existing FAISS index from {FAISS_INDEX_FILE}")
                        faiss_index_cpu = faiss.read_index(FAISS_INDEX_FILE)
                        if faiss_index_cpu.ntotal != num_indexed_previously or len(index_metadata) != num_indexed_previously:
                            logger.warning(f"Index/metadata mismatch (Index: {faiss_index_cpu.ntotal}, Meta: {len(index_metadata)}, Progress: {num_indexed_previously}). Resetting and re-indexing.")
                            num_indexed_previously = 0
                            index_metadata = []
                            faiss_index = None # Will be re-initialized
                        else:
                            if DEVICE_FAISS == 'cuda' and faiss_res:
                                try:
                                    faiss_index = faiss.index_cpu_to_gpu(faiss_res, 0, faiss_index_cpu)
                                    logger.info("FAISS index moved to GPU.")
                                except Exception as e:
                                    logger.error(f"Failed to move FAISS index to GPU: {e}. Using CPU index.")
                                    faiss_index = faiss_index_cpu
                            else:
                                faiss_index = faiss_index_cpu
                            logger.info(f"Successfully loaded FAISS index with {faiss_index.ntotal} vectors.")
                    else:
                        logger.warning(f"Progress/metadata indicate existing index, but {FAISS_INDEX_FILE} not found. Resetting.")
                        num_indexed_previously = 0
                        index_metadata = []
                        faiss_index = None
            except Exception as e:
                logger.error(f"Error loading progress/index: {e}. Starting fresh.")
                num_indexed_previously = 0
                index_metadata = []
                faiss_index = None
        
        if faiss_index is None:
            logger.info(f"Initializing new FAISS index (Dimension: {embedding_dim}).")
            faiss_index_cpu = faiss.IndexFlatIP(embedding_dim)
            if DEVICE_FAISS == 'cuda' and faiss_res:
                try:
                    faiss_index = faiss.index_cpu_to_gpu(faiss_res, 0, faiss_index_cpu)
                    logger.info("New FAISS index created on GPU.")
                except Exception as e:
                    logger.error(f"Failed to create FAISS index on GPU: {e}. Using CPU index.")
                    faiss_index = faiss_index_cpu # Fallback to CPU index object
            else:
                faiss_index = faiss_index_cpu # Use CPU index object
            index_metadata = [] # Ensure metadata is also reset
            num_indexed_previously = 0


        # Main processing loop
        logger.info(f"Starting/resuming Jira issue processing. Will skip the first {num_indexed_previously} issues from MongoDB.")
        
        # get_jira_issues now takes initial_skip
        for issues_batch in get_jira_issues(db, COLLECTION_NAME, batch_size=MONGO_BATCH_SIZE, initial_skip=num_indexed_previously):
            if not issues_batch:
                logger.info("Received an empty batch or no more new issues to process.")
                continue

            logger.info(f"Processing batch of {len(issues_batch)} new issues for embedding and indexing.")
            
            # Generate embeddings for the current new batch
            # generate_embeddings internal batch_size=32 is for SentenceTransformer model.encode
            batch_embeddings_data = generate_embeddings(issues_batch, embedding_model, EMBEDDING_MODEL)
            
            if batch_embeddings_data:
                vectors_list = [item['embedding'] for item in batch_embeddings_data]
                current_batch_metadata = [
                    {"issue_key": item["issue_key"], "text": item["text"], "issuetype": item["issuetype"]}
                    for item in batch_embeddings_data
                ]

                if vectors_list:
                    vectors_np = np.array(vectors_list).astype('float32')
                    
                    try:
                        faiss_index.add(vectors_np)
                        index_metadata.extend(current_batch_metadata)
                        
                        # Save progress
                        current_total_indexed = faiss_index.ntotal
                        
                        # Save FAISS index (CPU version)
                        if DEVICE_FAISS == 'cuda' and faiss_res and hasattr(faiss_index, 'is_GpuIndex') and faiss_index.is_GpuIndex(): # Check if it's actually a GPU index
                            faiss_index_cpu_to_save = faiss.index_gpu_to_cpu(faiss_index)
                            faiss.write_index(faiss_index_cpu_to_save, FAISS_INDEX_FILE)
                        else:
                            faiss.write_index(faiss_index, FAISS_INDEX_FILE)
                        
                        with open(INDEX_METADATA_FILE, 'w') as f:
                            json.dump(index_metadata, f) # Overwrite with updated full list
                        with open(INDEX_PROGRESS_FILE, 'w') as f:
                            json.dump({"num_indexed": current_total_indexed}, f)
                        
                        logger.info(f"Successfully indexed batch. Total items in index: {current_total_indexed}. Progress saved.")
                        num_indexed_previously = current_total_indexed # Update for next potential resume

                    except Exception as e:
                        logger.error(f"Error adding to FAISS index or saving: {e}")
                        # Potentially break or implement more robust retry/rollback
                        break 
                else:
                    logger.warning("No vectors extracted from batch_embeddings_data. Skipping FAISS add for this batch.")
            else:
                logger.warning(f"generate_embeddings returned no data for a batch of {len(issues_batch)} issues. This batch will be skipped.")

        logger.info(f"Finished processing all available Jira issues. Final index size: {faiss_index.ntotal if faiss_index else 'N/A'}.")

        if faiss_index and faiss_index.ntotal > 0 and index_metadata:
            logger.info(f"FAISS index ready with {faiss_index.ntotal} items.")

            # Example: Find issues similar to a new ticket description
            new_ticket_summary = "timeout error when trying to access the API endpoint"
            logger.info(f"Finding similar issues for: '{new_ticket_summary}'")
            
            similar_issues = search_faiss(new_ticket_summary, embedding_model, faiss_index, index_metadata, EMBEDDING_MODEL, top_n=3)

            if similar_issues:
                logger.info("Top similar issues found via FAISS:")
                for issue in similar_issues:
                    snippet = issue['text']
                    if isinstance(snippet, str) and snippet.lower().startswith('passage:'):
                        snippet = snippet[len('passage:'):].strip()
                    logger.debug(
                        f"Key: {issue['issue_key']} | Similarity: {issue['similarity']:.4f} | Text snippet: {snippet[:100]}..."
                    )

                # Generate a prompt for an LLM
                rag_prompt = generate_rag_prompt(new_ticket_summary, similar_issues)
                logger.debug("--- RAG Prompt for LLM ---")
                logger.debug(rag_prompt)
                logger.debug("--- End of RAG Prompt ---")

                # Assuming genai is already imported and configured as in the original snippet
                import google.generativeai as genai
                try:
                    if not GEMINI_API_KEY:
                        logger.error("GEMINI_API_KEY environment variable not set. Cannot use Gemini API.")
                        raise ValueError("Gemini API key not configured.")
                    
                    genai.configure(api_key=GEMINI_API_KEY) 
                    
                    gemini_model_name = 'gemini-2.5-flash-preview-04-17' # Or your preferred Gemini model
                    gemini_model = genai.GenerativeModel(gemini_model_name)
                    response = gemini_model.generate_content(rag_prompt)
                    logger.info("\\nLLM (Gemini) Response:")
                    logger.info(response.text)
                except Exception as e:
                    logger.error(f"Error with Gemini API: {e}")
                    logger.error("Please ensure you have the google-generativeai package installed (pip install google-generativeai)")
                    logger.error("And that your GEMINI_API_KEY environment variable is correctly set.")
            else:
                logger.info("No similar issues found to generate RAG prompt.")
        elif not (faiss_index and faiss_index.ntotal > 0):
            logger.error("FAISS index is empty or not initialized. Cannot perform search or RAG. Please check data source and indexing process.")
        else: # index_metadata is empty but index is not (should not happen with current logic)
             logger.error("FAISS index metadata is empty, though index has items. This indicates an inconsistency. Cannot perform search effectively.")


    logger.info("Script finished.")
