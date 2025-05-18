import json
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import torch

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
def get_jira_issues(db, collection_name, batch_size=1000): # Added batch_size, removed limit
    """Fetches Jira issues from the specified collection in batches and yields them."""
    if not db:
        logger.error("Database connection not available for get_jira_issues.")
        return

    collection = db[collection_name]
    skip = 0
    total_fetched_this_run = 0
    while True:
        # Fetch a batch: project only the nested summary, description, and issuetype name
        # Materialize the cursor to a list to check its length and process
        raw_batch_cursor = collection.find(
            {},
            {"fields.summary": 1, "fields.description": 1, "key": 1, "fields.issuetype.name": 1, "_id": 0}
        ).skip(skip).limit(batch_size)
        
        raw_batch = list(raw_batch_cursor) # Convert cursor to list

        if not raw_batch:
            logger.info(f"No more issues to fetch from {collection_name} after fetching a total of {total_fetched_this_run} issues in this run.")
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
        
        logger.info(f"Fetched batch of {len(issues_batch)} issues from {collection_name} (skip={skip}).")
        total_fetched_this_run += len(issues_batch)
        yield issues_batch # Yield the processed batch
        
        skip += batch_size
        # Optimization: if last batch fetched was smaller than batch_size, it's the end of the collection
        if len(raw_batch) < batch_size:
            logger.info(f"Last batch fetched ({len(raw_batch)}) was smaller than batch_size ({batch_size}). Assuming end of collection. Total fetched in this run: {total_fetched_this_run}.")
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
            "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb),
            "issuetype": issuetype_val # Store issuetype
        })
    logger.info(f"Generated {len(embeddings_data)} embeddings.")
    return embeddings_data

# --- 4. Store Embeddings (Example: JSON file, consider a vector database for larger datasets) ---
def store_embeddings(embeddings_data, filename="jira_embeddings.json"):
    """Stores the generated embeddings to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(embeddings_data, f, indent=4)
        logger.info(f"Embeddings stored in {filename}")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")

def build_faiss_index(embeddings_data, dim, index_path="jira_index.faiss"):
    """Builds a FAISS index of normalized embeddings and saves it."""
    vectors = np.array([item['embedding'] for item in embeddings_data]).astype('float32')
    # Embeddings are now normalized by SentenceTransformer, so no need for manual normalization here
    # faiss.normalize_L2(vectors) 
    index = faiss.IndexFlatIP(dim) # IndexFlatIP is for inner product, suitable for normalized vectors (cosine similarity)
    index.add(vectors)
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index built and saved to {index_path}")
    return index


def search_faiss(query_text, model, index, embeddings_data, model_name_str, top_n=3):
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
        item = embeddings_data[idx]
        results.append({
            'issue_key': item['issue_key'],
            'similarity': float(sim),
            'text': item['text'],
            'issuetype': item.get('issuetype', '') # Include issuetype in results
        })
    return results

# --- 5. RAG System (Conceptual) ---
# This section will depend on your choice of LLM (local or API)
# and how you want to perform retrieval (e.g., cosine similarity with embeddings)

# Example with a local LLM (conceptual)
# def setup_local_llm(model_name):
#     """Sets up a local LLM pipeline."""
#     return pipeline("text-generation", model=model_name) # Or question-answering, etc.

# Example with OpenAI API (conceptual)
# def setup_openai_api(api_key):
#     """Sets up OpenAI API."""
#     openai.api_key = api_key

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
        # Load embeddings model only once
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        logger.info("Embedding model loaded.")
 
        # Configuration for batch processing from MongoDB
        MONGO_BATCH_SIZE = 10000  # Adjust as needed for your system's memory and DB performance

        embeddings_file = "jira_embeddings.json" 
        embeddings_data = [] # Initialize embeddings_data

        try:
            with open(embeddings_file, 'r') as f:
                logger.info(f"Loading existing embeddings from {embeddings_file}")
                embeddings_data_loaded = json.load(f)
                # Basic check for 'issuetype' key in the first item if data exists
                if embeddings_data_loaded and (not isinstance(embeddings_data_loaded, list) or not embeddings_data_loaded[0] or 'issuetype' not in embeddings_data_loaded[0]):
                    logger.warning(f"Embeddings file {embeddings_file} might be in an old format or corrupted (missing 'issuetype' in first item). Regenerating embeddings.")
                    raise FileNotFoundError # Force regeneration
                embeddings_data = embeddings_data_loaded
                logger.info(f"Successfully loaded {len(embeddings_data)} embeddings from file.")

        except FileNotFoundError:
            logger.info(f"No existing embeddings file found at {embeddings_file} or it's marked for regeneration. Generating new embeddings.")
            
            all_accumulated_embeddings_data = []
            total_issues_processed_for_embedding = 0
            
            # get_jira_issues now returns a generator of batches
            for issues_batch in get_jira_issues(db, COLLECTION_NAME, batch_size=MONGO_BATCH_SIZE):
                if not issues_batch: # Should be handled by the generator's break condition, but as a safeguard
                    logger.info("Received an empty batch, continuing if more batches are expected or stopping if generator finished.")
                    continue

                logger.info(f"Generating embeddings for a batch of {len(issues_batch)} issues. Total processed for embedding so far: {total_issues_processed_for_embedding}")
                
                # generate_embeddings processes the current batch
                # Its internal batch_size=32 is for the SentenceTransformer model.encode, not MongoDB fetching
                batch_embeddings_data = generate_embeddings(issues_batch, embedding_model, EMBEDDING_MODEL)
                
                if batch_embeddings_data:
                    all_accumulated_embeddings_data.extend(batch_embeddings_data)
                    # Count based on successfully generated embeddings for this batch
                    total_issues_processed_for_embedding += len(batch_embeddings_data) 
                else:
                    # Log if a batch resulted in no embeddings, could indicate issues with input data for that batch
                    logger.warning(f"generate_embeddings returned no data for a batch of {len(issues_batch)} issues. This batch will be skipped.")

            embeddings_data = all_accumulated_embeddings_data # Assign accumulated data
            
            if embeddings_data:
                logger.info(f"Total {len(embeddings_data)} embeddings generated. Storing them...")
                store_embeddings(embeddings_data, embeddings_file)
            else:
                logger.error("No embeddings were generated after processing all batches. Please check data source and logs.")
                # The script will proceed, but FAISS indexing will likely fail or be empty.

        if embeddings_data: # Check if embeddings_data has content (either loaded or generated)
            logger.info(f"Successfully loaded/generated {len(embeddings_data)} embeddings.")

            # --- FAISS Integration for efficient similarity search ---
            index_file = "jira_index.faiss"
            try:
                index = faiss.read_index(index_file)
                logger.info(f"Loaded existing FAISS index from {index_file}")
            except Exception:
                logger.info("Building FAISS index...")
                index = build_faiss_index(embeddings_data, len(embeddings_data[0]['embedding']), index_file)

            # --- 5. RAG Example ---
            # This is a placeholder for how you might use the embeddings.
            # You'll need to integrate your chosen LLM here.

            # Example: Find issues similar to a new ticket description
            new_ticket_summary = "timeout error when trying to access the API endpoint"
            logger.info(f"Finding similar issues for: '{new_ticket_summary}'")
            
            # reuse the already loaded embedding_model for search
            similar_issues = search_faiss(new_ticket_summary, embedding_model, index, embeddings_data, EMBEDDING_MODEL, top_n=3)

            logger.info("Top similar issues found via FAISS:")
            for issue in similar_issues:
                # Clean up snippet by removing any leading 'passage:' prefix
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

            import google.generativeai as genai
              
            try:
                genai.configure(api_key="AIzaSyBbjIOvqYCUNEw5Of5BiIJZhGGnirBBMCc") # Replace with your actual key
                model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Or 'gemini-1.5-flash', 'gemini-1.0-pro', etc.
                response = model.generate_content(rag_prompt)
                logger.info("\\nLLM (Gemini) Response:")
                logger.info(response.text)
            except Exception as e:
                logger.error(f"Error with Gemini API: {e}")
                logger.error("Please ensure you have the google-generativeai package installed (pip install google-generativeai)")
                logger.error("And that your API key is correctly configured.")

        else:
            logger.error("Embeddings data is empty. Cannot proceed with FAISS index or RAG. Please check data source and embedding generation process.")

    logger.info("Script finished.")
