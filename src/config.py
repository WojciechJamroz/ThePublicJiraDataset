import os
from dotenv import load_dotenv
import torch  # add for device detection

# Load environment variables
load_dotenv()

# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "JiraRepos")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Apache")

# Embedding model and task description
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
TASK_DESCRIPTION = os.getenv("TASK_DESCRIPTION", "Given a new Jira ticket summary, retrieve relevant historical tickets for triage classification")

# FAISS index file paths
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "jira_index.faiss")
INDEX_METADATA_FILE = os.getenv("INDEX_METADATA_FILE", "jira_index_metadata.json")
INDEX_PROGRESS_FILE = os.getenv("INDEX_PROGRESS_FILE", "jira_index_progress.json")

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Default Gemini model name for API calls
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-04-17")

# MongoDB batch size (int)
try:
    MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "100000"))
except ValueError:
    MONGO_BATCH_SIZE = 100000

# Determine device for embeddings and FAISS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
