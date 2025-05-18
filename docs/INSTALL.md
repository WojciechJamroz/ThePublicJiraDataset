# Installation

Follow these steps to set up the Jira RAG Bootstrap project locally.

## 1. Clone the repository
```powershell
git clone https://github.com/your-org/jira-rag-bootstrap.git
cd jira-rag-bootstrap
```

## 2. Create a Python virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```powershell
pip install -r requirements.txt
```

## 4. Configure environment variables
Copy the example and populate with your own values (especially API keys):
```powershell
copy config\.env.example .env
# Then edit .env to set MONGO_URI, GEMINI_API_KEY, etc.
``` 

## 5. Verify setup
Run a quick help command to ensure CLI is working:
```powershell
python main.py --help
```
