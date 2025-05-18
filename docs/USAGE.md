# Usage

## Setup

1. Create a Python virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy and populate environment variables:

   ```powershell
   copy config\.env.example .env
   # Then open .env and fill in GEMINI_API_KEY etc.
   ```

## Indexing

Build or resume the FAISS index:

```powershell
python main.py index [--debug]
```

- `--debug` enables verbose logging.

## Querying

Find similar tickets and optionally generate a RAG prompt:

```powershell
python main.py query --text "Your ticket summary" [--rag] [--debug]
```

## Example

```powershell
python main.py index
python main.py query --text "timeout error when accessing API" --rag
```
