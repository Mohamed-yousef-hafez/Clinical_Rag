# Clinical Evidence Assistant

A Streamlit RAG application that answers hypertension questions using the included clinical guideline PDF, retrieves supporting evidence, cites page numbers, and blocks basic prompt-injection attempts.

## Requirements

- Python 3.12
- A Gemini API key for answer generation
- Internet access on first launch to download the embedding and reranker models

## Run locally (Windows PowerShell)

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root containing:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Do not commit `.env` or share your API key.

Start the app:

```powershell
python -m streamlit run app.py
```

The knowledge base is built automatically if the ChromaDB collection does not exist. To build it manually:

```powershell
python -m src.build_pipeline
```

## Streamlit Community Cloud

1. Push the project to GitHub.
2. In the app's **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-3.6-flash"
```

3. Set the main file path to `app.py`.
4. Deploy. The app will build its local ChromaDB index if it is not already present in the deployment filesystem.

Never put a real API key in GitHub, source code, screenshots, or this README.

## Tests

Run the lightweight local tests:

```powershell
python -m unittest discover -s tests -v
```

These tests cover basic safety classification, refusal behavior, precision@K, and citation bookkeeping. They do not replace an end-to-end test against the live Gemini API.
