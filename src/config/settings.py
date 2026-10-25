import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings:
    PROJECT_NAME = "Clinical Evidence Assistant"
    DOMAIN = "Hypertension"
    VERSION = "1.0"

    DATA_FOLDER = str(ROOT_DIR / "data" / "docs")
    DOCUMENT_NAME = "guideline_hypertension.pdf"
    PDF_PATH = str(Path(DATA_FOLDER) / DOCUMENT_NAME)

    CHROMA_PATH = str(ROOT_DIR / "chroma_db")
    COLLECTION_NAME = "hypertension_guidelines"

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    TOP_K = 5
    RERANK_TOP_K = 3

    # Cosine distance thresholds; lower is a closer match.
    SIMILARITY_THRESHOLD = 0.70
    DISTANCE_THRESHOLD = 0.70
    HIGH_CONFIDENCE_DISTANCE = 0.30

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()


settings = Settings()

# Streamlit Community Cloud stores secrets in st.secrets rather than .env.
if not (settings_key := os.getenv("GEMINI_API_KEY", "").strip()):
    try:
        import streamlit as st
        _streamlit_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        _streamlit_key = ""
else:
    _streamlit_key = settings_key

settings.GEMINI_API_KEY = _streamlit_key
