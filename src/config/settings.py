import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    PROJECT_NAME = "Clinical Evidence Assistant"
    DOMAIN = "Hypertension"
    VERSION = "1.0"

    DATA_FOLDER = "data/docs"
    DOCUMENT_NAME = "guideline_hypertension.pdf"
    PDF_PATH = os.path.join(DATA_FOLDER, DOCUMENT_NAME)

    CHROMA_PATH = "chroma_db"
    COLLECTION_NAME = "hypertension_guidelines"

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    TOP_K = 5
    RERANK_TOP_K = 3

    SIMILARITY_THRESHOLD = 0.70

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


settings = Settings()