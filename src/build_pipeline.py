"""
Build the Knowledge Base

Pipeline:
1. Load Hypertension Guideline PDF
2. Clean Text
3. Recursive Character Chunking
4. Generate BGE Embeddings
5. Store in ChromaDB
"""

from src.ingestion.loader import PDFLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import TextChunker
from src.embedding.embedder import Embedder
from src.config.settings import settings


def main():

    print("=" * 50)
    print("BUILDING HYPERTENSION KNOWLEDGE BASE")
    print("=" * 50)

    # -----------------------------------------
    # 1. Load PDF
    # -----------------------------------------

    print("Loading PDF...")

    loader = PDFLoader(
        settings.PDF_PATH
    )

    pages = loader.load()

    print(
        f"✓ PDF Loaded: {len(pages)} pages"
    )

    # -----------------------------------------
    # 2. Clean Text
    # -----------------------------------------

    print("Cleaning text...")

    cleaner = TextCleaner()

    clean_pages = cleaner.clean(
        pages
    )

    print("✓ Text Cleaned")

    # -----------------------------------------
    # 3. Chunk Text
    # -----------------------------------------

    print("Creating chunks...")

    chunker = TextChunker()

    chunks = chunker.chunk(
        clean_pages
    )

    print(
        f"✓ {len(chunks)} Chunks Created"
    )

    # -----------------------------------------
    # 4. Generate Embeddings
    # 5. Store in ChromaDB
    # -----------------------------------------

    print("Generating embeddings...")

    embedder = Embedder()

    embedder.store(
        chunks
    )

    print("✓ Chunks Stored in ChromaDB")

    print("=" * 50)
    print("✓ HYPERTENSION KNOWLEDGE BASE READY")
    print("=" * 50)


if __name__ == "__main__":
    main()