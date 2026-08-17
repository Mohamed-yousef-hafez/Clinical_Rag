"""
Embedding Generator

- Semantic Retrieval
- Vector Store
- Embedding Generation
"""

import chromadb

from sentence_transformers import SentenceTransformer

from src.config.settings import settings


class Embedder:

    def __init__(self):

        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH
        )

        # -----------------------------------------
        # Get existing collection
        # -----------------------------------------

        try:

            self.collection = self.client.get_collection(
                name=settings.COLLECTION_NAME
            )

            print(
                f"✓ Existing collection loaded: "
                f"{settings.COLLECTION_NAME}"
            )

        except Exception:

            self.collection = self.client.create_collection(
                name=settings.COLLECTION_NAME
            )

            print(
                f"✓ New collection created: "
                f"{settings.COLLECTION_NAME}"
            )

        print("Embedding Module Ready")

    # =====================================================
    # Store Embeddings
    # =====================================================

    def store(self, chunks):

        if not chunks:

            raise ValueError(
                "No chunks available to store."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = []

        for chunk in chunks:

            metadatas.append(
                {
                    "page": str(
                        chunk.get(
                            "page",
                            "Unknown"
                        )
                    ),
                    "document": chunk.get(
                        "document",
                        "Clinical Guideline"
                    ),
                    "section": chunk.get(
                        "section",
                        "Unknown"
                    )
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"✓ Stored {len(chunks)} chunks"
        )

        print(
            f"✓ Collection: "
            f"{settings.COLLECTION_NAME}"
        )