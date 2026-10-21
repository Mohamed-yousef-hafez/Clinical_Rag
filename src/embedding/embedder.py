
import chromadb
from sentence_transformers import SentenceTransformer

from src.config.settings import settings


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH
        )

        self.collection = None

    def store(self, chunks):
        if not chunks:
            raise ValueError("No chunks available to store.")

        try:
            self.client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 16

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [chunk["text"] for chunk in batch]

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).tolist()

            ids = [
                f"chunk_{start + i}"
                for i in range(len(batch))
            ]

            metadatas = [
                {
                    "page": str(chunk.get("page", "Unknown")),
                    "document": str(
                        chunk.get("document", "Hypertension Guideline")
                    ),
                    "section": str(chunk.get("section", "Unknown")),
                }
                for chunk in batch
            ]

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        print(f"Stored {len(chunks)} chunks in ChromaDB.")
