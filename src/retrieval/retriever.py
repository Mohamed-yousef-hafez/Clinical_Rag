import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from src.config.settings import settings


class Retriever:

    def __init__(self):

        print("Loading Embedding Model...")

        self.encoder = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

        print("Embedding Model Loaded")

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            settings.COLLECTION_NAME
        )

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        print("Preparing BM25 index...")

        self.bm25 = BM25Okapi(
            [
                document.lower().split()
                for document in self.documents
            ]
        )

        # Lazy loading
        self.reranker = None

        print(
            f"Loaded {len(self.documents)} chunks"
        )

        print("Hybrid Retriever Ready")

    def _load_reranker(self):

        if self.reranker is None:

            print("Loading Cross-Encoder...")

            self.reranker = CrossEncoder(
                settings.RERANKER_MODEL
            )

            print("Cross-Encoder Loaded")

        return self.reranker

    def semantic_search(self, question, top_k=None):

        top_k = top_k or settings.TOP_K

        embedding = self.encoder.encode(
            question,
            normalize_embeddings=True
        )

        result = self.collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        chunks = []

        for i, text in enumerate(
            result["documents"][0]
        ):

            metadata = result["metadatas"][0][i]

            chunks.append({
                "text": text,
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "section": metadata.get(
                    "section",
                    "Unknown"
                ),
                "document": metadata.get(
                    "document",
                    "Clinical Guideline"
                ),
                "distance": result["distances"][0][i]
            })

        return chunks

    def bm25_search(self, question, top_k=None):

        top_k = top_k or settings.TOP_K

        scores = self.bm25.get_scores(
            question.lower().split()
        )

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in indices:

            metadata = self.metadatas[index]

            results.append({
                "text": self.documents[index],
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "section": metadata.get(
                    "section",
                    "Unknown"
                ),
                "document": metadata.get(
                    "document",
                    "Clinical Guideline"
                ),
                "bm25_score": float(
                    scores[index]
                )
            })

        return results

    def search(self, question):

        semantic_results = self.semantic_search(
            question
        )

        bm25_results = self.bm25_search(
            question
        )

        combined = {}

        for chunk in (
            semantic_results + bm25_results
        ):

            key = (
                chunk.get("page"),
                chunk.get("text")
            )

            if key not in combined:

                combined[key] = chunk

        candidates = list(
            combined.values()
        )

        if not candidates:

            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low"
            }

        candidates = candidates[
            :settings.TOP_K
        ]

        best_distance = min(
            [
                c.get("distance", 1.0)
                for c in candidates
            ],
            default=1.0
        )

        if best_distance <= 0.30:

            confidence = "High"

        elif best_distance <= 0.70:

            confidence = "Medium"

        else:

            confidence = "Low"

        if confidence == "Low":

            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low"
            }

        return {
            "status": "success",
            "chunks": candidates,
            "confidence": confidence
        }

    def rerank(self, question, chunks):

        if not chunks:

            return []

        model = self._load_reranker()

        pairs = [
            (question, chunk["text"])
            for chunk in chunks
        ]

        scores = model.predict(
            pairs,
            batch_size=2
        )

        for chunk, score in zip(
            chunks,
            scores
        ):

            chunk["rerank_score"] = float(score)

        chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return chunks[
            :settings.RERANK_TOP_K
        ]