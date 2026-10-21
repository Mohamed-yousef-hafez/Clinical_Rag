
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from src.config.settings import settings


class Retriever:
    def __init__(self):
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=settings.COLLECTION_NAME
        )

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []

        if self.documents:
            tokenized = [
                document.lower().split()
                for document in self.documents
            ]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

        self.reranker = None

    def _load_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(
                settings.RERANKER_MODEL
            )
        return self.reranker

    def semantic_search(self, question, top_k=None):
        top_k = top_k or settings.TOP_K

        if not self.documents:
            return []

        embedding = self.encoder.encode(
            question,
            normalize_embeddings=True,
        )

        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, len(self.documents)),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        for text, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            metadata = metadata or {}

            chunks.append({
                "text": text,
                "page": metadata.get("page", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "document": metadata.get(
                    "document", "Hypertension Guideline"
                ),
                "distance": float(distance),
            })

        return chunks

    def bm25_search(self, question, top_k=None):
        top_k = top_k or settings.TOP_K

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(
            question.lower().split()
        )

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in indices:
            metadata = self.metadatas[index] or {}

            results.append({
                "text": self.documents[index],
                "page": metadata.get("page", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "document": metadata.get(
                    "document", "Hypertension Guideline"
                ),
                "bm25_score": float(scores[index]),
            })

        return results

    def search(self, question):
        semantic_results = self.semantic_search(question)
        bm25_results = self.bm25_search(question)

        if not semantic_results and not bm25_results:
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

        semantic_by_text = {
            chunk["text"]: chunk
            for chunk in semantic_results
        }

        max_bm25 = max(
            (chunk.get("bm25_score", 0.0) for chunk in bm25_results),
            default=0.0,
        )

        combined = {}

        for chunk in semantic_results + bm25_results:
            text = chunk["text"]

            if text not in combined:
                combined[text] = dict(chunk)

            current = combined[text]

            if "distance" in chunk:
                current["distance"] = chunk["distance"]

            if "bm25_score" in chunk:
                current["bm25_score"] = chunk["bm25_score"]

        for text, chunk in combined.items():
            distance = chunk.get("distance")

            semantic_score = (
                max(0.0, min(1.0, 1.0 - distance))
                if distance is not None
                else 0.0
            )

            bm25_score = chunk.get("bm25_score", 0.0)
            normalized_bm25 = (
                bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            )

            chunk["hybrid_score"] = (
                0.6 * semantic_score
                + 0.4 * normalized_bm25
            )

        candidates = sorted(
            combined.values(),
            key=lambda chunk: chunk["hybrid_score"],
            reverse=True,
        )[:settings.TOP_K]

        semantic_distances = [
            chunk["distance"]
            for chunk in semantic_results
            if "distance" in chunk
        ]

        best_distance = min(semantic_distances, default=None)

        high_threshold = getattr(
            settings, "HIGH_CONFIDENCE_DISTANCE", 0.30
        )
        distance_threshold = getattr(
            settings, "DISTANCE_THRESHOLD", 0.70
        )

        if best_distance is None or best_distance > distance_threshold:
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

        confidence = (
            "High"
            if best_distance <= high_threshold
            else "Medium"
        )

        return {
            "status": "success",
            "chunks": candidates,
            "confidence": confidence,
        }

    def rerank(self, question, chunks):
        if not chunks:
            return []

        model = self._load_reranker()

        pairs = [
            (question, chunk["text"])
            for chunk in chunks
        ]

        scores = model.predict(pairs, batch_size=2)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        return chunks[:settings.RERANK_TOP_K]
