
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config.settings import settings


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH
        )

        collections = {
            collection.name
            for collection in self.client.list_collections()
        }

        if settings.COLLECTION_NAME not in collections:
            from src.build_pipeline import main

            main()

        self.collection = self.client.get_collection(
            name=settings.COLLECTION_NAME
        )

        self.encoder = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []

        self.bm25 = None

        if self.documents:
            tokenized_documents = [
                self._tokenize(document)
                for document in self.documents
            ]

            self.bm25 = BM25Okapi(tokenized_documents)

        self.reranker = None

    @staticmethod
    def _tokenize(text):
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            (text or "").lower(),
        )

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
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        chunks = []
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        for text, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0],
        ):
            metadata = metadata or {}

            chunks.append({
                "text": text,
                "page": metadata.get("page", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "document": metadata.get(
                    "document",
                    "Hypertension Guideline",
                ),
                "distance": float(distance),
            })

        return chunks

    def bm25_search(self, question, top_k=None):
        top_k = top_k or settings.TOP_K

        if self.bm25 is None:
            return []

        tokens = self._tokenize(question)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in indices:
            score = float(scores[index])

            if score <= 0:
                continue

            metadata = self.metadatas[index] or {}

            results.append({
                "text": self.documents[index],
                "page": metadata.get("page", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "document": metadata.get(
                    "document",
                    "Hypertension Guideline",
                ),
                "bm25_score": score,
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

        combined = {}

        def add_ranked_results(results, source):
            for rank, chunk in enumerate(results, start=1):
                text = chunk["text"]

                if text not in combined:
                    combined[text] = dict(chunk)
                    combined[text]["rrf_score"] = 0.0

                current = combined[text]
                current["rrf_score"] += 1.0 / (60 + rank)

                if source == "semantic":
                    current["distance"] = chunk.get("distance")

                if source == "bm25":
                    current["bm25_score"] = chunk.get(
                        "bm25_score", 0.0
                    )

                for key in ("page", "section", "document"):
                    if current.get(key) in (None, "Unknown", ""):
                        current[key] = chunk.get(key, current.get(key))

        add_ranked_results(semantic_results, "semantic")
        add_ranked_results(bm25_results, "bm25")

        candidates = sorted(
            combined.values(),
            key=lambda chunk: chunk["rrf_score"],
            reverse=True,
        )[:settings.TOP_K]

        semantic_distances = [
            chunk["distance"]
            for chunk in semantic_results
            if "distance" in chunk
        ]

        best_distance = min(semantic_distances, default=None)

        high_threshold = getattr(
            settings,
            "HIGH_CONFIDENCE_DISTANCE",
            0.30,
        )

        distance_threshold = getattr(
            settings,
            "DISTANCE_THRESHOLD",
            0.70,
        )

        has_keyword_match = any(
            chunk.get("bm25_score", 0.0) > 0
            for chunk in candidates
        )

        if (
            best_distance is not None
            and best_distance > distance_threshold
            and not has_keyword_match
        ):
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

        if best_distance is None and not has_keyword_match:
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

        if best_distance is None:
            confidence = "Medium"
        elif best_distance <= high_threshold:
            confidence = "High"
        else:
            confidence = "Medium"

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

        scores = model.predict(
            pairs,
            batch_size=2,
        )

        reranked = []

        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        return reranked[:settings.RERANK_TOP_K]
