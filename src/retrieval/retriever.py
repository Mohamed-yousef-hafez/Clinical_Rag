
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

        collection_names = {
            item.name for item in self.client.list_collections()
        }

        if settings.COLLECTION_NAME not in collection_names:
            from src.build_pipeline import main
            main()

        self.collection = self.client.get_collection(
            name=settings.COLLECTION_NAME
        )

        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []

        self.bm25 = None
        if self.documents:
            self.bm25 = BM25Okapi([
                self._tokenize(document)
                for document in self.documents
            ])

        self.reranker = None

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\b[a-z0-9]+\b", (text or "").lower())

    def _load_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
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
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        for text, metadata, distance in zip(
            docs[0], metas[0], distances[0]
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
                    "document", "Hypertension Guideline"
                ),
                "bm25_score": score,
            })

        return results

    def search(self, question):
        semantic = self.semantic_search(question)
        keyword = self.bm25_search(question)

        if not semantic and not keyword:
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

        combined = {}

        for source_results in (semantic, keyword):
            for rank, chunk in enumerate(source_results, start=1):
                text = chunk["text"]

                if text not in combined:
                    combined[text] = dict(chunk)
                    combined[text]["rrf_score"] = 0.0

                combined[text]["rrf_score"] += 1.0 / (60 + rank)

                for key in ("page", "section", "document"):
                    if combined[text].get(key) in (None, "", "Unknown"):
                        combined[text][key] = chunk.get(
                            key, combined[text].get(key)
                        )

                for key in ("distance", "bm25_score"):
                    if key in chunk:
                        combined[text][key] = chunk[key]

        candidates = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )[:max(settings.TOP_K * 2, settings.TOP_K)]

        semantic_distances = [
            item["distance"]
            for item in semantic
            if "distance" in item
        ]
        best_distance = min(semantic_distances, default=None)

        high_threshold = getattr(
            settings, "HIGH_CONFIDENCE_DISTANCE", 0.30
        )
        distance_threshold = getattr(
            settings, "DISTANCE_THRESHOLD", 0.70
        )

        has_keyword_match = any(
            item.get("bm25_score", 0.0) > 0
            for item in candidates
        )

        if best_distance is None and not has_keyword_match:
            return {
                "status": "out_of_scope",
                "chunks": [],
                "confidence": "Low",
            }

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
        pairs = [(question, item["text"]) for item in chunks]
        scores = model.predict(pairs, batch_size=2)

        reranked = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:settings.RERANK_TOP_K]
