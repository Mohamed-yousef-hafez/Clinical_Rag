import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config.settings import settings


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.reranker = None

        # Build the index before loading the query encoder to avoid loading
        # two embedding models into memory at the same time on first startup.
        self.collection = self._get_or_build_collection()
        try:
            self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as exc:
            self.encoder = None
            print(
                "Embedding model could not be loaded; falling back to "
                f"keyword retrieval when possible: {type(exc).__name__}: {exc}"
            )

        data = self.collection.get(include=["documents", "metadatas"])
        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []

        self.bm25 = None
        if self.documents:
            tokenized_documents = [self._tokenize(text) for text in self.documents]
            self.bm25 = BM25Okapi(tokenized_documents)

    def _get_or_build_collection(self):
        collection_names = {
            item.name for item in self.client.list_collections()
        }

        if settings.COLLECTION_NAME not in collection_names:
            from src.build_pipeline import main
            main()

        try:
            collection = self.client.get_collection(name=settings.COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "The ChromaDB collection could not be opened after building "
                "the knowledge base. Check the PDF and data directory."
            ) from exc

        if collection.count() == 0:
            from src.build_pipeline import main
            main()
            collection = self.client.get_collection(name=settings.COLLECTION_NAME)

        if collection.count() == 0:
            raise RuntimeError(
                "The knowledge base is empty. The guideline PDF may contain "
                "no extractable text."
            )

        return collection

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\b[a-zA-Z0-9]+\b", (text or "").lower())

    def _load_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        return self.reranker

    def semantic_search(self, question, top_k=None):
        top_k = top_k or settings.TOP_K
        if not self.documents or self.encoder is None:
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

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        chunks = []

        for text, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            metadata = metadata or {}
            chunks.append({
                "text": text,
                "page": metadata.get("page", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "document": metadata.get("document", "Hypertension Guideline"),
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
                "document": metadata.get("document", "Hypertension Guideline"),
                "bm25_score": score,
            })
        return results

    def search(self, question):
        try:
            semantic_results = self.semantic_search(question)
        except Exception as exc:
            print(
                "Semantic search failed; trying keyword retrieval: "
                f"{type(exc).__name__}: {exc}"
            )
            semantic_results = []

        bm25_results = self.bm25_search(question)

        if not semantic_results and not bm25_results:
            return {"status": "out_of_scope", "chunks": [], "confidence": "Low"}

        combined = {}

        def add_results(results, source):
            for chunk in results:
                key = chunk["text"]
                if key not in combined:
                    combined[key] = dict(chunk)
                current = combined[key]
                if source == "semantic":
                    current["distance"] = chunk.get("distance")
                else:
                    current["bm25_score"] = chunk.get("bm25_score", 0.0)

                for field in ("page", "section", "document"):
                    if current.get(field) in (None, "", "Unknown"):
                        current[field] = chunk.get(field, current.get(field, "Unknown"))

        add_results(semantic_results, "semantic")
        add_results(bm25_results, "bm25")

        max_bm25 = max(
            (chunk.get("bm25_score", 0.0) for chunk in combined.values()),
            default=0.0,
        )

        for chunk in combined.values():
            distance = chunk.get("distance")
            semantic_score = (
                max(0.0, min(1.0, 1.0 - float(distance)))
                if distance is not None else 0.0
            )
            bm25_score = chunk.get("bm25_score", 0.0)
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            chunk["semantic_score"] = semantic_score
            chunk["normalized_bm25_score"] = normalized_bm25
            chunk["hybrid_score"] = 0.65 * semantic_score + 0.35 * normalized_bm25

        candidates = sorted(
            combined.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )[:settings.TOP_K]

        best_distance = min(
            (item["distance"] for item in semantic_results if "distance" in item),
            default=None,
        )
        has_keyword_match = max_bm25 > 0

        if best_distance is not None and best_distance > settings.DISTANCE_THRESHOLD and not has_keyword_match:
            return {"status": "out_of_scope", "chunks": [], "confidence": "Low"}

        if best_distance is None and not has_keyword_match:
            return {"status": "out_of_scope", "chunks": [], "confidence": "Low"}

        if best_distance is None:
            confidence = "Medium"
        elif best_distance <= settings.HIGH_CONFIDENCE_DISTANCE:
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

        try:
            model = self._load_reranker()
            pairs = [(question, chunk["text"]) for chunk in chunks]
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

        except Exception as exc:
            # Keep the RAG flow usable if the optional cross-encoder fails
            # to download or initialize in the deployment environment.
            print(
                "Reranker unavailable; using hybrid retrieval ranking: "
                f"{type(exc).__name__}: {exc}"
            )
            fallback = []
            for chunk in chunks:
                item = dict(chunk)
                item["rerank_score"] = float(item.get("hybrid_score", 0.0))
                fallback.append(item)
            fallback.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )
            return fallback[:settings.RERANK_TOP_K]
