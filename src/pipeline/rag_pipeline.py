import time

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.safety.classifier import RiskClassifier
from src.safety.refusal import SafeRefusal
from src.models.response import RAGResponse


class RAGPipeline:

    def __init__(self):

        print(
            "Initializing Clinical RAG Pipeline..."
        )

        self.retriever = Retriever()
        print("Retriever Loaded")

        self.generator = Generator()
        print("Generator Loaded")

        self.classifier = RiskClassifier()
        print("RiskClassifier Loaded")

        self.refusal = SafeRefusal()
        print("SafeRefusal Loaded")

        print("Clinical RAG Pipeline Ready")

    def ask(self, question):

        start_time = time.time()

        risk = self.classifier.classify(
            question
        )

        if risk != "safe":

            return RAGResponse(
                question=question,
                answer=self.refusal.respond(risk),
                risk=risk,
                status="blocked"
            )

        retrieval = self.retriever.search(
            question
        )

        if retrieval["status"] != "success":

            return RAGResponse(
                question=question,
                answer=self.refusal.respond(
                    "out_of_scope"
                ),
                risk="out_of_scope",
                status="out_of_scope"
            )

        chunks = retrieval.get(
            "chunks",
            []
        )

        chunks = self.retriever.rerank(
            question,
            chunks
        )

        if not chunks:

            return RAGResponse(
                question=question,
                answer=self.refusal.respond(
                    "out_of_scope"
                ),
                risk="out_of_scope",
                status="out_of_scope"
            )

        answer = self.generator.generate(
            question,
            chunks
        )

        citations = []
        pages = []

        for chunk in chunks:

            page = chunk.get(
                "page",
                "Unknown"
            )

            citation = f"Page {page}"

            if citation not in citations:
                citations.append(citation)

            if page not in pages:
                pages.append(page)

        latency = round(
            time.time() - start_time,
            2
        )

        return RAGResponse(
            question=question,
            answer=answer,
            chunks=chunks,
            citations=citations,
            retrieved_pages=pages,
            latency=latency,
            risk="safe",
            status="success",
            confidence=retrieval.get(
                "confidence",
                "Medium"
            )
        )