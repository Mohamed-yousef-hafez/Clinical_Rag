
import time

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.safety.classifier import RiskClassifier
from src.safety.refusal import SafeRefusal
from src.models.response import RAGResponse


class RAGPipeline:
    def __init__(self):
        print("Initializing Clinical RAG Pipeline...")

        self.retriever = Retriever()
        self.generator = Generator()
        self.classifier = RiskClassifier()
        self.refusal = SafeRefusal()

        print("Clinical RAG Pipeline Ready")

    def ask(self, question):
        start_time = time.time()
        question = (question or "").strip()

        if not question:
            return RAGResponse(
                question="",
                answer="Please enter a question.",
                risk="unsafe",
                status="blocked",
                latency=round(time.time() - start_time, 2),
            )

        risk = self.classifier.classify(question)

        if risk != "safe":
            return RAGResponse(
                question=question,
                answer=self.refusal.respond(risk),
                risk=risk,
                status="blocked",
                latency=round(time.time() - start_time, 2),
            )

        retrieval = self.retriever.search(question)

        if retrieval.get("status") != "success":
            return RAGResponse(
                question=question,
                answer=self.refusal.respond("out_of_scope"),
                risk="out_of_scope",
                status="out_of_scope",
                confidence="Low",
                latency=round(time.time() - start_time, 2),
            )

        chunks = self.retriever.rerank(
            question,
            retrieval.get("chunks", []),
        )

        if not chunks:
            return RAGResponse(
                question=question,
                answer=self.refusal.respond("out_of_scope"),
                risk="out_of_scope",
                status="out_of_scope",
                confidence="Low",
                latency=round(time.time() - start_time, 2),
            )

        try:
            answer = self.generator.generate(question, chunks)

        except RuntimeError as exc:
            return RAGResponse(
                question=question,
                answer=str(exc),
                chunks=chunks,
                citations=[],
                retrieved_pages=[],
                latency=round(time.time() - start_time, 2),
                risk="safe",
                status="generation_error",
                confidence=retrieval.get("confidence", "Medium"),
                evidence_count=len(chunks),
            )

        if not isinstance(answer, str) or not answer.strip():
            return RAGResponse(
                question=question,
                answer="Gemini returned an empty response.",
                chunks=chunks,
                citations=[],
                retrieved_pages=[],
                latency=round(time.time() - start_time, 2),
                risk="safe",
                status="generation_error",
                confidence=retrieval.get("confidence", "Medium"),
                evidence_count=len(chunks),
            )