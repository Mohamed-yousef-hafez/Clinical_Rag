import time
import traceback

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
        start_time = time.perf_counter()
        question = (question or "").strip()

        if not question:
            return RAGResponse(
                question="",
                answer="Please enter a question.",
                risk="unsafe",
                status="blocked",
                latency=round(time.perf_counter() - start_time, 2),
            )

        risk = self.classifier.classify(question)
        if risk != "safe":
            return RAGResponse(
                question=question,
                answer=self.refusal.respond(risk),
                risk=risk,
                status="blocked",
                latency=round(time.perf_counter() - start_time, 2),
            )

        try:
            retrieval = self.retriever.search(question)
            if not retrieval or retrieval.get("status") != "success":
                return RAGResponse(
                    question=question,
                    answer=self.refusal.respond("out_of_scope"),
                    risk="out_of_scope",
                    status="out_of_scope",
                    confidence="Low",
                    latency=round(time.perf_counter() - start_time, 2),
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
                    latency=round(time.perf_counter() - start_time, 2),
                )

            answer = self.generator.generate(question, chunks)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Generator returned an empty response.")

            citations = []
            retrieved_pages = []
            for chunk in chunks:
                page = chunk.get("page", "Unknown")
                if page != "Unknown" and page not in retrieved_pages:
                    retrieved_pages.append(page)
                citations.append({
                    "page": page,
                    "document": chunk.get("document", "Hypertension Guideline"),
                    "section": chunk.get("section", "Unknown"),
                    "text": chunk.get("text", ""),
                })

            return RAGResponse(
                question=question,
                answer=answer,
                chunks=chunks,
                citations=citations,
                retrieved_pages=retrieved_pages,
                latency=round(time.perf_counter() - start_time, 2),
                risk="safe",
                status="success",
                confidence=retrieval.get("confidence", "Medium"),
                retrieval_type="Hybrid",
                evidence_count=len(chunks),
            )

        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print(f"RAG pipeline error: {detail}")
            traceback.print_exc()
            return RAGResponse(
                question=question,
                answer="The request could not be completed. Please check the technical details and try again.",
                latency=round(time.perf_counter() - start_time, 2),
                risk="safe",
                status="generation_error",
                confidence="Low",
                error_detail=detail,
            )
