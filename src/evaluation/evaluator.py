
"""
Empirical Evaluation Engine

Metrics:
- Precision@K
- Citation Accuracy
- Unsupported Claim Rate
- Average Latency
"""

import re


class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def precision_at_k(self, retrieved_pages, expected_pages, k=None):
        if not expected_pages or not retrieved_pages:
            return None if not expected_pages else 0.0

        retrieved = [str(page) for page in retrieved_pages]
        expected = {str(page) for page in expected_pages}

        if k is None:
            k = len(retrieved)

        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant = sum(
            1 for page in top_k if page in expected
        )

        return round(relevant / k, 3)

    def citation_accuracy(self, response):
        if response.status != "success":
            return 0.0

        retrieved_pages = {
            str(chunk.get("page", "Unknown"))
            for chunk in (response.chunks or [])
            if chunk.get("page", "Unknown") != "Unknown"
        }

        answer_pages = set(
            re.findall(
                r"\[Page\s+(\d+)\]",
                response.answer or "",
                re.IGNORECASE,
            )
        )

        if not answer_pages:
            for citation in response.citations or []:
                if isinstance(citation, dict):
                    page = citation.get("page")
                    if page is not None and str(page) != "Unknown":
                        answer_pages.add(str(page))
                elif isinstance(citation, str):
                    match = re.search(r"\bPage\s+(\d+)\b", citation, re.IGNORECASE)
                    if match:
                        answer_pages.add(match.group(1))

        if not answer_pages:
            return 0.0

        correct = len(answer_pages.intersection(retrieved_pages))
        return round(correct / len(answer_pages), 3)

    def unsupported_claim_rate(self, response):
        if response.status != "success":
            return 0.0

        answer = (response.answer or "").lower()

        refusal_phrases = [
            "couldn't find sufficient evidence",
            "cannot answer reliably",
            "unable to answer from the available evidence",
        ]

        if any(phrase in answer for phrase in refusal_phrases):
            return 0.0

        return round(1.0 - self.citation_accuracy(response), 3)

    def evaluate_case(self, case):
        try:
            response = self.pipeline.ask(case["question"])

            retrieved_pages = getattr(
                response, "retrieved_pages", []
            ) or []

            chunks = getattr(response, "chunks", []) or []

            # Use retrieved chunks when available, preserving their ranking.
            ranked_pages = [
                str(chunk.get("page"))
                for chunk in chunks
                if chunk.get("page") is not None
                and str(chunk.get("page")) != "Unknown"
            ]

            if not ranked_pages:
                ranked_pages = [str(page) for page in retrieved_pages]

            expected_pages = case.get("expected_pages", [])
            k = case.get("k", len(ranked_pages))

            precision = self.precision_at_k(
                ranked_pages,
                expected_pages,
                k=k,
            )

            citation = self.citation_accuracy(response)
            unsupported = self.unsupported_claim_rate(response)

            return {
                "id": case["id"],
                "question": case["question"],
                "status": response.status,
                "risk": response.risk,
                "confidence": getattr(response, "confidence", "Low"),
                "latency": response.latency,
                "retrieved_chunks": len(chunks),
                "retrieved_pages": retrieved_pages,
                "precision_at_k": precision,
                "citation_accuracy": citation,
                "unsupported_claim_rate": unsupported,
                "error": "",
            }

        except Exception as exc:
            error_message = str(exc)

            if "429" in error_message or "quota" in error_message.lower():
                status = "quota_exceeded"
                error_message = "Gemini API quota exceeded."
            else:
                status = "error"

            return {
                "id": case["id"],
                "question": case["question"],
                "status": status,
                "risk": "unknown",
                "confidence": "Unknown",
                "latency": 0.0,
                "retrieved_chunks": 0,
                "retrieved_pages": [],
                "precision_at_k": None,
                "citation_accuracy": None,
                "unsupported_claim_rate": None,
                "error": error_message,
            }

    def run(self, test_cases):
        results = []

        for case in test_cases:
            result = self.evaluate_case(case)
            results.append(result)

            if result["status"] == "quota_exceeded":
                break

        return results
