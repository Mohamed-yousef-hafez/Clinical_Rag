
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

    def precision_at_k(self, retrieved_pages, expected_pages):
        if not retrieved_pages:
            return 0.0

        if not expected_pages:
            return None

        retrieved = {str(page) for page in retrieved_pages}
        expected = {str(page) for page in expected_pages}

        relevant = len(retrieved.intersection(expected))

        return round(relevant / len(retrieved), 3)

    def citation_accuracy(self, response):
        if response.status != "success":
            return 0.0

        retrieved_pages = {
            str(chunk.get("page", "Unknown"))
            for chunk in response.chunks
            if chunk.get("page", "Unknown") != "Unknown"
        }

        ui_pages = set()

        for citation in response.citations or []:
            if isinstance(citation, dict):
                page = citation.get("page")

                if page is not None and str(page) != "Unknown":
                    ui_pages.add(str(page))

            elif isinstance(citation, str):
                match = re.search(r"\d+", citation)

                if match:
                    ui_pages.add(match.group(0))

        answer_pages = set()

        patterns = [
            r"\[Page\s*(\d+)\]",
            r"\(Page\s*(\d+)\)",
            r"\bPage\s*(\d+)\b",
            r"\bpage\s*(\d+)\b",
        ]

        for pattern in patterns:
            answer_pages.update(
                re.findall(pattern, response.answer, re.IGNORECASE)
            )

        cited_pages = answer_pages or ui_pages

        if not cited_pages:
            return 0.0

        correct = len(cited_pages.intersection(retrieved_pages))

        return round(correct / len(cited_pages), 3)

    def unsupported_claim_rate(self, response):
        if response.status != "success":
            return 0.0

        refusal_phrases = [
            "couldn't find sufficient evidence",
            "cannot answer reliably",
            "unable to answer from the available evidence",
        ]

        if any(
            phrase in response.answer.lower()
            for phrase in refusal_phrases
        ):
            return 0.0

        citation_score = self.citation_accuracy(response)

        if citation_score > 0:
            return round(1 - citation_score, 3)

        return 1.0

    def evaluate_case(self, case):
        try:
            start_pages = []
            response = self.pipeline.ask(case["question"])

            retrieved_pages = getattr(
                response,
                "retrieved_pages",
                [],
            ) or []

            chunks = getattr(response, "chunks", []) or []

            precision = self.precision_at_k(
                retrieved_pages,
                case.get("expected_pages", []),
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

            if (
                "429" in error_message
                or "quota" in error_message.lower()
            ):
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
