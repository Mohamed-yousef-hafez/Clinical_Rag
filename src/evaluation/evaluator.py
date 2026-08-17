
"""
Empirical Evaluation Engine

 Metrics

- Precision@K
- Citation Accuracy
- Unsupported Claim Rate
- Average Latency
"""

import re


class Evaluator:

    def __init__(self, pipeline):

        self.pipeline = pipeline

    # =====================================================
    # Precision@K
    # =====================================================

    def precision_at_k(
        self,
        retrieved_pages,
        expected_pages
    ):

        if not retrieved_pages:

            return 0.0

        if not expected_pages:

            return None

        retrieved = set(
            str(p)
            for p in retrieved_pages
        )

        expected = set(
            str(p)
            for p in expected_pages
        )

        relevant = len(
            retrieved.intersection(
                expected
            )
        )

        return round(
            relevant / len(retrieved),
            3
        )

    # =====================================================
    # Citation Accuracy
    # =====================================================

    def citation_accuracy(
        self,
        response
    ):

        if response.status != "success":

            return 0.0

        # ---------------------------------------------
        # Pages actually retrieved
        # ---------------------------------------------

        retrieved_pages = set(
            str(
                chunk.get(
                    "page",
                    "Unknown"
                )
            )
            for chunk in response.chunks
        )

        # ---------------------------------------------
        # Pages shown in UI
        # ---------------------------------------------

        ui_pages = set()

        for citation in response.citations:

            match = re.search(
                r"(\d+)",
                citation
            )

            if match:

                ui_pages.add(
                    match.group(1)
                )

        # ---------------------------------------------
        # Pages written by Gemini
        # ---------------------------------------------

        answer_pages = set()

        patterns = [

            r"\[Page\s*(\d+)\]",

            r"\(Page\s*(\d+)\)",

            r"Page\s*(\d+)",

            r"page\s*(\d+)"

        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                response.answer,
                re.IGNORECASE
            )

            answer_pages.update(
                matches
            )

        # ---------------------------------------------
        # Choose cited pages
        # ---------------------------------------------

        cited_pages = (
            answer_pages
            or ui_pages
        )

        if not cited_pages:

            return 0.0

        correct = len(
            cited_pages.intersection(
                retrieved_pages
            )
        )

        return round(
            correct / len(cited_pages),
            3
        )

    # =====================================================
    # Unsupported Claim Rate
    # =====================================================

    def unsupported_claim_rate(
        self,
        response
    ):

        if response.status != "success":

            return 0.0

        refusal = (
            "couldn't find sufficient evidence"
        )

        if (
            refusal.lower()
            in response.answer.lower()
        ):

            return 0.0

        citation_score = (
            self.citation_accuracy(
                response
            )
        )

        if citation_score > 0:

            return round(
                1 - citation_score,
                3
            )

        return 1.0

    # =====================================================
    # Evaluate One Case
    # =====================================================

    def evaluate_case(
        self,
        case
    ):

        try:

            response = self.pipeline.ask(
                case["question"]
            )

            precision = (
                self.precision_at_k(
                    response.retrieved_pages,
                    case.get(
                        "expected_pages",
                        []
                    )
                )
            )

            citation = (
                self.citation_accuracy(
                    response
                )
            )

            unsupported = (
                self.unsupported_claim_rate(
                    response
                )
            )

            return {

                "id":
                    case["id"],

                "question":
                    case["question"],

                "status":
                    response.status,

                "risk":
                    response.risk,

                "confidence":
                    getattr(
                        response,
                        "confidence",
                        "Low"
                    ),

                "latency":
                    response.latency,

                "retrieved_chunks":
                    len(response.chunks),

                "retrieved_pages":
                    response.retrieved_pages,

                "precision_at_k":
                    precision,

                "citation_accuracy":
                    citation,

                "unsupported_claim_rate":
                    unsupported

            }

        except Exception as e:

            error_message = str(e)

            # ---------------------------------------------
            # Gemini Quota
            # ---------------------------------------------

            if (
                "429" in error_message
                or "quota"
                in error_message.lower()
            ):

                return {

                    "id":
                        case["id"],

                    "question":
                        case["question"],

                    "status":
                        "quota_exceeded",

                    "risk":
                        "unknown",

                    "confidence":
                        "Unknown",

                    "latency":
                        0.0,

                    "retrieved_chunks":
                        0,

                    "retrieved_pages":
                        [],

                    "precision_at_k":
                        None,

                    "citation_accuracy":
                        None,

                    "unsupported_claim_rate":
                        None,

                    "error":
                        "Gemini API quota exceeded."

                }

            # ---------------------------------------------
            # Other Errors
            # ---------------------------------------------

            return {

                "id":
                    case["id"],

                "question":
                    case["question"],

                "status":
                    "error",

                "risk":
                    "unknown",

                "confidence":
                    "Unknown",

                "latency":
                    0.0,

                "retrieved_chunks":
                    0,

                "retrieved_pages":
                    [],

                "precision_at_k":
                    None,

                "citation_accuracy":
                    None,

                "unsupported_claim_rate":
                    None,

                "error":
                    error_message

            }

    # =====================================================
    # Run All Benchmark Cases
    # =====================================================

    def run(
        self,
        test_cases
    ):

        results = []

        for case in test_cases:

            result = self.evaluate_case(
                case
            )

            results.append(
                result
            )

            if (
                result["status"]
                == "quota_exceeded"
            ):

                break

        return results

