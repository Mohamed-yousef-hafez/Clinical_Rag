"""
Evaluation Metrics
"""


class RetrievalMetrics:

    @staticmethod
    def precision_at_k(
        retrieved_pages,
        expected_pages
    ):

        if not retrieved_pages:

            return 0.0

        if not expected_pages:

            return None

        retrieved = set(
            str(page)
            for page in retrieved_pages
        )

        expected = set(
            str(page)
            for page in expected_pages
        )

        correct = len(
            retrieved.intersection(
                expected
            )
        )

        return round(
            correct / len(retrieved),
            3
        )


class CitationMetrics:

    @staticmethod
    def citation_accuracy(
        predicted,
        expected
    ):

        if not predicted:

            return 0.0

        predicted = set(
            str(page)
            for page in predicted
        )

        expected = set(
            str(page)
            for page in expected
        )

        if not expected:

            return None

        correct = len(
            predicted.intersection(
                expected
            )
        )

        return round(
            correct / len(predicted),
            3
        )


class FaithfulnessMetrics:

    @staticmethod
    def faithfulness(
        answer,
        context
    ):

        if not answer or not context:

            return 0.0

        answer_words = set(
            answer.lower().split()
        )

        context_words = set(
            context.lower().split()
        )

        if not answer_words:

            return 0.0

        supported_words = len(
            answer_words.intersection(
                context_words
            )
        )

        return round(
            supported_words / len(answer_words),
            3
        )


class UnsupportedClaimMetrics:

    @staticmethod
    def unsupported_rate(
        total_answers,
        unsupported_answers
    ):

        if total_answers == 0:

            return 0.0

        return round(
            unsupported_answers / total_answers,
            3
        )