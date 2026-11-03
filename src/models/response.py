
from dataclasses import dataclass, field


@dataclass
class RAGResponse:

    question: str

    answer: str

    chunks: list = field(
        default_factory=list
    )

    citations: list = field(
        default_factory=list
    )

    retrieved_pages: list = field(
        default_factory=list
    )

    latency: float = 0.0

    risk: str = "safe"

    status: str = "success"

    # -----------------------------------------
    #  Metadata
    # -----------------------------------------

    confidence: str = "Low"

    retrieval_type: str = "Standard"

    evidence_count: int = 0

