import streamlit as st


def render_rubric():

    st.title("🏆 Judging Criteria")

    st.markdown(
        """
##  Rubric Mapping

The project maps each judging criterion to a
concrete engineering implementation for the
**Hypertension Clinical Evidence Assistant**.
"""
    )

    criteria = [

        (
            "Problem Quality",
            "Evidence-based hypertension question answering",
            "Focused hypertension scope + clinical guideline knowledge base"
        ),

        (
            "Grounding & Citation",
            "Answers must be supported by retrieved evidence",
            "RAG + page-level citations + grounded generation"
        ),

        (
            "System Architecture",
            "Modular end-to-end AI system",
            "PDF Ingestion → Chunking → Embeddings → Hybrid Retrieval → LLM"
        ),

        (
            "Evaluation Depth",
            "Empirical measurement of system performance",
            "Precision@K + Citation Accuracy + Unsupported Claim Rate + Latency"
        ),

        (
            "Safety",
            "Prevent unsupported or malicious requests",
            "Prompt Injection Detection + Out-of-Scope Refusal"
        ),

        (
            "UX / Demo",
            "Clear interactive demonstration",
            "Streamlit + Hypertension + Safety Demo Cases"
        )

    ]

    for title, requirement, implementation in criteria:

        with st.expander(
            f" {title}"
        ):

            st.write(
                f"**Requirement:** {requirement}"
            )

            st.success(
                f"**Implementation:** {implementation}"
            )

    st.divider()

    st.header("Engineering Architecture")

    st.code(
        """
Clinical Hypertension Guideline
            ↓
     PDF Ingestion
            ↓
  Recursive Text Chunking
            ↓
      BGE Embeddings
            ↓
       ChromaDB
            ↓
 ┌──────────┴──────────┐
 │                     │
Semantic Search       BM25
 │                     │
 └──────────┬──────────┘
            ↓
    Hybrid Retrieval
            ↓
   Cross-Encoder Ranking
            ↓
   Retrieved Evidence
            ↓
    Prompt Engineering
            ↓
       Gemini LLM
            ↓
 Grounded Answer + Citation
            ↓
      Evaluation + Safety
        """
    )