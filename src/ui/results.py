import streamlit as st

from src.config.settings import settings


def render_results():

    st.title("📈 Final Results")

    st.markdown(
        """
## Clinical Evidence Assistant — Hypertension

The current system provides an end-to-end RAG pipeline
for evidence-based hypertension question answering.
"""
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Embedding Model",
            getattr(
                settings,
                "EMBEDDING_MODEL",
                "BAAI/bge-small-en-v1.5"
            )
        )

        st.metric(
            "Vector Store",
            "ChromaDB"
        )

        st.metric(
            "LLM",
            "Gemini"
        )

        st.metric(
            "Top K",
            getattr(
                settings,
                "TOP_K",
                5
            )
        )

    with col2:

        st.metric(
            "Chunk Size",
            getattr(
                settings,
                "CHUNK_SIZE",
                500
            )
        )

        st.metric(
            "Chunk Overlap",
            getattr(
                settings,
                "CHUNK_OVERLAP",
                50
            )
        )

        st.metric(
            "Similarity Threshold",
            getattr(
                settings,
                "SIMILARITY_THRESHOLD",
                0.70
            )
        )

        st.metric(
            "Safety",
            "Enabled"
        )

    st.divider()

    st.subheader(
        " Implemented Requirements"
    )

    st.success(
        "✅ Intelligent Ingestion"
    )

    st.success(
        "✅ LangChain Recursive Text Splitting"
    )

    st.success(
        "✅ BGE Semantic Embeddings"
    )

    st.success(
        "✅ ChromaDB Vector Database"
    )

    st.success(
        "✅ BM25 + Semantic Hybrid Search"
    )

    st.success(
        "✅ Cross-Encoder Re-ranking"
    )

    st.success(
        "✅ Prompt Engineering"
    )

    st.success(
        "✅ Gemini LLM Integration"
    )

    st.success(
        "✅ Grounded Generation"
    )

    st.success(
        "✅ Page-Level Citation"
    )

    st.success(
        "✅ Safety & Guardrails"
    )

    st.success(
        "✅ Empirical Evaluation"
    )

    st.success(
        "✅ Live Demonstration"
    )