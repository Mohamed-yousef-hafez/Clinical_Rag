import streamlit as st


def render_about():

    st.title("ℹ️ About The Project")

    st.markdown("""
##  Clinical Evidence Assistant

Clinical Evidence Assistant is a **Retrieval-Augmented Generation (RAG)**
system designed to answer clinical questions about **Hypertension**
using the uploaded clinical guideline.

The system retrieves relevant evidence before generating
grounded answers with citations.
""")

    st.divider()

    st.subheader(" Clinical Scope")

    st.markdown("""
### Supported Clinical Topic

- **Adult Hypertension Management**

### Guideline Source

- WHO

The system answers only from the uploaded
hypertension guideline.
""")

    st.divider()

    st.subheader(" System Components")

    st.markdown("""
- PDF Loading
- Text Cleaning
- LangChain Recursive Text Splitting
- BGE Embeddings
- ChromaDB Vector Store
- Semantic Search
- BM25 Keyword Search
- Hybrid Retrieval
- Cross-Encoder Re-ranking
- Prompt Engineering
- Gemini LLM
- Safety & Guardrails
- Evaluation
""")

    st.divider()

    st.subheader("📊 Evaluation Metrics")

    st.markdown("""
- Precision@K
- Citation Accuracy
- Unsupported Claim Rate
- Average Latency
- Safe Refusal Rate
""")

    st.divider()

    st.subheader(" Future Roadmap")

    st.markdown("""
- Additional Clinical Guidelines
- Hospital Deployment
- Multilingual Support
""")

    st.divider()

    st.subheader("Hackathon Requirements Covered")

    st.success("✅ Intelligent Ingestion")
    st.success("✅ Semantic Retrieval")
    st.success("✅ Hybrid Search (BM25 + Vector)")
    st.success("✅ Cross-Encoder Re-ranking")
    st.success("✅ Vector Database")
    st.success("✅ Prompt Engineering")
    st.success("✅ LLM Integration")
    st.success("✅ Grounded Generation")
    st.success("✅ Citation")
    st.success("✅ Safety & Guardrails")
    st.success("✅ Evaluation Dashboard")
    st.success("✅ Live Demonstration")