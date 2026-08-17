import streamlit as st


def render_home():

    st.title(" Clinical Evidence Assistant")

    st.markdown(
        """
## Evidence-Based Clinical Question Answering using Retrieval-Augmented Generation (RAG)

This system helps healthcare professionals answer clinical questions
using only trusted hypertension guidelines.

Instead of relying on the LLM's internal knowledge,
the system retrieves the most relevant evidence from the uploaded guideline
before generating a grounded answer with citations.
"""
    )

    st.divider()

    st.subheader(" Project Objectives")

    st.success("Provide evidence-based clinical answers")

    st.success("Reduce hallucinations using Retrieval-Augmented Generation")

    st.success("Support transparent answers with citations")

    st.success("Protect against prompt injection and unsafe requests")

    st.success("Evaluate system performance using empirical metrics")

    st.divider()

    st.subheader(" Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info(" Intelligent PDF Ingestion")

        st.info(" Intelligent Text Chunking")

        st.info(" BGE Semantic Embeddings")

        st.info(" Chroma Vector Database")

        st.info(" Semantic Retrieval")

    with col2:

        st.info(" Gemini LLM")

        st.info(" Prompt Engineering")

        st.info(" Grounded Generation")

        st.info(" Safety Guardrails")

        st.info(" Evaluation Dashboard")

    st.divider()

    st.subheader(" Technology Stack")

    st.markdown("""
- Python
- Streamlit
- ChromaDB
- Sentence Transformers (BGE)
- Google Gemini
- PyMuPDF
- Retrieval-Augmented Generation (RAG)
""")

    st.divider()

    st.subheader(" Hackathon Coverage")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.checkbox("LLMs", value=True, disabled=True)
        st.checkbox("RAG", value=True, disabled=True)
        st.checkbox("Vector Database", value=True, disabled=True)
        st.checkbox("Generative AI", value=True, disabled=True)

    with c2:
        st.checkbox("Prompt Engineering", value=True, disabled=True)
        st.checkbox("Semantic Retrieval", value=True, disabled=True)
        st.checkbox("Grounded Generation", value=True, disabled=True)

    with c3:
        st.checkbox("Safety", value=True, disabled=True)
        st.checkbox("Evaluation", value=True, disabled=True)
        st.checkbox("System Design", value=True, disabled=True)

    st.success("✅ Ready for Live Clinical Demonstration")