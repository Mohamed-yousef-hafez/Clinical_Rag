import streamlit as st


def render_knowledge_base():

    st.title(" Knowledge Base")

    st.markdown(
        """
### Hypertension Guideline Knowledge Base

The system uses the uploaded **hypertension clinical guideline**
as the single knowledge source for retrieval and grounded answers.
"""
    )

    st.divider()

    st.info(
        "The current knowledge base is built from the "
        "hypertension guideline used by the project."
    )

    st.subheader(" Knowledge Base Components")

    col1, col2 = st.columns(2)

    with col1:

        st.success("✅ PDF Loader")

        st.success("✅ Text Cleaning")

        st.success("✅ Recursive Text Chunking")

        st.success("✅ BGE Embeddings")

    with col2:

        st.success("✅ ChromaDB")

        st.success("✅ Semantic Retrieval")

        st.success("✅ BM25 Hybrid Search")

        st.success("✅ Cross-Encoder Re-ranking")

    st.divider()

    st.subheader("📄 Source Document")

    st.write(
        "Clinical Hypertension Guideline"
    )

    st.caption(
        "Only the uploaded hypertension guideline "
        "is used as the project's knowledge source."
    )

    st.divider()

    st.subheader(" Knowledge Flow")

    st.code(
        """
Hypertension Guideline PDF
            ↓
       PDF Loader
            ↓
      Text Cleaning
            ↓
Recursive Text Splitting
            ↓
     BGE Embeddings
            ↓
        ChromaDB
            ↓
 Semantic + BM25 Retrieval
            ↓
   Cross-Encoder Ranking
        """
    )