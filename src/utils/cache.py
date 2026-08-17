import streamlit as st

from src.pipeline.rag_pipeline import RAGPipeline


@st.cache_resource(
    show_spinner="Loading Clinical RAG Pipeline..."
)
def load_pipeline():

    return RAGPipeline()