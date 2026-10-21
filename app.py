
import streamlit as st

from src.ui.home import render_home
from src.ui.ask import render_ask
from src.ui.evidence import render_evidence
from src.ui.safety import render_safety
from src.ui.evaluation import render_evaluation
from src.ui.demo import render_demo
from src.ui.results import render_results

st.set_page_config(
    page_title="Clinical Evidence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Clinical Evidence Assistant")
st.sidebar.markdown("---")

pages = {
    "Home": render_home,
    "Ask AI": render_ask,
    "Evidence": render_evidence,
    "Safety": render_safety,
    "Evaluation": render_evaluation,
    "Live Demo": render_demo,
    "Results": render_results,
}

page = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.success("RAG + Hybrid Search + ChromaDB")
st.sidebar.caption("Hypertension guideline only")

try:
    pages[page]()

except Exception as exc:
    st.error(f"An error occurred while loading {page}.")
    st.exception(exc)
