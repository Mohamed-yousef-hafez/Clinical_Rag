
import streamlit as st

from src.ui.home import render_home
from src.ui.ask import render_ask
from src.ui.safety import render_safety
from src.ui.evaluation import render_evaluation
from src.ui.demo import render_demo
from src.ui.roadmap import render_roadmap
from src.ui.about import render_about


st.set_page_config(
    page_title="Clinical Evidence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.sidebar.title("🩺 Clinical Evidence Assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💬 Ask AI",
        "🛡 Safety",
        "📊 Evaluation",
        "🚀 Live Demo",
        
    ],
)

st.sidebar.markdown("---")
st.sidebar.success("RAG + LLM + ChromaDB")
st.sidebar.caption("Clinical Evidence Assistant")


try:
    if page == "🏠 Home":
        render_home()

    

    elif page == "💬 Ask AI":
        render_ask()

    elif page == "🛡 Safety":
        render_safety()

    elif page == "📊 Evaluation":
        render_evaluation()

    elif page == "🚀 Live Demo":
        render_demo()

    
except Exception:
    import traceback

    st.error("Application Error")
    st.code(traceback.format_exc())
