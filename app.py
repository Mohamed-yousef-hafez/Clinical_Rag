import streamlit as st

from src.ui.home import render_home
from src.ui.architecture import render_architecture
from src.ui.ask import render_ask
from src.ui.evidence import render_evidence
from src.ui.safety import render_safety
from src.ui.evaluation import render_evaluation
from src.ui.demo import render_demo
from src.ui.results import render_results
from src.ui.roadmap import render_roadmap
from src.ui.about import render_about


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Clinical Evidence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title(
    " Clinical Evidence Assistant"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        " Home",
        " Ask AI",
        " Evidence",
        " Safety",
        " Evaluation",
        " Live Demo",
        " Results",
        " Roadmap",
        " About"
    ]
)


st.sidebar.markdown("---")

st.sidebar.success(
    "RAG + LLM + ChromaDB"
)

st.sidebar.caption(
    "Clinical Evidence Assistant"
)


# =====================================================
# Page Routing
# =====================================================

try:

    if page == " Home":

        render_home()

    

    elif page == "💬 Ask AI":

        render_ask()

    elif page == " Evidence":

        render_evidence()

    elif page == " Safety":

        render_safety()

    elif page == " Evaluation":

        render_evaluation()

    elif page == " Live Demo":

        render_demo()

    elif page == "📈 Results":

        render_results()

    elif page == "🛣 Roadmap":

        render_roadmap()

    elif page == "ℹ️ About":

        render_about()


except Exception as e:

    st.error(
        "❌ Application Error"
    )

    st.exception(e)