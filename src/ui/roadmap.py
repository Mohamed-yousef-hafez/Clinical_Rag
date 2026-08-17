import streamlit as st


def render_roadmap():

    st.title("🛣 Project Roadmap")

    roadmap = [

        "✅ Current Version — Hypertension Evidence Assistant",

        "⬜ Improve Hybrid Search and Retrieval Quality",

        "⬜ Improve Cross-Encoder Re-ranking",

        "⬜ Add More Evaluation Test Cases",

        "⬜ Medical Knowledge Graph",

        "⬜ FHIR Integration",

        "⬜ Hospital Deployment",

        "⬜ Multilingual Clinical Support"

    ]

    for step in roadmap:

        st.write(step)

    st.info(
        """
Future work focuses on improving retrieval quality,
evaluation, interoperability, scalability and deployment
while keeping the current system focused on hypertension.
"""
    )