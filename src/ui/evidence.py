

import streamlit as st


def render_evidence():

    st.title(" Retrieved Evidence")

    if "last_response" not in st.session_state:

        st.info(
            "No evidence available yet.\n\nGo to **Ask AI** and ask a question first."
        )

        return

    response = st.session_state["last_response"]

    st.subheader("Question")

    st.write(response.question)

    st.subheader("Answer")

    st.success(response.answer)

    st.subheader("Retrieved Pages")

    if response.retrieved_pages:

        st.write(
            ", ".join(
                [str(page) for page in response.retrieved_pages]
            )
        )

    st.subheader("Citations")

    if response.citations:

        for citation in response.citations:

            st.write(f"• {citation}")

    st.subheader("Evidence Chunks")

    for index, chunk in enumerate(response.chunks, start=1):

        with st.expander(
            f"Evidence {index} | Page {chunk['page']}"
        ):

            st.markdown(
                f"**Similarity Distance:** {chunk['distance']}"
            )

            st.write(chunk["text"])