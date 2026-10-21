
import streamlit as st


def render_evidence():
    st.title("Retrieved Evidence")

    response = st.session_state.get("last_response")

    if response is None:
        st.info(
            "No evidence available yet.\n\n"
            "Go to Ask AI and ask a question first."
        )
        return

    st.subheader("Question")
    st.write(response.question)

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Retrieved Pages")
    pages = getattr(response, "retrieved_pages", []) or []

    if pages:
        st.write(", ".join(str(page) for page in pages))
    else:
        st.caption("No page information available.")

    st.subheader("Citations")
    citations = getattr(response, "citations", []) or []

    if citations:
        for citation in citations:
            st.write(f"• {citation}")
    else:
        st.caption("No citations available.")

    st.subheader("Evidence Chunks")
    chunks = getattr(response, "chunks", []) or []

    if not chunks:
        st.info("No evidence chunks were retrieved.")
        return

    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "Unknown")

        with st.expander(f"Evidence {index} | Page {page}"):
            distance = chunk.get("distance")
            rerank_score = chunk.get("rerank_score")

            if distance is not None:
                st.caption(f"Similarity Distance: {distance:.4f}")

            if rerank_score is not None:
                st.caption(f"Rerank Score: {rerank_score:.4f}")

            st.write(chunk.get("text", "No text available."))
