import streamlit as st

from src.utils.cache import load_pipeline


def render_ask():

    st.title("💬 Ask AI")

    question = st.text_area(
        "Enter your question",
        height=120,
        placeholder=(
            "Example: What is hypertension?"
        )
    )

    if st.button(" Generate Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return

        try:

            # Load cached RAG pipeline
            with st.spinner(
                "Searching hypertension evidence..."
            ):

                pipeline = load_pipeline()

                response = pipeline.ask(
                    question.strip()
                )

            answer = str(
                response.answer
            )

            # Gemini quota protection
            if (
                "429" in answer
                or "quota" in answer.lower()
            ):

                st.warning(
                    " Gemini API quota has been reached. "
                    "Please wait and try again later."
                )

                return

            # Safety blocked
            if response.status == "blocked":

                st.error(
                    response.answer
                )

                return

            # Out of scope
            if response.status == "out_of_scope":

                st.warning(
                    "I couldn't find sufficient evidence "
                    "in the uploaded hypertension guideline."
                )

                return

            # Success
            st.success(
                "✅ Answer Generated"
            )

            st.subheader(
                " Evidence-Based Answer"
            )

            st.write(
                response.answer
            )

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Latency",
                    f"{response.latency:.2f} sec"
                )

            with col2:

                st.metric(
                    "Retrieved Chunks",
                    len(response.chunks)
                )

            with col3:

                evidence_pages = len(
                    set(
                        str(
                            chunk.get(
                                "page",
                                "Unknown"
                            )
                        )
                        for chunk in response.chunks
                    )
                )

                st.metric(
                    "Evidence Pages",
                    evidence_pages
                )

            with col4:

                confidence = getattr(
                    response,
                    "confidence",
                    "Low"
                )

                st.metric(
                    "Confidence",
                    confidence
                )

            # Citations
            pages = []

            for chunk in response.chunks:

                page = str(
                    chunk.get(
                        "page",
                        "Unknown"
                    )
                )

                if page not in pages:

                    pages.append(page)

            if pages:

                st.subheader(
                    " Citations"
                )

                st.write(
                    ", ".join(
                        f"Page {page}"
                        for page in pages
                    )
                )

            # Retrieved Evidence
            st.subheader(
                " Retrieved Evidence"
            )

            for i, chunk in enumerate(
                response.chunks,
                start=1
            ):

                page = chunk.get(
                    "page",
                    "Unknown"
                )

                distance = chunk.get(
                    "distance",
                    None
                )

                rerank_score = chunk.get(
                    "rerank_score",
                    None
                )

                with st.expander(
                    f"Evidence {i} | Page {page}"
                ):

                    st.write(
                        chunk.get(
                            "text",
                            ""
                        )
                    )

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:

                        if distance is not None:

                            st.metric(
                                "Distance",
                                f"{distance:.4f}"
                            )

                        else:

                            st.metric(
                                "Distance",
                                "N/A"
                            )

                    with metric_col2:

                        if rerank_score is not None:

                            st.metric(
                                "Rerank Score",
                                f"{rerank_score:.4f}"
                            )

                        else:

                            st.metric(
                                "Rerank Score",
                                "N/A"
                            )

        except Exception as e:

            error_message = str(e)

            if (
                "429" in error_message
                or "quota" in error_message.lower()
            ):

                st.warning(
                    " Gemini API quota has been reached."
                )

                st.info(
                    "Please wait before sending another question."
                )

            else:

                st.error(
                    f"❌ Ask AI Error: {error_message}"
                )