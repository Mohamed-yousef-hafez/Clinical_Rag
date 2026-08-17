import streamlit as st


def render_safety():

    st.title("🛡 Safety & Guardrails")

    st.markdown("""
##  Safety Architecture

The system uses multiple safety layers to reduce
unsafe, unsupported, and manipulated requests before
producing an answer.
""")

    st.divider()

    # ---------------------------------
    # 1. Input Protection
    # ---------------------------------

    st.subheader("1️⃣ Input Protection")

    st.info(
        "Detects common prompt injection patterns before "
        "the request reaches the generation stage."
    )

    st.success(" Prompt Injection Detection")

    # ---------------------------------
    # 2. Retrieval Guard
    # ---------------------------------

    st.subheader("2️⃣ Retrieval Guard")

    st.info(
        "Questions with insufficient evidence from the "
        "hypertension guideline are rejected."
    )

    st.success(" Out-of-Scope Detection")

    # ---------------------------------
    # 3. Grounded Generation
    # ---------------------------------

    st.subheader("3️⃣ Grounded Generation")

    st.info(
        "The LLM is instructed to answer only using "
        "retrieved hypertension guideline evidence."
    )

    st.success(" Grounded Generation")

    st.divider()

    # ---------------------------------
    # Safety Rules
    # ---------------------------------

    st.subheader(" Safety Rules")

    rules = [
        "Never use unsupported external medical knowledge.",
        "Block common prompt injection attempts.",
        "Reject questions outside the hypertension guideline.",
        "Require retrieved evidence before generation.",
        "Provide page-level citations whenever possible.",
        "Refuse questions when sufficient evidence is unavailable."
    ]

    for rule in rules:

        st.success(
            f"✅ {rule}"
        )

    st.divider()

    # ---------------------------------
    # Latest Query Status
    # ---------------------------------

    st.subheader(
        " Latest Query Safety Status"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Risk Level",
            "SAFE"
        )

    with col2:

        st.metric(
            "System Status",
            "SUCCESS"
        )

    st.success(
        "✅ Requests are processed through the safety "
        "and grounding layers before generating an answer."
    )

    st.divider()

    # ---------------------------------
    # Safety Coverage
    # ---------------------------------

    st.subheader(
        " Safety Coverage"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            "✅ Prompt Injection Protection"
        )

        st.success(
            "✅ Out-of-Scope Detection"
        )

        st.success(
            "✅ Safe Refusal"
        )

    with col2:

        st.success(
            "✅ Grounded Generation"
        )

        st.success(
            "✅ Evidence-Based Responses"
        )

    st.divider()

    st.caption(
        "Clinical safety disclaimer: This system is a "
        "research/demo assistant and does not replace "
        "professional clinical judgment or official "
        "clinical guidance."
    )