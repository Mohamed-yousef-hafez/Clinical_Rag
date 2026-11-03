"""
Module:
Multi-Step Retriever

Purpose:
Handle complex questions by decomposing them
into smaller retrieval queries.
"""

import google.generativeai as genai

from src.config.settings import settings


class MultiStepRetriever:

    def __init__(self, retriever):

        self.retriever = retriever

        genai.configure(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash"
        )

    def decompose(self, question):

        prompt = f"""
You are a query decomposition module.

Break the following complex clinical question
into 2 to 3 simple evidence-search questions.

Rules:

- Return only the questions.
- One question per line.
- Do not answer the questions.
- Do not add explanations.

Complex question:

{question}
"""

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 256
            }
        )

        queries = []

        for line in response.text.splitlines():

            line = line.strip()

            line = line.lstrip(
                "0123456789.-) "
            )

            if line and len(line) > 5:

                queries.append(line)

        return queries[:3]

    def search(self, question):

        sub_questions = self.decompose(question)

        all_chunks = []

        seen = set()

        for sub_question in sub_questions:

            result = self.retriever.search(
                sub_question
            )

            if result["status"] != "success":
                continue

            for chunk in result["chunks"]:

                key = (
                    chunk["page"],
                    chunk["text"]
                )

                if key not in seen:

                    seen.add(key)

                    all_chunks.append(chunk)

        return {
            "status": (
                "success"
                if all_chunks
                else "out_of_scope"
            ),
            "sub_questions": sub_questions,
            "chunks": all_chunks
        }