
##Step One
#python -m venv venv

#venv\Scripts\activate

#python -m pip install --upgrade pip

#pip install -r requirements.txt

#python -m src.build_pipeline
#venv\Scripts\python.exe -m streamlit run app.py
#venv\Scripts\python.exe -m src.build_pipeline //Knowledge base
#python -c "import google.generativeai as genai; from src.config.settings import settings; genai.configure(api_key=settings.GEMINI_API_KEY); print([m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods])"

-------------------------
# Clinical Evidence Assistant

## Overview

A Retrieval-Augmented Generation (RAG) assistant that answers domain-specific questions using uploaded PDF guidelines with grounded responses, citations, safety guardrails, and evaluation metrics.

## Features

- Intelligent PDF ingestion
- Semantic retrieval
- Grounded LLM generation
- Citations
- Safety guardrails
- Evaluation dashboard
- Live demo scenarios

---------------------------------
SYSTEM_DESCRIPTION = """
A Retrieval-Augmented Generation assistant that answers questions
using trusted uploaded documents with citations and safety guardrails.
"""