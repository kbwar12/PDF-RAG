import requests
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from typing import Optional, Tuple

from src.config.constants import OLLAMA_HOST, MODEL_NAME
from src.services.chat_service import get_chat_context, create_prompt

@st.cache_data
def query_document(_vector_store: Chroma, _retriever: EnsembleRetriever, query: str, k: int = 5) -> str:
    """Process a query and return the response."""
    if not _vector_store or not _retriever:
        return "Please upload a document first."
    
    chat_context = get_chat_context()
    results = _retriever.get_relevant_documents(query)
    context_parts = []
    
    for doc in results:
        context_parts.append(f"[Relevant Content]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    prompt = create_prompt(context, query, chat_context)
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=240
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama request failed with status {response.status_code}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}" 