import streamlit as st
from typing import List, Dict

def create_chat_interface():
    """Initialize the chat interface in the session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})

def clear_chat():
    """Clear the chat history."""
    st.session_state.messages = []

def get_chat_context(num_messages: int = 6) -> str:
    """Get recent chat history as a formatted string."""
    if not hasattr(st.session_state, 'messages'):
        return ""
        
    recent_messages = st.session_state.messages[-num_messages:]
    return "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in recent_messages 
        if msg['role'] != 'system'
    ])

def create_prompt(context: str, query: str, chat_history: str = "") -> str:
    """Create an enhanced prompt for the LLM."""
    return f"""You are a helpful AI assistant answering questions about a document. Your task is to provide accurate, relevant answers based on the provided context and chat history.

    Guidelines:
    1. Use the document context as your primary source of information
    2. Consider the chat history for context and follow-up questions
    3. If referring to something mentioned in chat history, explicitly state that
    4. If the context doesn't contain enough information, clearly say so
    5. Keep answers focused and relevant to the document content
    
    Previous Chat History:
    {chat_history}

    Document Context:
    {context}

    Current Question: {query}

    Please provide a clear, well-structured answer that:
    - Directly addresses the question
    - References specific parts of the document when relevant
    - Acknowledges any connection to previous questions if applicable
    - Clearly states if any information is missing from the context

    Answer:""" 