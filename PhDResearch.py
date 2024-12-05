import fitz
import streamlit as st
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import hashlib
import requests

# Constants
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2"

class PDFProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./model_cache"
        )
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="all-MiniLM-L6-v2",
            chunk_overlap=2,
            tokens_per_chunk=256
        )
        Path("./vector_stores").mkdir(parents=True, exist_ok=True)
        Path("./model_cache").mkdir(exist_ok=True)
        
    def get_store_path(self, title: str) -> str:
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        return f"./vector_stores/{safe_title}_{title_hash}"

    @staticmethod
    def get_file_hash(pdf_bytes: bytes) -> str:
        return hashlib.md5(pdf_bytes).hexdigest()

    def create_hybrid_retriever(self, documents, vector_store):
        bm25_docs = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ) for doc in documents
        ]
        
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 5
        
        semantic_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )
        return hybrid_retriever

    @st.cache_resource
    def process_document(_self, pdf_bytes: bytes, title: str) -> Tuple[Optional[Chroma], Optional[EnsembleRetriever]]:
        store_path = _self.get_store_path(title)
        file_hash = _self.get_file_hash(pdf_bytes)
        
        if os.path.exists(f"{store_path}_{file_hash}"):
            vector_store = Chroma(
                persist_directory=f"{store_path}_{file_hash}",
                embedding_function=_self.embeddings
            )
            stored_docs = vector_store.get()
            if stored_docs['documents']:
                documents = [{
                    "content": doc,
                    "metadata": meta
                } for doc, meta in zip(stored_docs['documents'], stored_docs['metadatas'])]
                hybrid_retriever = _self.create_hybrid_retriever(documents, vector_store)
                return vector_store, hybrid_retriever
            
        try:
            text = _self._extract_text(pdf_bytes)
            chunks = _self._create_chunks(text)
            vector_store = _self._create_vector_store(chunks, f"{store_path}_{file_hash}")
            hybrid_retriever = _self.create_hybrid_retriever(chunks, vector_store)
            return vector_store, hybrid_retriever
            
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            return None, None

    @staticmethod
    def _extract_text(pdf_bytes: bytes) -> str:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = " ".join(page.get_text() for page in doc)
        return text

    def _create_chunks(self, text: str) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        return [{
            'content': chunk,
            'metadata': {
                'chunk_id': i
            }
        } for i, chunk in enumerate(chunks)]

    def _create_vector_store(self, chunks: List[Dict], store_path: str) -> Chroma:
        vector_store = Chroma.from_texts(
            texts=[chunk['content'] for chunk in chunks],
            embedding=self.embeddings,
            metadatas=[chunk['metadata'] for chunk in chunks],
            persist_directory=store_path
        )
        vector_store.persist()
        return vector_store

def get_existing_stores() -> List[Tuple[str, str]]:
    stores = []
    vector_store_path = Path("./vector_stores")
    
    if vector_store_path.exists():
        for store_dir in vector_store_path.iterdir():
            if store_dir.is_dir() and not store_dir.name.startswith('.'):
                title = " ".join(store_dir.name.split('_')[:-1])
                stores.append((title, str(store_dir)))
    
    return sorted(stores)

def load_existing_store(store_path: str) -> Tuple[Optional[Chroma], Optional[EnsembleRetriever]]:
    try:
        vector_store = Chroma(
            persist_directory=store_path,
            embedding_function=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./model_cache"
            )
        )
        
        stored_docs = vector_store.get()
        if stored_docs['documents']:
            documents = [{
                "content": doc,
                "metadata": meta
            } for doc, meta in zip(stored_docs['documents'], stored_docs['metadatas'])]
            
            bm25_docs = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ) for doc in documents
            ]
            
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = 5
            
            semantic_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )
            
            return vector_store, hybrid_retriever
            
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
    
    return None, None

def create_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def clear_chat():
    st.session_state.messages = []

@st.cache_data
def interpret_query(_query: str, chat_history: str = "") -> str:
    """
    Interprets user query for RAG retrieval while handling both standalone and follow-up questions.

    Args:
        _query (str): Current user question
        chat_history (str): Previous conversation context (optional)

    Returns:
        str: Enhanced search query optimized for RAG retrieval
    """
    context = f"Chat History:\n{chat_history}\n" if chat_history else ""
    interpretation_prompt = f"""{context}Question: {_query}

Rewrite as a concise search query using key technical terms and context. Return ONLY the search query, nothing else."""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": interpretation_prompt,
                "stream": False
            },
            timeout=240
        )

        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return _query

    except Exception as e:
        st.error(f"Query interpretation failed: {str(e)}")
        return _query

def create_prompt(context: str, query: str, chat_history: str = "") -> str:
    """Create an enhanced prompt for the LLM that better handles follow-up questions"""
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

@st.cache_data
def query_document(_vector_store: Chroma, _retriever: EnsembleRetriever, query: str, k: int = 5) -> str:
    if not _vector_store or not _retriever:
        return "Please upload a document first."
    
    # Get chat history context (last 3 exchanges)
    chat_context = ""
    if hasattr(st.session_state, 'messages'):
        recent_messages = st.session_state.messages[-6:]
        chat_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages 
            if msg['role'] != 'system'
        ])
    
    # Interpret query with chat history context
    enhanced_query = interpret_query(query, chat_context)
    
    results = _retriever.get_relevant_documents(enhanced_query)
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

def main():
    st.title("PDF Question & Answer System")
    
    tab1, tab2 = st.tabs(["Process New Document", "Chat with Documents"])
    
    with tab1:
        pdf_processor = PDFProcessor()
        
        pdf_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            on_change=lambda: st.cache_data.clear()
        )
        
        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
            
            with st.spinner('Processing document...'):
                vector_store, retriever = pdf_processor.process_document(pdf_bytes, pdf_file.name)
            
            if vector_store and retriever:
                st.success("Document processed successfully!")
                
                st.subheader("Ask Questions About the Document")
                question = st.text_input("Enter your question:", key="new_doc_question")
                
                if st.button("Ask", key="new_doc_ask"):
                    if question:
                        with st.spinner('Processing query...'):
                            enhanced_query = interpret_query(question)
                            st.info(f"Interpreted query: {enhanced_query}")
                            
                            with st.spinner('Searching for answer...'):
                                answer = query_document(vector_store, retriever, question)
                                st.write("Answer:", answer)
                    else:
                        st.warning("Please enter a question.")
            else:
                st.error("Failed to process document.")
        else:
            st.info("Please upload a PDF document to begin.")
    
    with tab2:
        st.subheader("Chat with Existing Documents")
        
        create_chat_interface()
        
        with st.sidebar:
            existing_stores = get_existing_stores()
            
            if not existing_stores:
                st.info("No processed documents found. Please process a document first.")
                return
                
            selected_title = st.selectbox(
                "Select a document to chat with:",
                options=[title for title, _ in existing_stores],
                format_func=lambda x: x
            )
            
            if st.button("Clear Chat History"):
                clear_chat()
                st.rerun()
        
        selected_path = next(path for title, path in existing_stores if title == selected_title)
        
        if selected_path:
            vector_store, retriever = load_existing_store(selected_path)
            
            if vector_store and retriever:
                messages_container = st.container()
                with messages_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                
                if prompt := st.chat_input("Ask a question about the document..."):
                    add_message("user", prompt)
                    
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner('Thinking...'):
                            enhanced_query = interpret_query(prompt)
                            
                            response_placeholder = st.empty()
                            response_placeholder.write("🤔 Interpreting query...")
                            
                            st.write(f"*Interpreted as: {enhanced_query}*")
                            
                            response_placeholder.write("🔍 Searching document...")
                            answer = query_document(vector_store, retriever, prompt)
                            
                            response_placeholder.write(answer)
                            
                            add_message("assistant", answer)
                            add_message("system", f"Query interpreted as: {enhanced_query}")
            else:
                st.error("Failed to load the selected document.")

if __name__ == "__main__":
    main()
