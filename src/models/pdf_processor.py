import os
import fitz
from typing import Dict, List, Optional, Tuple
import streamlit as st
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from src.config.constants import (
    EMBEDDING_MODEL_NAME,
    MODEL_CACHE_DIR,
    CHUNK_OVERLAP,
    TOKENS_PER_CHUNK,
    RETRIEVER_K
)
from src.utils.file_utils import get_store_path, get_file_hash

def load_existing_store(store_path: str) -> Tuple[Optional[Chroma], Optional[EnsembleRetriever]]:
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=str(MODEL_CACHE_DIR)
        )
        vector_store = Chroma(
            persist_directory=store_path,
            embedding_function=embeddings
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
            bm25_retriever.k = RETRIEVER_K
            
            semantic_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K}
            )
            
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )
            
            return vector_store, hybrid_retriever
            
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
    
    return None, None

class PDFProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=str(MODEL_CACHE_DIR)
        )
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=EMBEDDING_MODEL_NAME,
            chunk_overlap=CHUNK_OVERLAP,
            tokens_per_chunk=TOKENS_PER_CHUNK
        )

    def create_hybrid_retriever(self, documents, vector_store):
        bm25_docs = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ) for doc in documents
        ]
        
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = RETRIEVER_K
        
        semantic_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K}
        )
        
        return EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )

    @st.cache_resource
    def process_document(_self, pdf_bytes: bytes, title: str) -> Tuple[Optional[Chroma], Optional[EnsembleRetriever]]:
        store_path = get_store_path(title)
        file_hash = get_file_hash(pdf_bytes)
        
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