import streamlit as st
from src.models.pdf_processor import PDFProcessor, load_existing_store
from src.services.chat_service import create_chat_interface, add_message, clear_chat
from src.services.query_service import query_document
from src.utils.file_utils import get_existing_stores

def main():
    # Add custom styling and a more professional title
    st.set_page_config(page_title="PDF Q&A System", layout="wide")
    
    st.markdown("""
        <h1 style='text-align: center; color: #2e4057;'> PDF Question & Answer System</h1>
        <p style='text-align: center; color: #666;'>Upload PDFs and chat with their contents</p>
        """, unsafe_allow_html=True)
    
    # Use more descriptive tab names with icons
    tab1, tab2 = st.tabs(["📥 Upload & Process", "💬 Chat Interface"])
    
    with tab1:
        # Add a cleaner upload interface
        st.markdown("### Upload Your Document")
        col1, col2 = st.columns([2, 1])
        with col1:
            pdf_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                on_change=lambda: st.cache_data.clear(),
                help="Upload a PDF document to process and analyze"
            )
        
        pdf_processor = PDFProcessor()
        
        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
            
            with st.spinner('Processing document...'):
                vector_store, retriever = pdf_processor.process_document(pdf_bytes, pdf_file.name)
            
            if vector_store and retriever:
                st.success("✅ Document processed successfully!")
                
                st.markdown("### Ask Your First Question")
                with st.container():
                    question = st.text_input(
                        "What would you like to know about the document?",
                        key="new_doc_question",
                        placeholder="Enter your question here..."
                    )
                    
                    if st.button("🔍 Get Answer", key="new_doc_ask", type="primary"):
                        if question:
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
        st.markdown("### 💬 Document Chat Interface")
        
        # Move document selection to a more prominent position
        col1, col2 = st.columns([3, 1])
        with col1:
            existing_stores = get_existing_stores()
            if not existing_stores:
                st.info("📝 No processed documents found. Please process a document first.")
                return
                
            selected_title = st.selectbox(
                "Select a document to chat with:",
                options=[title for title, _ in existing_stores],
                format_func=lambda x: x,
                help="Choose a previously processed document to interact with"
            )
        
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary"):
                clear_chat()
                st.rerun()
        
        # Add a visual separator
        st.divider()
        
        create_chat_interface()
        
        selected_path = next(path for title, path in existing_stores if title == selected_title)
        
        if selected_path:
            vector_store, retriever = load_existing_store(selected_path)
            
            if vector_store and retriever:
                messages_container = st.container()
                with messages_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                
                if prompt := st.chat_input("Type your question here..."):
                    add_message("user", prompt)
                    
                    with st.chat_message("user"):
                        st.markdown(f"**Question:** {prompt}")
                    
                    with st.chat_message("assistant"):
                        with st.spinner('Searching through document...'):
                            response_placeholder = st.empty()
                            response_placeholder.markdown("🔍 *Analyzing document...*")
                            answer = query_document(vector_store, retriever, prompt)
                            
                            response_placeholder.markdown(f"**Answer:** {answer}")
                            
                            add_message("assistant", answer)
            else:
                st.error("Failed to load the selected document.")

if __name__ == "__main__":
    main() 