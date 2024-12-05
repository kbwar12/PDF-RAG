# PDF Question & Answer System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using advanced language models and vector search.

## Features

- PDF document processing and text extraction
- Vector-based document storage using Chroma DB
- Hybrid search combining BM25 and semantic search
- Interactive chat interface with context awareness
- Support for follow-up questions
- Document management system

## Project Structure

```
.
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── constants.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pdf_processor.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   └── query_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py
│   └── __init__.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install and start Ollama server (required for query processing)
4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Upload a PDF document using the "Process New Document" tab
2. Wait for the document to be processed
3. Ask questions about the document's content
4. Use the "Chat with Documents" tab to interact with previously processed documents

## Dependencies

- streamlit
- PyMuPDF
- langchain
- sentence-transformers
- chromadb
- requests

## Notes

- The application requires an Ollama server running locally for query processing
- Processed documents are stored in the `./vector_stores` directory
- Model cache is stored in the `./model_cache` directory 