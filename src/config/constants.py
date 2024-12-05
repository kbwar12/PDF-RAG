from pathlib import Path
from typing import Final

# ======================
# API Configuration
# ======================
OLLAMA_HOST: Final[str] = "http://localhost:11434"
MODEL_NAME: Final[str] = "llama3.2"

# ======================
# File System Paths
# ======================
VECTOR_STORES_DIR: Final[Path] = Path("./vector_stores")
MODEL_CACHE_DIR: Final[Path] = Path("./model_cache")

# ======================
# Model Parameters
# ======================
EMBEDDING_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
CHUNK_OVERLAP: Final[int] = 2
TOKENS_PER_CHUNK: Final[int] = 256
RETRIEVER_K: Final[int] = 5

# Create necessary directories
VECTOR_STORES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True) 