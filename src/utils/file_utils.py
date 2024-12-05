import hashlib
from pathlib import Path
from typing import List, Tuple

def get_store_path(title: str) -> str:
    """Generate a safe path for storing vector data."""
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
    return f"./vector_stores/{safe_title}_{title_hash}"

def get_file_hash(pdf_bytes: bytes) -> str:
    """Generate MD5 hash for PDF file contents."""
    return hashlib.md5(pdf_bytes).hexdigest()

def get_existing_stores() -> List[Tuple[str, str]]:
    """Get list of existing vector stores."""
    stores = []
    vector_store_path = Path("./vector_stores")
    
    if vector_store_path.exists():
        for store_dir in vector_store_path.iterdir():
            if store_dir.is_dir() and not store_dir.name.startswith('.'):
                title = " ".join(store_dir.name.split('_')[:-1])
                stores.append((title, str(store_dir)))
    
    return sorted(stores) 