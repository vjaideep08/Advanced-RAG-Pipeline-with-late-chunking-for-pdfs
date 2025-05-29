# ==============================================================================
# FILE: utils/helpers.py
# ==============================================================================
import logging
from typing import Any, Dict
import hashlib

logger = logging.getLogger(__name__)

def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """Generate a unique ID for a document chunk"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    source = metadata.get('source', 'unknown')
    chunk_index = metadata.get('chunk_index', 0)
    return f"{source}_{chunk_index}_{content_hash[:8]}"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')  # Remove null bytes
    
    return text.strip()

def chunk_text_by_words(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Simple text chunking by words with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks
