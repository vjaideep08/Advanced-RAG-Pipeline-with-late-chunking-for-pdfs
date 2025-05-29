# ==============================================================================
# FILE: services/vector_store_factory.py
# ==============================================================================
import logging

logger = logging.getLogger(__name__)

def create_vector_store():
    """Factory function to create appropriate vector store"""
    try:
        # Try to create Qdrant vector store first
        from services.vector_store import VectorStore
        vector_store = VectorStore()
        logger.info("Successfully initialized Qdrant vector store")
        return vector_store
        
    except Exception as e:
        logger.warning(f"Failed to initialize Qdrant vector store: {e}")
        logger.info("Falling back to in-memory vector store")
        
        # Fallback to in-memory vector store
        from services.vector_store_memory import InMemoryVectorStore
        return InMemoryVectorStore()
