# ==============================================================================
# FILE: services/vector_store_memory.py
# ==============================================================================
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid

from models.document import DocumentChunk
from services.embedding_service import EmbeddingService
from config import settings

logger = logging.getLogger(__name__)

class InMemoryVectorStore:
    """In-memory vector store for development/testing without Qdrant"""
    
    def __init__(self):
        self.documents: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_service = EmbeddingService()
        logger.info("Initialized In-Memory Vector Store")
    
    async def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to in-memory store"""
        if not documents:
            return
        
        try:
            # Generate embeddings
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Store documents and embeddings
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.id if doc.id else str(uuid.uuid4())
                self.documents[doc_id] = doc
                self.embeddings[doc_id] = embedding
            
            logger.info(f"Added {len(documents)} documents to in-memory vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to in-memory vector store: {e}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Search for similar documents using cosine similarity"""
        try:
            if not self.documents:
                logger.warning("No documents in vector store")
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                try:
                    # Convert to numpy arrays
                    query_vec = np.array(query_embedding).reshape(1, -1)
                    doc_vec = np.array(doc_embedding).reshape(1, -1)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_vec, doc_vec)[0][0]
                    similarities.append((doc_id, similarity))
                    
                except Exception as e:
                    logger.warning(f"Error calculating similarity for doc {doc_id}: {e}")
                    continue
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_docs = similarities[:k]
            
            # Return DocumentChunk objects
            result_docs = []
            for doc_id, similarity in top_docs:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    # Add similarity score to metadata
                    doc.metadata['similarity_score'] = similarity
                    result_docs.append(doc)
            
            logger.info(f"Found {len(result_docs)} similar documents for query: {query[:50]}...")
            return result_docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def close(self):
        """Close connection (no-op for in-memory store)"""
        logger.info("Closing in-memory vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }
