# ==============================================================================
# FILE: services/vector_store.py
# ==============================================================================
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import time

from models.document import DocumentChunk
from services.embedding_service import EmbeddingService
from config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Qdrant vector store implementation with connection retry"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_service = EmbeddingService()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._connect_with_retry()
        self._initialize_collection()
    
    def _connect_with_retry(self):
        """Connect to Qdrant with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.client = QdrantClient(url=settings.QDRANT_URL)
                # Test connection
                self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {settings.QDRANT_URL}")
                return
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All connection attempts failed")
                    raise Exception(f"Could not connect to Qdrant after {self.max_retries} attempts: {e}")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    async def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to vector store"""
        if not documents:
            return
        
        try:
            # Generate embeddings
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Create points for Qdrant
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            # Convert results to DocumentChunk objects
            documents = []
            for result in search_results:
                doc = DocumentChunk(
                    id=str(result.id),
                    content=result.payload["content"],
                    metadata=result.payload["metadata"]
                )
                # Add similarity score
                doc.metadata['similarity_score'] = result.score
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def close(self):
        """Close connection to Qdrant"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing vector store connection: {e}")
