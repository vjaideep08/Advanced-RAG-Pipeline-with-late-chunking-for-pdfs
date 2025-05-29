# services/embedding_service.py
import logging
from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os

from config import settings

logger = logging.getLogger(__name__)

# Suppress transformers progress bars and warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

class EmbeddingService:
    """Service for generating embeddings using late chunking"""
    
    def __init__(self):
        logger.info("Loading embedding model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            settings.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("✓ Embedding model loaded successfully")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (batch processing)"""
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = []
        
        # Process in batches
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = await self._process_batch(batch_texts)
            embeddings.extend(batch_embeddings)
            
            # Log progress every few batches
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(texts):
                processed = min(i + batch_size, len(texts))
                logger.info(f"  Progress: {processed}/{len(texts)} embeddings generated")
        
        logger.info("✓ All embeddings generated successfully")
        return embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts"""
        batch_embeddings = []
        
        for text in texts:
            try:
                if hasattr(self.model, 'encode'):
                    try:
                        # Try to disable progress bar
                        embedding = self.model.encode(text, show_progress_bar=False)
                    except TypeError:
                        # Fallback if parameter not supported
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            embedding = self.model.encode(text)
                else:
                    # Manual encoding
                    inputs = self.tokenizer(
                        text, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True,
                        max_length=512
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                if hasattr(embedding, 'tolist'):
                    batch_embeddings.append(embedding.tolist())
                else:
                    batch_embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
                batch_embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
        
        return batch_embeddings
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]
