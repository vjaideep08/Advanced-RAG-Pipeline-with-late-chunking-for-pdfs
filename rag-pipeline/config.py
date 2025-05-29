from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    CLAUDE_API_KEY: str = "<Your api key here>"
    
    # Paths
    PDF_PATH: str = "/home/imart/jayadeep/poc/rag_poc/pdfs/"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "project_documents"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v2-base-en"
    EMBEDDING_DIMENSION: int = 768
    
    # Chunking Configuration
    MAX_CHUNK_LENGTH: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # LLM Configuration
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate critical settings"""
        # Check if PDF path exists
        if not Path(self.PDF_PATH).exists():
            print(f"Warning: PDF path {self.PDF_PATH} does not exist")
        
        # Check API key
        if not self.CLAUDE_API_KEY:
            print("Error: CLAUDE_API_KEY environment variable is required")
            print("Please set it in your .env file or environment:")
            print("export CLAUDE_API_KEY='your-api-key-here'")

settings = Settings()
