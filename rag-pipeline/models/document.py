from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid

class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any]
    
    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
        super().__init__(**data)

class FinePrint(BaseModel):
    """Represents a fine-print extracted from documents"""
    title: str
    content: str
    importance: str  # "high", "medium", "low"
    category: str    # "timeline", "budget", "requirements", "deliverables", etc.
    source_document: str
