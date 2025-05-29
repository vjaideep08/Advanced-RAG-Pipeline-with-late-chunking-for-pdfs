# ==============================================================================
# FILE: models/chat.py
# ==============================================================================
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    """Represents a chat message"""
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
