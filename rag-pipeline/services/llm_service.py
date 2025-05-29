# ==============================================================================
# FILE: services/llm_service.py
# ==============================================================================
import logging
from typing import List, Optional
import os
from anthropic import Anthropic

from models.document import DocumentChunk
from models.chat import ChatMessage
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Claude API"""
    
    def __init__(self):
        # Validate API key
        if not settings.CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY is not set in environment variables")
        
        if not settings.CLAUDE_API_KEY.startswith('sk-ant-'):
            logger.warning("Claude API key format seems incorrect. Expected format: sk-ant-...")
        
        try:
            # Initialize Anthropic client with proper parameters
            self.client = Anthropic(
                api_key=settings.CLAUDE_API_KEY,
                max_retries=3,
                timeout=60.0
            )
            logger.info("Successfully initialized Claude API client")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude API client: {e}")
            raise
    
    async def generate_response(
        self, 
        query: str, 
        context: List[DocumentChunk],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Generate response using Claude with RAG context"""
        
        # Prepare context from retrieved documents
        context_text = self._prepare_context(context)
        
        # Build conversation history
        messages = []
        if chat_history:
            for msg in chat_history[-5:]:  # Keep last 5 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current query with context
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(query, context_text)
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = self.client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                system=system_prompt,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating response from Claude: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude"""
        return """You are an AI assistant specialized in helping users draft project proposals. 
        You have access to a knowledge base of project documents and can provide detailed, 
        accurate information to assist in proposal writing.
        
        Your capabilities include:
        - Analyzing project requirements and constraints
        - Identifying key deliverables and milestones
        - Providing insights on timelines and budgets
        - Suggesting best practices for proposal writing
        - Answering specific questions about project documentation
        
        Always provide helpful, accurate, and contextual responses based on the available documents.
        If you cannot find relevant information in the provided context, clearly state this limitation."""
    
    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with context"""
        return f"""Based on the following context from project documents, please answer the user's question:

Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what additional information might be needed."""
    
    def _prepare_context(self, documents: List[DocumentChunk]) -> str:
        """Prepare context string from retrieved documents"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"Document {i} (Source: {source}):\n{doc.content}\n")
        
        return "\n".join(context_parts)
    
    async def extract_fine_prints(self, documents: List[DocumentChunk]) -> List[str]:
        """Extract fine-prints from documents using Claude"""
        context_text = self._prepare_context(documents)
        
        prompt = f"""Analyze the following project documents and extract the most critical "fine-prints" - 
        key details that are essential for drafting project proposals. Focus on:

        1. Important deadlines and timelines
        2. Budget constraints and financial requirements
        3. Technical specifications and requirements
        4. Deliverables and milestones
        5. Compliance and regulatory requirements
        6. Risk factors and limitations
        7. Stakeholder expectations and dependencies

        Documents:
        {context_text}

        Please extract and list the most critical fine-prints, categorizing them by importance and type."""
        
        try:
            response = self.client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=settings.MAX_TOKENS,
                temperature=0.3,  # Lower temperature for more focused extraction
                system="You are an expert at analyzing project documents and extracting critical details for proposal writing.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.split('\n')
            
        except Exception as e:
            logger.error(f"Error extracting fine-prints: {e}")
            return ["Error occurred while extracting fine-prints."]
