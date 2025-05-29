# ==============================================================================
# FILE: services/fine_prints_extractor.py
# ==============================================================================
import logging
from typing import List
import re

from models.document import FinePrint
from services.llm_service import LLMService
from services.vector_store_factory import create_vector_store

logger = logging.getLogger(__name__)

class FinePrintsExtractor:
    """Service for extracting fine-prints from documents"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.vector_store = None
    
    async def extract_fine_prints(self) -> List[FinePrint]:
        """Extract fine-prints from all documents"""
        try:
            # Initialize vector store if not already done
            if not self.vector_store:
                self.vector_store = create_vector_store()
            
            # Get all documents from vector store (or a representative sample)
            all_docs = await self._get_all_documents()
            
            if not all_docs:
                logger.warning("No documents found for fine-prints extraction")
                return []
            
            # Use LLM to extract fine-prints
            fine_prints_text = await self.llm_service.extract_fine_prints(all_docs)
            
            # Parse the extracted text into structured fine-prints
            structured_fine_prints = self._parse_fine_prints(fine_prints_text, all_docs)
            
            return structured_fine_prints
            
        except Exception as e:
            logger.error(f"Error extracting fine-prints: {e}")
            return []
    
    async def _get_all_documents(self) -> List:
        """Get all documents from vector store"""
        try:
            # Since Qdrant doesn't have a direct "get all" method, 
            # we'll use a broad search query
            broad_queries = [
                "project requirements", "timeline", "budget", "deliverables",
                "specifications", "deadlines", "milestones", "constraints"
            ]
            
            all_docs = []
            seen_ids = set()
            
            for query in broad_queries:
                docs = await self.vector_store.similarity_search(query, k=10)
                for doc in docs:
                    if doc.id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc.id)
            
            return all_docs[:50]  # Limit to top 50 documents to avoid token limits
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _parse_fine_prints(self, fine_prints_text: List[str], source_docs) -> List[FinePrint]:
        """Parse extracted fine-prints text into structured objects"""
        fine_prints = []
        
        # Categories for classification
        categories = {
            'timeline': ['deadline', 'timeline', 'schedule', 'due date', 'milestone'],
            'budget': ['budget', 'cost', 'financial', 'payment', 'funding'],
            'requirements': ['requirement', 'specification', 'must', 'shall', 'mandatory'],
            'deliverables': ['deliverable', 'output', 'product', 'result', 'outcome'],
            'compliance': ['compliance', 'regulation', 'standard', 'policy', 'law'],
            'risks': ['risk', 'limitation', 'constraint', 'issue', 'challenge']
        }
        
        for i, text_line in enumerate(fine_prints_text):
            if not text_line.strip():
                continue
                
            # Determine category
            category = self._classify_fine_print(text_line.lower(), categories)
            
            # Determine importance (simple heuristic)
            importance = self._determine_importance(text_line)
            
            # Extract title (first few words or sentence)
            title = self._extract_title(text_line)
            
            # Determine source document (simplified)
            source_doc = source_docs[i % len(source_docs)].metadata.get('source', 'Unknown') if source_docs else 'Unknown'
            
            fine_print = FinePrint(
                title=title,
                content=text_line.strip(),
                importance=importance,
                category=category,
                source_document=source_doc
            )
            
            fine_prints.append(fine_print)
        
        return fine_prints
    
    def _classify_fine_print(self, text: str, categories: dict) -> str:
        """Classify fine-print into categories"""
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'general'
    
    def _determine_importance(self, text: str) -> str:
        """Determine importance level of fine-print"""
        high_importance_indicators = [
            'critical', 'essential', 'mandatory', 'required', 'must',
            'deadline', 'final', 'non-negotiable', 'urgent'
        ]
        
        medium_importance_indicators = [
            'important', 'should', 'recommended', 'preferred', 'expected'
        ]
        
        text_lower = text.lower()
        
        if any(indicator in text_lower for indicator in high_importance_indicators):
            return 'high'
        elif any(indicator in text_lower for indicator in medium_importance_indicators):
            return 'medium'
        else:
            return 'low'
    
    def _extract_title(self, text: str) -> str:
        """Extract title from fine-print text"""
        # Take first sentence or first 10 words
        sentences = re.split(r'[.!?]', text)
        if sentences and sentences[0].strip():
            title = sentences[0].strip()
        else:
            words = text.split()
            title = ' '.join(words[:10])
        
        return title[:100]  # Limit title length
