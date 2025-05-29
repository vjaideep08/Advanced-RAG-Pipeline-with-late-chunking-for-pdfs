# ==============================================================================
# FILE: services/document_processor.py
# ==============================================================================
import os
import logging
from typing import List, Tuple
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path

from models.document import DocumentChunk
from config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF processing and late chunking"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.EMBEDDING_MODEL, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            settings.EMBEDDING_MODEL, 
            trust_remote_code=True
        )
        
    async def process_pdf_directory(self, pdf_path: str) -> List[DocumentChunk]:
        """Process all PDFs in the given directory"""
        chunks = []
        pdf_files = list(Path(pdf_path).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_path}")
            return []
            
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                pdf_chunks = await self.process_single_pdf(pdf_file)
                chunks.extend(pdf_chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
                
        return chunks
    
    async def process_single_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a single PDF file using late chunking"""
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []
        
        # Apply late chunking
        chunks, span_annotations = self._chunk_by_sentences(text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, (chunk, span) in enumerate(zip(chunks, span_annotations)):
            if chunk.strip():  # Skip empty chunks
                chunk_obj = DocumentChunk(
                    content=chunk.strip(),
                    metadata={
                        "source": str(pdf_path.name),
                        "chunk_index": i,
                        "span_start": span[0],
                        "span_end": span[1],
                        "total_chunks": len(chunks)
                    }
                )
                document_chunks.append(chunk_obj)
        
        logger.info(f"Created {len(document_chunks)} chunks from {pdf_path.name}")
        return document_chunks
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
        return text
    
    def _chunk_by_sentences(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split text into chunks using late chunking strategy
        Based on the provided notebook implementation
        """
        inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        
        # Find sentence boundaries
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                i + 1 < len(token_offsets) and
                (token_offsets[i + 1][0] - token_offsets[i][1] > 0
                 or token_ids[i + 1] == sep_id)
            )
        ]
        
        if not chunk_positions:
            # Fallback: split by max length if no sentence boundaries found
            return self._fallback_chunking(input_text)
        
        # Create chunks based on positions
        chunks = [
            input_text[x[1]: y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        
        span_annotations = [
            (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        
        return chunks, span_annotations
    
    def _fallback_chunking(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Fallback chunking method when sentence boundaries aren't found"""
        chunk_size = settings.MAX_CHUNK_LENGTH
        overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        spans = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                while end > start and text[end] not in [' ', '\n', '\t']:
                    end -= 1
                if end == start:  # No word boundary found
                    end = min(start + chunk_size, len(text))
            
            chunks.append(text[start:end])
            spans.append((start, end))
            start = max(start + chunk_size - overlap, end)
            
        return chunks, spans
