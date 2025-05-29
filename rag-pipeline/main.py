# ==============================================================================
# FILE: main.py
# ==============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import List

from models.chat import ChatRequest, ChatResponse
from models.document import FinePrint
from services.document_processor import DocumentProcessor
from services.vector_store_factory import create_vector_store
from services.llm_service import LLMService
from services.fine_prints_extractor import FinePrintsExtractor
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
document_processor: DocumentProcessor = None
vector_store = None
llm_service: LLMService = None
fine_prints_extractor: FinePrintsExtractor = None

@asynccontextmanager  
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global document_processor, vector_store, llm_service, fine_prints_extractor
    
    logger.info("Initializing RAG Pipeline services...")
    
    try:
        # Initialize document processor
        document_processor = DocumentProcessor()
        logger.info("✓ Document processor initialized")
        
        # Initialize vector store (with fallback)
        vector_store = create_vector_store()
        logger.info("✓ Vector store initialized")
        
        # Initialize LLM service (may fail if no API key)
        try:
            llm_service = LLMService()
            logger.info("✓ LLM service initialized")
        except Exception as e:
            logger.error(f"✗ LLM service initialization failed: {e}")
            logger.warning("Chat and fine-prints extraction will not be available")
            llm_service = None
        
        # Initialize fine prints extractor (depends on LLM service)
        if llm_service:
            fine_prints_extractor = FinePrintsExtractor(llm_service)
            logger.info("✓ Fine-prints extractor initialized")
        else:
            fine_prints_extractor = None
            logger.warning("✗ Fine-prints extractor not available (no LLM service)")
        
        # Process documents and populate vector store
        await initialize_knowledge_base()
        
        logger.info("🚀 RAG Pipeline initialization completed!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        logger.warning("Starting with limited functionality")
    
    yield
    
    # Cleanup
    if vector_store:
        await vector_store.close()

async def initialize_knowledge_base():
    """Process PDFs and populate vector database"""
    try:
        logger.info("Processing PDF documents...")
        # processed_docs = await document_processor.process_pdf_directory(settings.PDF_PATH)
        
        # if not processed_docs:
        #     logger.warning("No documents were processed")
        #     return
        
        # logger.info(f"Processed {len(processed_docs)} document chunks")
        
        # # Store in vector database
        # await vector_store.add_documents(processed_docs)
        # logger.info("Documents stored in vector database successfully")
        
        # Log vector store stats if available
        if hasattr(vector_store, 'get_stats'):
            stats = vector_store.get_stats()
            logger.info(f"Vector store stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
        # Don't raise - allow app to start with limited functionality
        logger.warning("Continuing without knowledge base")

app = FastAPI(
    title="RAG Pipeline for Project Proposals",
    description="A RAG pipeline using late chunking strategy to assist in drafting project proposals",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fine-prints", response_model=List[FinePrint])
async def get_fine_prints():
    """
    Extract and return fine-prints (key details) from all documents.
    These are critical details needed for drafting project proposals.
    """
    try:
        if not fine_prints_extractor:
            raise HTTPException(status_code=503, detail="Fine-prints extractor not available")
            
        fine_prints = await fine_prints_extractor.extract_fine_prints()
        return fine_prints
    except Exception as e:
        logger.error(f"Error extracting fine-prints: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting fine-prints: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint to interact with documents and answer user queries.
    Uses RAG to provide context-aware responses.
    """
    try:
        if not vector_store or not llm_service:
            raise HTTPException(status_code=503, detail="Chat service not available")
        
        # Retrieve relevant documents
        relevant_docs = await vector_store.similarity_search(request.query, k=5)
        
        # Generate response using LLM with context
        response = await llm_service.generate_response(
            query=request.query,
            context=relevant_docs,
            chat_history=request.chat_history
        )
        
        return ChatResponse(
            response=response,
            sources=[doc.metadata.get("source", "unknown") for doc in relevant_docs]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "message": "RAG Pipeline is running",
        "vector_store": type(vector_store).__name__ if vector_store else "Not available",
        "services": {
            "document_processor": document_processor is not None,
            "vector_store": vector_store is not None,
            "llm_service": llm_service is not None,
            "fine_prints_extractor": fine_prints_extractor is not None
        }
    }
    return status

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "vector_store_type": type(vector_store).__name__ if vector_store else "None"
    }
    
    if vector_store and hasattr(vector_store, 'get_stats'):
        stats.update(vector_store.get_stats())
    
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


