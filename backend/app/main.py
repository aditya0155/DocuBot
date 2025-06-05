import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .core.config import settings
from .services.document_ingestor import DocumentIngestorService
from .services.document_processor import DocumentProcessorService
from .services.vector_db_service import VectorDBService
# Import API routers
from .api.query_api import router as query_router
from .api.documents_api import router as documents_router # Placeholder for future document management API

# Configure logging
# Basic configuration, can be expanded with handlers, formatters, etc.
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global instances of services - can also be managed with dependency injection
# For simplicity in a smaller app, global instances can be okay.
# For larger apps, consider FastAPI's Depends system more deeply.

# Import the service instance variables from dependencies.py
# These will be populated during the lifespan event.
from . import dependencies as deps 
# Also import service classes for type hinting if needed, though deps might cover it.
from .services.document_ingestor import DocumentIngestorService
from .services.document_processor import DocumentProcessorService
from .services.vector_db_service import VectorDBService
from .services.query_processor import QueryProcessorService
from .services.theme_analyzer import ThemeAnalyzerService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Application startup...")
    # Services will be initialized and assigned to variables in deps module
    
    # Initialize services
    # The CSV path should ideally come from config or be more dynamic
    # For now, hardcoding the known CSV path relative to the project root.
    # The DocumentIngestorService expects path relative to where it's run or absolute.
    # Constructing a path relative to this main.py file to reach the project root.
    # main.py is in backend/app/
    # We will assume legal_text_classification.csv is in backend/data/
    import os
    current_module_dir = os.path.dirname(os.path.abspath(__file__)) # backend/app
    backend_dir = os.path.dirname(current_module_dir) # backend
    csv_file_path = os.path.join(backend_dir, "data", "legal_text_classification.csv")
    # Alternatively, make this path configurable via settings.CSV_FILE_PATH
    # and set it in Vercel environment variables.
    # For now, using a path relative to main.py that should work if CSV is in backend/data/.
    logger.info(f"Attempting to load CSV from calculated path: {csv_file_path}")
    
    # Assign initialized services to the variables in the dependencies module
    deps.document_ingestor_service = DocumentIngestorService(csv_file_path=csv_file_path)
    deps.document_processor_service = DocumentProcessorService()
    deps.vector_db_service = VectorDBService()

    # Initial CSV loading logic has been removed.
    # The database will start empty unless data already exists from previous runs.
    # Users will populate the database by uploading documents via the UI or API.
    logger.info("Initial CSV auto-loading on startup is disabled.")
    if deps.vector_db_service.collection: # Use deps.vector_db_service
        logger.info(f"Current items in DB: {deps.vector_db_service.get_collection_count()}")
    else:
        logger.warning("VectorDB collection not available on startup.")

    # Initialize other services that depend on the base ones
    deps.query_processor_service = QueryProcessorService(
        vector_db_service=deps.vector_db_service, # Use deps.
        doc_processor_service=deps.document_processor_service # Use deps.
    )
    deps.theme_analyzer_service = ThemeAnalyzerService()
    logger.info("QueryProcessorService and ThemeAnalyzerService globally initialized in dependencies module.")

    logger.info("All application services initialized.")
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")
    # Add any cleanup tasks here if needed (e.g., closing database connections if not handled by clients)


app = FastAPI(
    title=settings.project_name,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    lifespan=lifespan # Use the lifespan context manager
)

# --- API Routers ---
app.include_router(query_router, prefix=settings.api_v1_prefix, tags=["Query"])
app.include_router(documents_router, prefix=settings.api_v1_prefix, tags=["Documents"])


# Placeholder for future documents API router
# from .api.documents_api import router as documents_router # Create this file later
# app.include_router(documents_router, prefix=settings.api_v1_prefix, tags=["Documents"])

# --- Dependency Provider functions are now in dependencies.py ---
# No longer need to define them here.
# from .dependencies import get_vector_db_serv, get_doc_processor_serv, get_query_processor_serv, get_theme_analyzer_serv

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint for health check.
    """
    logger.info("Root endpoint accessed.")
    return {"message": f"Welcome to {settings.project_name}!"}

# Placeholder for an endpoint to manually trigger ingestion if needed for testing
# This would ideally be more secure or part of an admin interface
@app.post("/api/v1/admin/ingest-csv", tags=["Admin"], status_code=202)
async def trigger_csv_ingestion():
    """
    Manually triggers the CSV ingestion process.
    Useful for development or if initial startup ingestion fails.
    """
    logger.info("Manual CSV ingestion triggered via API.")
    try:
        # Use services from deps module
        if deps.vector_db_service and deps.vector_db_service.collection:
            logger.info("Clearing existing vector DB collection before manual ingestion.")
            deps.vector_db_service.clear_collection()
        
        raw_docs = deps.document_ingestor_service.ingest_documents_from_csv()
        if not raw_docs:
            raise HTTPException(status_code=400, detail="No documents ingested from CSV or CSV not found/readable.")
        
        logger.info(f"Ingested {len(raw_docs)} raw documents.")
        processed_chunks = deps.document_processor_service.process_documents(raw_docs)
        
        if not processed_chunks:
            raise HTTPException(status_code=500, detail="No chunks processed from ingested documents.")
            
        logger.info(f"Processed into {len(processed_chunks)} chunks.")
        deps.vector_db_service.add_documents(processed_chunks)
        
        return {
            "message": "CSV ingestion process initiated and completed.",
            "documents_ingested": len(raw_docs),
            "chunks_created_and_stored": len(processed_chunks),
            "db_items_total": deps.vector_db_service.get_collection_count()
        }
    except FileNotFoundError as fnf_e:
        logger.error(f"CSV file not found during manual ingestion: {fnf_e}")
        raise HTTPException(status_code=404, detail=f"CSV file not found: {deps.document_ingestor_service.csv_file_path}")
    except Exception as e:
        logger.error(f"Error during manual CSV ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during CSV ingestion: {str(e)}")


if __name__ == "__main__":
    # This block is for direct execution (e.g., `python main.py`)
    # Uvicorn is typically used to run FastAPI apps in production or development.
    # Example: uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
    import uvicorn
    logger.info(f"Starting Uvicorn server directly from main.py on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())