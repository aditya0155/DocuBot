import logging
from typing import Optional

# Import service classes
from .services.vector_db_service import VectorDBService
from .services.document_processor import DocumentProcessorService
from .services.query_processor import QueryProcessorService
from .services.theme_analyzer import ThemeAnalyzerService
from .services.document_ingestor import DocumentIngestorService # Added for completeness if needed elsewhere

logger = logging.getLogger(__name__)

# Global service instances - these will be initialized by main.py's lifespan
# They are defined here so other modules can import them and type hint against them,
# and so provider functions can return them.

document_ingestor_service: Optional[DocumentIngestorService] = None
document_processor_service: Optional[DocumentProcessorService] = None
vector_db_service: Optional[VectorDBService] = None
query_processor_service: Optional[QueryProcessorService] = None
theme_analyzer_service: Optional[ThemeAnalyzerService] = None

# --- Dependency Provider Functions ---
# These functions will be used by FastAPI's Depends in the API routers.
# They return the globally initialized instances.

def get_document_ingestor_serv() -> DocumentIngestorService:
    if document_ingestor_service is None:
        logger.error("DocumentIngestorService not initialized.")
        raise RuntimeError("DocumentIngestorService not available.")
    return document_ingestor_service

def get_vector_db_serv() -> VectorDBService:
    if vector_db_service is None:
        logger.error("VectorDBService not initialized.")
        raise RuntimeError("VectorDBService not available.")
    return vector_db_service

def get_doc_processor_serv() -> DocumentProcessorService:
    if document_processor_service is None:
        logger.error("DocumentProcessorService not initialized.")
        raise RuntimeError("DocumentProcessorService not available.")
    return document_processor_service

def get_query_processor_serv() -> QueryProcessorService:
    if query_processor_service is None:
        logger.error("QueryProcessorService not initialized.")
        raise RuntimeError("QueryProcessorService not available.")
    return query_processor_service

def get_theme_analyzer_serv() -> ThemeAnalyzerService:
    if theme_analyzer_service is None:
        logger.error("ThemeAnalyzerService not initialized.")
        raise RuntimeError("ThemeAnalyzerService not available.")
    return theme_analyzer_service

logger.info("Dependency providers defined in dependencies.py")