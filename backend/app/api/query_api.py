import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from ..models import schemas # Pydantic models
from ..services.query_processor import QueryProcessorService
from ..services.theme_analyzer import ThemeAnalyzerService
# No longer need to import VectorDBService and DocumentProcessorService here directly for instantiation
# from ..services.vector_db_service import VectorDBService 
# from ..services.document_processor import DocumentProcessorService

# Import the dependency provider functions from main
# This creates a slight circular dependency potential if main also imports from api deeply,
# but for provider functions it's generally okay.
# A better way would be to have providers in a separate 'dependencies.py' file.
from ..dependencies import get_query_processor_serv, get_theme_analyzer_serv # Changed from ..main

logger = logging.getLogger(__name__)
router = APIRouter()

# No longer need local lazy initialization for these services
# _query_processor_service_instance: Optional[QueryProcessorService] = None
# _theme_analyzer_service_instance: Optional[ThemeAnalyzerService] = None

# def get_query_processor_service() -> QueryProcessorService:
#     global _query_processor_service_instance
#     if _query_processor_service_instance is None:
#         local_vector_db_service = VectorDBService()
#         local_document_processor_service = DocumentProcessorService()
#         if not local_vector_db_service.collection: 
#              logger.error("Failed to initialize VectorDBService for QueryProcessorService.")
#              raise RuntimeError("VectorDB service is not ready for query processing.")
#         _query_processor_service_instance = QueryProcessorService(
#             vector_db_service=local_vector_db_service,
#             doc_processor_service=local_document_processor_service
#         )
#         logger.info("QueryProcessorService instance created for query_api with local service instances.")
#     return _query_processor_service_instance

# def get_theme_analyzer_service() -> ThemeAnalyzerService:
#     global _theme_analyzer_service_instance
#     if _theme_analyzer_service_instance is None:
#         _theme_analyzer_service_instance = ThemeAnalyzerService()
#         logger.info("ThemeAnalyzerService instance created for query_api.")
#     return _theme_analyzer_service_instance
# --- End Service Initialization ---

@router.post("/query", response_model=schemas.QueryResponse)
async def handle_query(
    request: schemas.QueryRequest,
    qp_service: QueryProcessorService = Depends(get_query_processor_serv),
    ta_service: ThemeAnalyzerService = Depends(get_theme_analyzer_serv)
):
    """
    Handles a user's natural language query.
    1. Retrieves relevant document chunks.
    2. Extracts answers from each chunk.
    3. Identifies common themes across answers.
    4. Returns individual answers and synthesized themes.
    """
    logger.info(f"Received query: '{request.query_text}', n_results: {request.n_results}")

    try:
        # Services are now injected via Depends
        # qp_service = get_query_processor_service()
        # ta_service = get_theme_analyzer_service()

        # Step 1 & 2: Get individual answers
        individual_answers = await qp_service.process_query(
            query_text=request.query_text, 
            n_results=request.n_results
        )

        if not individual_answers:
            logger.info(f"No individual answers found for query: '{request.query_text}'")
            # Proceed to theme analysis, which should handle empty answers.
        
        # Step 3: Identify themes
        themes = await ta_service.analyze_themes(
            answers=individual_answers, 
            query_text=request.query_text
        )
        
        logger.info(f"Query processed. Found {len(individual_answers)} individual answers and {len(themes)} themes.")

        return schemas.QueryResponse(
            original_query=request.query_text,
            individual_answers=individual_answers,
            themes=themes
        )

    except HTTPException as http_exc:
        logger.error(f"HTTPException during query processing: {http_exc.detail}", exc_info=True)
        raise http_exc # Re-raise FastAPI's HTTPException
    except Exception as e:
        logger.error(f"Unexpected error processing query '{request.query_text}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# TODO: Add more endpoints as needed, e.g., for listing documents, document details, etc.