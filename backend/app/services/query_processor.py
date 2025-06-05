import logging
from typing import List, Dict, Any
import google.generativeai as genai

from ..core.config import settings
from ..models import schemas
from .vector_db_service import VectorDBService
from .document_processor import DocumentProcessorService # For generating query embeddings

logger = logging.getLogger(__name__)

class QueryProcessorService:
    def __init__(self, 
                 vector_db_service: VectorDBService, 
                 doc_processor_service: DocumentProcessorService):
        """
        Initializes the QueryProcessorService.

        Args:
            vector_db_service: An instance of VectorDBService.
            doc_processor_service: An instance of DocumentProcessorService (for query embedding).
        """
        self.vector_db = vector_db_service
        self.doc_processor = doc_processor_service
        
        # Check the API key from settings, not directly from genai module
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error("Google AI client not configured (API key missing or placeholder). Query processing will be impacted.")
            self.chat_model = None # Ensure chat_model is None if key is missing
        else:
            # Attempt to configure genai if not already done (though it's usually done at module/app start)
            # This is more of a safeguard; primary configuration should be central.
            # DocumentProcessorService already configures genai at module level if key is present.
            # genai.configure can be called multiple times.
            try:
                # Ensure configuration before initializing model, in case it wasn't done elsewhere
                # or if this service is instantiated before DocumentProcessor's module-level configure runs.
                genai.configure(api_key=settings.gemini_api_key)
                # Initialize the Gemini chat model
                self.chat_model = genai.GenerativeModel(settings.gemini_chat_model_id)
                logger.info(f"Gemini chat model '{settings.gemini_chat_model_id}' initialized for QueryProcessorService.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini chat model '{settings.gemini_chat_model_id}': {e}")
                self.chat_model = None


    async def generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Generates an embedding for the given query text.
        Uses the RETRIEVAL_QUERY task type for embeddings optimized for querying.
        """
        # Check the API key from settings
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error("Google AI client not configured (API key missing or placeholder). Cannot generate query embedding.")
            return []
        
        # Ensure genai is configured (safeguard)
        # genai.configure can be called multiple times.
        # It's assumed to be configured by DocumentProcessorService or application startup.
        # If not, the embed_content call will fail, which is handled by the try-except.
        try:
            # Note: doc_processor.get_embeddings uses RETRIEVAL_DOCUMENT.
            # For queries, it's better to use RETRIEVAL_QUERY.
            result = genai.embed_content(
                model=f"models/{settings.gemini_embedding_model_id}",
                content=query_text,
                task_type="RETRIEVAL_QUERY" # Important for query embeddings
            )
            embedding = result.get('embedding', []) if isinstance(result, dict) else result.embedding
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                 # If 'embedding' is a list of lists (for batch), and we sent one query, take the first.
                 # If it's already a flat list (single embedding), use as is.
                return embedding[0] if isinstance(embedding[0], list) else embedding
            else:
                logger.error(f"Failed to generate a valid query embedding for: {query_text}. Result: {result}")
                return []
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    async def extract_answer_from_chunk(self, query_text: str, chunk_text: str) -> str:
        """
        Uses the Gemini chat model to extract a concise answer from a text chunk
        based on the user's query.
        """
        if not self.chat_model:
            logger.error("Chat model not initialized. Cannot extract answer.")
            return "Error: Chat model not available."

        prompt = f"""
        User Query: "{query_text}"

        Context from Document:
        ---
        {chunk_text}
        ---

        Based *only* on the provided "Context from Document", answer the "User Query" concisely.
        If the context does not contain information to answer the query, respond with "Information not found in this document segment."
        Do not use any external knowledge.
        """
        try:
            response = await self.chat_model.generate_content_async(prompt)
            # Ensure response.text or equivalent attribute access
            answer = response.text if hasattr(response, 'text') else response.parts[0].text if response.parts else "Error: Could not parse LLM response."
            return answer.strip()
        except Exception as e:
            logger.error(f"Error extracting answer with Gemini: {e}")
            return "Error: Could not extract answer due to an LLM issue."

    async def process_query(self, query_text: str, n_results: int) -> List[schemas.IndividualAnswer]:
        """
        Processes a user query:
        1. Generates query embedding.
        2. Retrieves relevant chunks from VectorDB.
        3. Extracts answers from each chunk using LLM.
        4. Formats and returns individual answers.
        """
        logger.info(f"Processing query: '{query_text}'")

        query_embedding = await self.generate_query_embedding(query_text)
        if not query_embedding:
            logger.error("Failed to generate query embedding. Cannot proceed.")
            return []

        relevant_chunks = self.vector_db.query_documents(
            query_embedding=query_embedding,
            n_results=n_results
        )

        if not relevant_chunks:
            logger.info("No relevant document chunks found in VectorDB.")
            return []
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks from VectorDB.")

        individual_answers: List[schemas.IndividualAnswer] = []
        for i, chunk_data in enumerate(relevant_chunks):
            text_chunk_content = chunk_data.get("text_chunk")
            doc_id = chunk_data.get("metadata", {}).get("doc_id", chunk_data.get("chunk_id", "UnknownDoc").split("_chunk_")[0]) # Fallback for doc_id
            metadata = chunk_data.get("metadata", {})
            
            logger.debug(f"Extracting answer from chunk {i+1}/{len(relevant_chunks)} (ID: {chunk_data.get('chunk_id')}) for doc '{doc_id}'")

            if not text_chunk_content:
                logger.warning(f"Skipping chunk {chunk_data.get('chunk_id')} due to missing text content.")
                continue

            extracted_answer_text = await self.extract_answer_from_chunk(query_text, text_chunk_content)
            
            # Basic citation (can be improved)
            citation = f"Document ID: {doc_id}"
            if metadata.get("source_file"):
                citation += f" (Source: {metadata['source_file']}"
                if metadata.get("source_row_number"):
                    citation += f", Row: {metadata['source_row_number']}"
                citation += ")"
            if metadata.get("chunk_number") is not None:
                 citation += f", Chunk: {metadata['chunk_number']}"


            individual_answers.append(schemas.IndividualAnswer(
                doc_id=str(doc_id),
                extracted_answer=extracted_answer_text,
                citation_source=citation,
                text_chunk=text_chunk_content,
                metadata=schemas.DocumentMetadata(**metadata), # Convert dict to Pydantic model
                relevance_score=chunk_data.get("distance") # ChromaDB returns distance, lower is better
            ))
        
        logger.info(f"Generated {len(individual_answers)} individual answers for query '{query_text}'.")
        return individual_answers

# Example Usage (for local testing - requires services to be set up)
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # This example requires running services and a populated DB.
    # For a standalone test, you'd mock VectorDBService and DocumentProcessorService.
    logger.info("--- QueryProcessorService Example (requires running services and data) ---")
    
    # Mock services for this example if not running full app
    class MockVectorDB:
        def query_documents(self, query_embedding, n_results, filter_metadata=None):
            logger.info(f"MockVectorDB: query_documents called with n_results={n_results}")
            return [
                {"chunk_id": "MOCK001_chunk0", "text_chunk": "The sky is blue due to Rayleigh scattering.", "metadata": {"doc_id": "MOCK001", "source_file": "science.txt", "chunk_number": 0}, "distance": 0.1},
                {"chunk_id": "MOCK002_chunk0", "text_chunk": "Blueberries are a type of fruit.", "metadata": {"doc_id": "MOCK002", "source_file": "foods.txt", "chunk_number": 0}, "distance": 0.5}
            ]

    class MockDocProcessor:
        async def get_embeddings(self, texts: List[str], task_type: str) -> List[List[float]]: # Adjusted for task_type
            logger.info(f"MockDocProcessor: get_embeddings called for task_type {task_type}")
            # Assume Gemini embedding dimension is 768
            return [[0.1] * 768 for _ in texts] # Return dummy embeddings

        async def generate_query_embedding(self, query_text: str) -> List[float]:
             logger.info(f"MockDocProcessor: generate_query_embedding for '{query_text}'")
             return [0.1] * 768


    # Initialize with mocks
    mock_vector_db = MockVectorDB()
    mock_doc_processor = DocumentProcessorService() # Using real one for query embedding generation if API key is set
                                                    # or replace with MockDocProcessor for full mock

    if not genai.API_KEY:
        logger.error("GEMINI_API_KEY not set. This example will likely fail or use mocked LLM calls.")
        # Fallback to a fully mocked doc_processor if API key is missing
        mock_doc_processor_for_query_embedding = MockDocProcessor()
        query_processor = QueryProcessorService(mock_vector_db, mock_doc_processor_for_query_embedding)
    else:
        query_processor = QueryProcessorService(mock_vector_db, DocumentProcessorService())


    async def run_example_query():
        test_query = "Why is the sky blue?"
        logger.info(f"\nTesting query: '{test_query}'")
        answers = await query_processor.process_query(test_query, n_results=2)
        if answers:
            for ans in answers:
                logger.info(f"  Doc ID: {ans.doc_id}, Answer: '{ans.extracted_answer[:100]}...', Citation: {ans.citation_source}, Score: {ans.relevance_score}")
        else:
            logger.info("  No answers returned for the test query.")

    asyncio.run(run_example_query())
    logger.info("--- QueryProcessorService Example Finished ---")