import logging
import re
from typing import List, Dict, Any, Iterator
import google.generativeai as genai
# from google.generativeai.types import TaskType # Removed this import
from ..core.config import settings
from ..core.exceptions import DocumentProcessingError, LLMError # Import custom exceptions

logger = logging.getLogger(__name__)

# Configure Google Gemini client
try:
    if settings.gemini_api_key and settings.gemini_api_key != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=settings.gemini_api_key)
        # Test with a simple model listing to ensure API key is valid, if possible, or rely on first use.
        # For now, we assume configuration is successful if API key is present.
        logger.info("Google Generative AI client configured successfully.")
    else:
        logger.error("GEMINI_API_KEY is not set or is using the placeholder. Google AI client not configured.")
        # This will cause issues later, but we'll let the embedding call fail to see the error.
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI client: {e}")
    # Handle as appropriate, e.g., by not allowing embedding generation.

class DocumentProcessorService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initializes the DocumentProcessorService.

        Args:
            chunk_size (int): The target size for text chunks (e.g., in characters or tokens).
                               This is less relevant for sentence-based chunking but kept for potential future use.
            chunk_overlap (int): The overlap between consecutive chunks (for fixed-size chunking).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # The genai.configure call is handled at the module level.
        # If settings.gemini_api_key was missing or invalid, an error is logged there.
        # No direct genai.API_KEY attribute exists to check here.

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning. Can be expanded.
        - Removes multiple spaces.
        - Removes leading/trailing whitespace.
        """
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
        return text.strip()

    def chunk_text_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> Iterator[str]:
        """
        Chunks text by a specified number of sentences.
        More robust sentence tokenization might be needed (e.g., NLTK).
        """
        # Simple sentence splitting by common delimiters. This is basic.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk_sentences = []
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            if (i + 1) % sentences_per_chunk == 0 or i == len(sentences) - 1:
                yield " ".join(current_chunk_sentences)
                current_chunk_sentences = []
    
    def chunk_text_fixed_size(self, text: str) -> List[str]:
        """
        Chunks text into fixed-size, potentially overlapping chunks.
        This is a very basic character-based chunker.
        Token-based chunking (e.g., using tiktoken) would be more aligned with LLM context windows.
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += (self.chunk_size - self.chunk_overlap)
            if start >= len(text): # Ensure we don't create an empty chunk if overlap is large
                break
        return chunks


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text chunks using the Google Gemini API.
        """
        # Check for API key in settings, as genai.configure might not raise an immediate error
        # for a missing key but will fail on first API call.
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            msg = "GEMINI_API_KEY is not properly set in settings. Cannot generate embeddings."
            logger.error(msg)
            raise DocumentProcessingError(message=msg, details="Missing or placeholder API key.")

        all_embeddings = [None] * len(texts) # Initialize with None for each text

        # Check for API key in settings
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            msg = "GEMINI_API_KEY is not properly set in settings. Cannot generate embeddings."
            logger.error(msg)
            raise DocumentProcessingError(message=msg, details="Missing or placeholder API key.")

        try:
            # Attempt batch embedding first
            logger.info(f"Attempting batch embedding for {len(texts)} texts using model models/{settings.gemini_embedding_model_id}.")
            result = genai.embed_content(
                model=f"models/{settings.gemini_embedding_model_id}",
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            if 'embedding' in result and isinstance(result['embedding'], list) and len(result['embedding']) == len(texts):
                all_embeddings = result['embedding']
                logger.info(f"Successfully generated {len(all_embeddings)} embeddings in batch.")
                return all_embeddings # Return early if batch succeeds
            else:
                logger.warning(f"Batch embedding did not return expected structure or count. Expected {len(texts)}, got result: {result}. Will attempt individual embeddings.")
                # Proceed to individual embedding fallback

        except Exception as batch_exception:
            logger.warning(f"Batch embedding failed for {len(texts)} texts: {batch_exception}. Attempting individual embeddings.", exc_info=True)
            # Fall through to individual embedding attempts

        # Fallback to individual embeddings if batch failed or had unexpected results
        successful_embeddings_count = 0
        for i, text_content in enumerate(texts):
            try:
                logger.debug(f"Attempting individual embedding for text {i+1}/{len(texts)} (length: {len(text_content)} chars).")
                # Ensure text_content is not empty, as some models might reject empty strings
                if not text_content.strip():
                    logger.warning(f"Skipping empty text content at index {i} for individual embedding.")
                    all_embeddings[i] = [] # Or handle as appropriate, e.g., a zero vector of correct dimensionality if known
                    continue

                individual_result = genai.embed_content(
                    model=f"models/{settings.gemini_embedding_model_id}",
                    content=text_content, # Single text content
                    task_type="RETRIEVAL_DOCUMENT"
                )
                if 'embedding' in individual_result and isinstance(individual_result['embedding'], list):
                    all_embeddings[i] = individual_result['embedding']
                    successful_embeddings_count += 1
                else:
                    logger.error(f"Unexpected individual embedding result for text {i+1}: {individual_result}. Storing None.")
                    all_embeddings[i] = None # Mark as failed
            except Exception as individual_exception:
                logger.error(f"Individual embedding failed for text {i+1}/{len(texts)}: {individual_exception}", exc_info=True)
                all_embeddings[i] = None # Mark as failed
        
        logger.info(f"Individual embedding attempts completed. Successfully embedded {successful_embeddings_count}/{len(texts)} texts.")
        
        # Check if all embeddings failed in the fallback
        if successful_embeddings_count == 0 and texts:
            # If all individual attempts failed, raise an error to signal complete failure for this set of texts.
            # The calling function (process_documents) will then skip the document.
            msg = f"All {len(texts)} individual embedding attempts failed."
            logger.error(msg)
            raise LLMError(message=msg, details="Refer to individual embedding failure logs above.")
            
        return all_embeddings


    def process_documents(self, raw_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of raw documents: cleans text, chunks, generates embeddings.

        Args:
            raw_documents (List[Dict[str, Any]]): List of documents from DocumentIngestorService.
                Each dict should have 'doc_id', 'text', and 'metadata'.

        Returns:
            List[Dict[str, Any]]: List of processed chunks, each with 'chunk_id', 
                                  'doc_id', 'text_chunk', 'embedding', and 'metadata'.
        """
        processed_chunks = []
        # Check for API key in settings before processing
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            msg = "GEMINI_API_KEY is not properly set in settings. Cannot process documents."
            logger.error(msg)
            raise DocumentProcessingError(message=msg, details="Missing or placeholder API key for document processing.")
            # Caller should handle this, or the application won't start if this is critical path.

        for i, doc in enumerate(raw_documents):
            doc_id = doc.get("doc_id")
            text_content = doc.get("text")
            metadata = doc.get("metadata", {})
            source_file = doc.get("source_file", "N/A")
            source_row = doc.get("source_row_number", -1)

            if not doc_id or not text_content:
                logger.warning(f"Skipping document with missing ID or text content (Source: {source_file}, Row: {source_row}).")
                continue

            cleaned_text = self.clean_text(text_content)
            
            # Using sentence-based chunking for potentially better semantic coherence
            # text_chunks_content = self.chunk_text_fixed_size(cleaned_text) 
            text_chunks_content = list(self.chunk_text_by_sentences(cleaned_text, sentences_per_chunk=5))


            if not text_chunks_content:
                logger.warning(f"Document {doc_id} resulted in no text chunks after cleaning/chunking. Skipping.")
                continue
            
            logger.info(f"Processing document {i+1}/{len(raw_documents)}: '{doc_id}' from {source_file} (Row {source_row}). Cleaned length: {len(cleaned_text)}, Chunks: {len(text_chunks_content)}")

            try:
                chunk_embeddings_list = self.get_embeddings(text_chunks_content) # This now returns a list, potentially with Nones
            except LLMError as e: 
                logger.error(f"Failed to get any embeddings for document {doc_id} due to LLMError: {e.message}. Details: {e.details}. Skipping document.")
                continue 
            except Exception as e: 
                logger.error(f"Unexpected error getting embeddings for document {doc_id}: {e}. Skipping document.", exc_info=True)
                continue

            # Iterate through chunks and their corresponding embeddings (which might be None)
            processed_doc_chunks_count = 0
            for chunk_index, chunk_text in enumerate(text_chunks_content):
                embedding = chunk_embeddings_list[chunk_index] if chunk_index < len(chunk_embeddings_list) else None
                
                if embedding: # Only process if embedding was successful
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    processed_chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text_chunk": chunk_text,
                        "embedding": embedding,
                        "metadata": {
                            **metadata, 
                            "source_file": source_file,
                            "source_row_number": source_row,
                            "chunk_number": chunk_index 
                        } 
                    })
                    processed_doc_chunks_count +=1
                else:
                    logger.warning(f"Skipping chunk {chunk_index} of document {doc_id} due to failed embedding generation for this specific chunk.")
            
            if processed_doc_chunks_count > 0:
                 logger.info(f"Successfully processed {processed_doc_chunks_count}/{len(text_chunks_content)} chunks for document {doc_id}.")
            elif text_chunks_content: # If there were chunks but none were embedded
                 logger.warning(f"No chunks were successfully embedded for document {doc_id} out of {len(text_chunks_content)} potential chunks.")


            if (i + 1) % 100 == 0: 
                logger.info(f"Attempted processing for {i+1} documents...")


        logger.info(f"Successfully processed {len(raw_documents)} raw documents into {len(processed_chunks)} chunks.")
        return processed_chunks

# Example Usage (for local testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock raw documents (as if from DocumentIngestorService)
    sample_raw_docs = [
        {
            "doc_id": "SAMPLE001", 
            "text": "This is the first sentence. This is the second sentence. And a third one here! Fourth sentence follows. Fifth sentence makes a paragraph. Sixth sentence starts a new one. Seventh is short. Eighth is the last for this chunk.", 
            "metadata": {"case_title": "Sample Case 1", "case_outcome": "Won"},
            "source_file": "dummy.csv",
            "source_row_number": 2
        },
        {
            "doc_id": "SAMPLE002", 
            "text": "Another document. It has fewer sentences. Just three. This is the third.", 
            "metadata": {"case_title": "Sample Case 2", "case_outcome": "Lost"},
            "source_file": "dummy.csv",
            "source_row_number": 3
        },
        {
            "doc_id": "SAMPLE003",
            "text": "", # Empty text
            "metadata": {"case_title": "Empty Case", "case_outcome": "N/A"},
            "source_file": "dummy.csv",
            "source_row_number": 4
        }
    ]

    processor = DocumentProcessorService(chunk_size=50, chunk_overlap=10) # chunk_size/overlap less relevant for sentence chunking
    
    if not genai.API_KEY:
        logger.error("Cannot run example: Google AI client not configured. Check GEMINI_API_KEY in .env")
    else:
        logger.info(f"Using Google Gemini embedding model: models/{settings.gemini_embedding_model_id}")
        processed_data = processor.process_documents(sample_raw_docs)
        
        if processed_data:
            logger.info(f"Processed {len(processed_data)} chunks.")
            for chunk_data in processed_data:
                logger.info(
                    f"Chunk ID: {chunk_data['chunk_id']}, Doc ID: {chunk_data['doc_id']}, "
                    f"Text: '{chunk_data['text_chunk'][:50]}...', "
                    f"Embedding len: {len(chunk_data['embedding']) if chunk_data['embedding'] else 0}, "
                    f"Metadata: {chunk_data['metadata']}"
                )
        else:
            logger.warning("No data was processed in the example.")
            
    logger.info("DocumentProcessorService example finished.")