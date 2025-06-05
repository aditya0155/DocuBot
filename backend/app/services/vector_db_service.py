import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from ..core.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        """
        Initializes the VectorDBService.
        Connects to ChromaDB and gets or creates the specified collection.
        """
        try:
            # Using a persistent client that saves to disk
            self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
            logger.info(f"ChromaDB client initialized. Data will be persisted at: {settings.chroma_db_path}")

            # Get or create the collection.
            # We are generating embeddings externally with Gemini, so we don't need
            # ChromaDB's embedding function here. We'll pass embeddings directly.
            # However, if we wanted Chroma to use Gemini, we'd need a custom embedding function.
            # For now, we manage embeddings ourselves.
            self.collection_name = settings.chroma_db_collection_name
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
                # embedding_function can be omitted if you always provide embeddings
            )
            logger.info(f"Connected to ChromaDB collection: '{self.collection_name}'")
            logger.info(f"Current number of items in collection: {self.collection.count()}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}")
            self.client = None
            self.collection = None
            # Depending on desired robustness, could raise an exception to halt startup
            # raise RuntimeError(f"ChromaDB initialization failed: {e}")


    def add_documents(self, processed_chunks: List[Dict[str, Any]]):
        """
        Adds processed document chunks (with their embeddings) to the ChromaDB collection.

        Args:
            processed_chunks (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary represents a chunk and must contain 'chunk_id' (for id),
                'text_chunk' (for document), 'embedding', and 'metadata'.
        """
        if not self.collection:
            logger.error("ChromaDB collection is not available. Cannot add documents.")
            return

        if not processed_chunks:
            logger.info("No processed chunks provided to add to ChromaDB.")
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in processed_chunks:
            if not chunk.get("embedding"): # Skip chunks where embedding failed
                logger.warning(f"Chunk {chunk.get('chunk_id', 'N/A')} has no embedding. Skipping.")
                continue
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text_chunk"])
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])
        
        if not ids:
            logger.info("No valid chunks with embeddings to add to ChromaDB.")
            return

        try:
            # ChromaDB's add method can take lists for batch insertion.
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully added/updated {len(ids)} chunks in ChromaDB collection '{self.collection_name}'.")
            logger.info(f"Total items in collection now: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            # Consider how to handle partial failures if the batch add is not atomic.

    def query_documents(self, query_embedding: List[float], n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries the ChromaDB collection for similar documents based on a query embedding.

        Args:
            query_embedding (List[float]): The embedding vector for the query.
            n_results (int): The number of top similar results to retrieve.
            filter_metadata (Optional[Dict[str, Any]]): A dictionary for metadata filtering.
                                                       Example: {"source_file": "some_doc.pdf"}

        Returns:
            List[Dict[str, Any]]: A list of query results. Each result includes
                                  the document chunk, metadata, distance, etc.
        """
        if not self.collection:
            logger.error("ChromaDB collection is not available. Cannot query documents.")
            return []
        
        if not query_embedding:
            logger.error("Query embedding is empty. Cannot perform search.")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], # Query expects a list of embeddings
                n_results=n_results,
                where=filter_metadata if filter_metadata else None, # 'where' for metadata filtering
                include=['documents', 'metadatas', 'distances'] # Specify what to include in results
            )
            
            # The result structure for a single query embedding is a dict where keys like 'ids', 'documents'
            # map to lists of lists (one inner list per query embedding). Since we send one query, we take the first element.
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "chunk_id": results['ids'][0][i],
                        "text_chunk": results['documents'][0][i] if results.get('documents') else None,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') else None,
                        "distance": results['distances'][0][i] if results.get('distances') else None,
                    })
                logger.info(f"ChromaDB query returned {len(formatted_results)} results.")
            else:
                logger.info("ChromaDB query returned no results.")
            
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    def get_collection_count(self) -> int:
        """Returns the number of items in the collection."""
        if not self.collection:
            return 0
        return self.collection.count()

    def clear_collection(self):
        """
        Clears all documents from the collection. USE WITH CAUTION.
        This might involve deleting and recreating the collection if `clear` is not directly supported
        or if we want to ensure a completely fresh state.
        ChromaDB collections can be deleted by name.
        """
        if not self.client or not self.collection_name:
            logger.error("ChromaDB client or collection name not available. Cannot clear collection.")
            return
        try:
            logger.warning(f"Attempting to clear collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted.")
            # Recreate it empty
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' recreated and is empty. Count: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}")


# Example Usage (for local testing)
if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # This example requires the DocumentProcessorService and its dependencies (Google Gemini SDK)
    # to be functional for generating embeddings.
    # For simplicity, we'll mock some data here.

    logger.info("--- VectorDBService Example ---")
    vector_db_service = VectorDBService()

    if not vector_db_service.collection:
        logger.error("Failed to initialize VectorDBService for example. Exiting.")
    else:
        # 0. Clear collection for a fresh start in the example
        logger.info(f"Initial count: {vector_db_service.get_collection_count()}")
        vector_db_service.clear_collection()
        logger.info(f"Count after clearing: {vector_db_service.get_collection_count()}")

        # 1. Mock processed chunks (as if from DocumentProcessorService)
        #    In a real scenario, embeddings would be actual vectors.
        #    ChromaDB expects embeddings to match the dimensionality of what it's configured for
        #    or what was first inserted. For this example, let's assume a dummy dimension of 3.
        #    Gemini embeddings will have a specific dimension (e.g., 768 or 1024).
        #    This mock will likely fail if ChromaDB has existing data with different dimensions.
        #    The clear_collection() above helps mitigate this for the example run.
        
        # IMPORTANT: The dimension of these dummy embeddings MUST match what your
        # actual `gemini-embedding-exp-03-07` model produces.
        # Let's assume Gemini embedding dimension is 768 for this example.
        dummy_embedding_dim = 768 

        mock_processed_chunks = [
            {
                "chunk_id": "doc1_chunk0", "doc_id": "doc1", 
                "text_chunk": "The quick brown fox jumps over the lazy dog.", 
                "embedding": [0.1] * dummy_embedding_dim, # Dummy embedding
                "metadata": {"source_file": "file1.txt", "category": "animals"}
            },
            {
                "chunk_id": "doc1_chunk1", "doc_id": "doc1", 
                "text_chunk": "A lazy dog sat by the river.", 
                "embedding": [0.2] * dummy_embedding_dim, # Dummy embedding
                "metadata": {"source_file": "file1.txt", "category": "animals"}
            },
            {
                "chunk_id": "doc2_chunk0", "doc_id": "doc2", 
                "text_chunk": "Artificial intelligence is rapidly evolving.", 
                "embedding": [0.8] * dummy_embedding_dim, # Dummy embedding
                "metadata": {"source_file": "file2.txt", "category": "technology"}
            }
        ]
        vector_db_service.add_documents(mock_processed_chunks)
        logger.info(f"Count after adding documents: {vector_db_service.get_collection_count()}")

        # 2. Mock a query embedding
        mock_query_embedding = [0.15] * dummy_embedding_dim # Query related to dogs

        # 3. Query documents
        logger.info("\nQuerying for 'dogs':")
        query_results = vector_db_service.query_documents(mock_query_embedding, n_results=2)
        for res in query_results:
            logger.info(f"  Result: ID={res['chunk_id']}, Dist={res['distance']:.4f}, Text='{res['text_chunk'][:30]}...', Meta={res['metadata']}")

        # 4. Query with metadata filter
        logger.info("\nQuerying for 'technology' category:")
        # For metadata filter, the query embedding might be more generic or specific to tech
        mock_tech_query_embedding = [0.75] * dummy_embedding_dim 
        filtered_results = vector_db_service.query_documents(
            mock_tech_query_embedding, 
            n_results=2, 
            filter_metadata={"category": "technology"}
        )
        for res in filtered_results:
            logger.info(f"  Filtered Result: ID={res['chunk_id']}, Dist={res['distance']:.4f}, Text='{res['text_chunk'][:30]}...', Meta={res['metadata']}")
        
        logger.info("\n--- VectorDBService Example Finished ---")