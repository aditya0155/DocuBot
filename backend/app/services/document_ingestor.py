import csv
import logging
import os
import sys
from typing import List, Dict, Any
from ..core.config import settings # Example if config is needed, may not be for this service directly
from ..core.exceptions import DocumentIngestionError # Import custom exception

logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by FastAPI/Uvicorn
# logging.basicConfig(level=settings.log_level)


class DocumentIngestorService:
    def __init__(self, csv_file_path: str):
        """
        Initializes the DocumentIngestorService.

        Args:
            csv_file_path (str): The path to the CSV file containing the documents.
        """
        self.csv_file_path = csv_file_path
        # Confirmed CSV path: legal_text_classification.csv (relative to project root)
        # Confirmed column mappings:
        self.text_column_name = "case_text"
        self.id_column_name = "case_id"
        # All other columns will be treated as metadata.
        # 'case_title' can be a source for keywords/catchphrases.
        # 'case_outcome' is specific case metadata.
        # No explicit 'citations' or 'catchphrases' columns were in the header.
        # We will define metadata_columns dynamically based on the CSV header, excluding id and text.
        self.dynamic_metadata_columns = [] # Will be populated after reading header

    def ingest_documents_from_csv(self) -> List[Dict[str, Any]]:
        """
        Reads documents from the CSV file, extracts text and metadata.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a document with its ID, text, and metadata.
                                  Example: [{"doc_id": "xyz", "text": "...", "metadata": {"catchphrases": "...", ...}}]
        """
        documents = []
        logger.info(f"Starting ingestion from CSV: {self.csv_file_path}")
        try:
            # Increase the field size limit for the CSV reader
            # Calculate current max int
            max_int = sys.maxsize
            while True:
                # decrease the max_int value by factor 10
                # as long as the OverflowError occurs.
                try:
                    csv.field_size_limit(max_int)
                    break
                except OverflowError:
                    max_int = int(max_int/10)
            logger.info(f"CSV field size limit set to {csv.field_size_limit()}")

            with open(self.csv_file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                if not reader.fieldnames:
                    logger.error(f"CSV file at {self.csv_file_path} appears to be empty or has no header.")
                    return []

                # Dynamically determine metadata columns
                self.dynamic_metadata_columns = [col for col in reader.fieldnames if col not in [self.id_column_name, self.text_column_name]]
                
                # Validate that essential ID and text columns exist
                if self.id_column_name not in reader.fieldnames:
                    logger.error(f"CSV file is missing the required ID column: '{self.id_column_name}'. Available columns: {reader.fieldnames}")
                    return []
                if self.text_column_name not in reader.fieldnames:
                    logger.error(f"CSV file is missing the required text column: '{self.text_column_name}'. Available columns: {reader.fieldnames}")
                    return []
                
                logger.info(f"Identified ID column: '{self.id_column_name}', Text column: '{self.text_column_name}'")
                logger.info(f"Identified metadata columns: {self.dynamic_metadata_columns}")

                for i, row in enumerate(reader):
                    try:
                        doc_id = row.get(self.id_column_name)
                        text_content = row.get(self.text_column_name)

                        if not doc_id:
                            logger.warning(f"Row {i+2} is missing document ID in column '{self.id_column_name}'. Skipping.")
                            continue
                        if not text_content:
                            logger.warning(f"Row {i+2} (ID: {doc_id}) is missing text content in column '{self.text_column_name}'. Skipping.")
                            continue

                        metadata = {}
                        for meta_col in self.dynamic_metadata_columns:
                            metadata[meta_col] = row.get(meta_col, "") # Get value or empty string if column is missing for a row (should not happen with DictReader)

                        documents.append({
                            "doc_id": str(doc_id), 
                            "text": str(text_content),
                            "metadata": metadata, # Contains all other columns like case_title, case_outcome
                            "source_file": os.path.basename(self.csv_file_path),
                            "source_row_number": i + 2 # i+2 because of header and 0-indexing
                        })
                        
                        if (i + 1) % 1000 == 0: # Log progress every 1000 documents
                            logger.info(f"Ingested {i+1} documents...")

                    except Exception as e:
                        logger.error(f"Error processing row {i+2} (ID: {row.get(self.id_column_name, 'N/A')}): {e}")
                        continue # Skip problematic rows but log them

            logger.info(f"Successfully ingested {len(documents)} documents from {self.csv_file_path}.")
        except FileNotFoundError as e:
            msg = f"CSV file not found at path: {self.csv_file_path}"
            logger.error(msg)
            raise DocumentIngestionError(message=msg, details=str(e))
        except Exception as e:
            msg = f"An unexpected error occurred during CSV ingestion from {self.csv_file_path}"
            logger.error(f"{msg}: {e}", exc_info=True)
            raise DocumentIngestionError(message=msg, details=str(e))
        
        return documents

# Example usage (for testing purposes, would typically be called by another service or API endpoint)
if __name__ == "__main__":
    import os
    # This example assumes a dummy CSV in the project root for testing.
    # In the actual application, the path will be configured.
    
    # Create a dummy CSV for testing that matches the user's header
    dummy_csv_path = "dummy_legal_text_classification.csv" 
    dummy_data = [
        {"case_id": "CASE001", "case_outcome": "Won", "case_title": "Contract Law Dispute", "case_text": "This is the first legal document about contract law."},
        {"case_id": "CASE002", "case_outcome": "Lost", "case_title": "IP Infringement Case", "case_text": "The second document discusses intellectual property."},
        {"case_id": "CASE003", "case_outcome": "Settled", "case_title": "Empty Text Example", "case_text": ""}, # Empty text
        {"case_id": "", "case_outcome": "Pending", "case_title": "No ID Example", "case_text": "Document with no ID."}, # No ID
        {"case_id": "CASE004", "case_outcome": "Won", "case_title": "Tort Law Negligence", "case_text": "This is about tort law."},
    ]
    
    fieldnames = ["case_id", "case_outcome", "case_title", "case_text"]

    with open(dummy_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dummy_data)

    # Configure basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize and run the ingestor using the dummy CSV
    # The actual CSV path for the user's data is "legal_text_classification.csv"
    # but for this __main__ block, we use the dummy one.
    ingestor_service = DocumentIngestorService(csv_file_path=dummy_csv_path)
    # The column names are now set based on user's CSV header:
    # ingestor_service.id_column_name = "case_id"
    # ingestor_service.text_column_name = "case_text"
    # ingestor_service.dynamic_metadata_columns will be ["case_outcome", "case_title"]
    
    ingested_docs = ingestor_service.ingest_documents_from_csv()
    
    if ingested_docs:
        logger.info(f"Example Ingestion - First document: {ingested_docs[0]}")
        logger.info(f"Example Ingestion - Total documents ingested: {len(ingested_docs)}")
        # Expected: 3 documents (CASE003 and the one with no ID should be skipped)
        # Actually, CASE003 will be ingested if text is empty but ID is present.
        # The one with no ID will be skipped.
        # So, expected 4 documents. Let's refine the dummy data or logic.
        # Corrected: The current logic skips rows with empty text_content or empty doc_id.
        # So, CASE003 (empty text) and the "No ID Example" will be skipped.
        # Expected: 3 documents (CASE001, CASE002, CASE004)
    else:
        logger.warning("Example Ingestion - No documents were ingested.")

    # Clean up dummy CSV
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
    
    # --- Test with a non-existent file ---
    logger.info("\n--- Testing with a non-existent CSV file ---")
    ingestor_non_existent = DocumentIngestorService(csv_file_path="non_existent_file.csv")
    ingestor_non_existent.ingest_documents_from_csv()

    # --- Test with a CSV with missing essential columns ---
    logger.info("\n--- Testing with a CSV missing essential columns ---")
    dummy_missing_col_path = "dummy_missing_cols.csv"
    dummy_missing_data = [
        {"some_other_id": "ID1", "some_text": "Text 1", "outcome": "Won"}
    ]
    fieldnames_missing = ["some_other_id", "some_text", "outcome"]
    with open(dummy_missing_col_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_missing)
        writer.writeheader()
        writer.writerows(dummy_missing_data)
    
    ingestor_missing_cols = DocumentIngestorService(csv_file_path=dummy_missing_col_path)
    # Default id_column_name is "case_id", text_column_name is "case_text"
    ingestor_missing_cols.ingest_documents_from_csv()
    if os.path.exists(dummy_missing_col_path):
        os.remove(dummy_missing_col_path)

    # --- Test with an empty CSV ---
    logger.info("\n--- Testing with an empty CSV file ---")
    empty_csv_path = "empty.csv"
    with open(empty_csv_path, mode='w', newline='', encoding='utf-8') as f:
        pass # Create an empty file
    ingestor_empty = DocumentIngestorService(csv_file_path=empty_csv_path)
    ingestor_empty.ingest_documents_from_csv()
    if os.path.exists(empty_csv_path):
        os.remove(empty_csv_path)
    
    logger.info("\n--- End of example usage ---")