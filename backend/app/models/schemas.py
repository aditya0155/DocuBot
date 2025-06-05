from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Document Schemas ---

class DocumentMetadata(BaseModel):
    """
    Represents metadata associated with a document or a chunk.
    This can be flexible.
    """
    source_file: Optional[str] = None
    source_row_number: Optional[int] = None
    case_title: Optional[str] = None
    case_outcome: Optional[str] = None
    # Add any other metadata fields that come from the CSV or are generated
    chunk_number: Optional[int] = None 
    # Potentially page numbers, paragraph numbers if extracted

class DocumentChunk(BaseModel):
    """
    Represents a single processed chunk of a document.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk (e.g., doc_id_chunk_num)")
    doc_id: str = Field(..., description="Identifier of the original document")
    text_chunk: str = Field(..., description="The actual text content of the chunk")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Metadata associated with the chunk")
    # Embedding is not typically sent over API responses unless specifically needed for debugging.
    # It's primarily used internally for DB storage and querying.

# --- API Request/Response Schemas ---

# For Document Upload (Placeholder for now, as initial ingestion is from CSV)
class DocumentUploadRequest(BaseModel):
    # For file uploads, FastAPI uses UploadFile. This schema is more for metadata if needed.
    filename: Optional[str] = None
    # custom_metadata: Optional[Dict[str, Any]] = None # If users can provide metadata during upload

class DocumentUploadResponse(BaseModel):
    message: str
    uploaded_files: List[str] = []
    processed_doc_ids: List[str] = []

# For Ingesting from pre-configured CSV
class CSVIngestionRequest(BaseModel):
    csv_file_path: str = Field(description="Path to the CSV file to ingest (relative to project or absolute)")
    # Potentially add options for column name mapping if we want to make it configurable via API
    # id_column: Optional[str] = None
    # text_column: Optional[str] = None
    # metadata_columns: Optional[List[str]] = None

class CSVIngestionResponse(BaseModel):
    message: str
    documents_ingested: int
    chunks_created: int
    db_collection_status: str
    db_total_items: int

# For Querying
class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Natural language query from the user")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of relevant document chunks to retrieve")
    # filter_metadata: Optional[Dict[str, Any]] = None # For advanced filtering

class IndividualAnswer(BaseModel):
    doc_id: str
    extracted_answer: str
    citation_source: str # e.g., "DocID, Page X, Para Y" or "DocID (Chunk Z)"
    text_chunk: Optional[str] = None # The original chunk from which answer was extracted
    metadata: Optional[DocumentMetadata] = None
    relevance_score: Optional[float] = None # e.g., distance from vector search

class Theme(BaseModel):
    theme_name: str
    theme_summary: str
    supporting_doc_ids: List[str]
    # supporting_chunks: Optional[List[IndividualAnswer]] = None # If we want to link themes to specific answers/chunks

class QueryResponse(BaseModel):
    original_query: str
    individual_answers: List[IndividualAnswer] = []
    themes: List[Theme] = []
    # consolidated_answer: Optional[str] = None # A single, overarching answer if required

# For general status messages
class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

# For listing documents (basic)
class DocumentInfo(BaseModel):
    doc_id: str
    title: Optional[str] = None # e.g., case_title
    source: Optional[str] = None # e.g., filename or CSV row
    # Add other summary fields as needed

class ListDocumentsResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]