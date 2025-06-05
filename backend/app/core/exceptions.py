"""
Custom exceptions for the chatbot application.
"""

class ChatbotBaseException(Exception):
    """Base exception for this application."""
    def __init__(self, message: str = "An application error occurred."):
        self.message = message
        super().__init__(self.message)

class DocumentIngestionError(ChatbotBaseException):
    """Custom exception for errors during document ingestion."""
    def __init__(self, message: str = "Error during document ingestion.", details: str = None):
        super().__init__(message)
        self.details = details

class DocumentProcessingError(ChatbotBaseException):
    """Custom exception for errors during document processing (cleaning, chunking, embedding)."""
    def __init__(self, message: str = "Error during document processing.", details: str = None):
        super().__init__(message)
        self.details = details

class VectorDBError(ChatbotBaseException):
    """Custom exception for errors related to vector database operations."""
    def __init__(self, message: str = "Error interacting with the vector database.", details: str = None):
        super().__init__(message)
        self.details = details

class QueryProcessingError(ChatbotBaseException):
    """Custom exception for errors during query processing or answer extraction."""
    def __init__(self, message: str = "Error during query processing.", details: str = None):
        super().__init__(message)
        self.details = details

class ThemeAnalysisError(ChatbotBaseException):
    """Custom exception for errors during theme analysis."""
    def __init__(self, message: str = "Error during theme analysis.", details: str = None):
        super().__init__(message)
        self.details = details

class LLMError(ChatbotBaseException):
    """Custom exception for errors interacting with the LLM (e.g., API errors, parsing LLM responses)."""
    def __init__(self, message: str = "Error interacting with the Language Model.", details: str = None):
        super().__init__(message)
        self.details = details

class ConfigurationError(ChatbotBaseException):
    """Custom exception for configuration-related errors."""
    def __init__(self, message: str = "Configuration error.", details: str = None):
        super().__init__(message)
        self.details = details

# Example of how these might be used:
# raise DocumentIngestionError("Failed to parse CSV.", details="File not found at path: /path/to/file.csv")
# raise LLMError("Failed to get response from Gemini API.", details=str(api_exception))