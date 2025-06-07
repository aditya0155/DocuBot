# Document Research & Theme Identification Chatbot

## 1. Overview/Objective

This project is an AI-powered chatbot designed to ingest and process a large volume of documents (initially from a CSV of 25,000 legal case documents). Users can ask natural language questions about the content. The chatbot will:
*   Extract relevant answers from individual documents.
*   Provide accurate citations for each answer (e.g., Document ID, page, paragraph).
*   Identify common themes across all retrieved document answers for a given query.
*   Summarize these themes and present them in a synthesized, chat-style format with supporting document citations.
*   Offer a simple and intuitive web interface for interaction.

This project is being developed as part of the Wasserstoff AI Software Intern Task.

## 3. System Architecture

The system is designed with a decoupled frontend and backend architecture to ensure modularity and scalability.


**Detailed Components:**
*   **Frontend:** Streamlit application (`frontend/app.py`) for user interaction.
    *   Provides UI for asking questions.
    *   Provides UI for uploading TXT/PDF documents.
    *   Displays processed documents, individual answers, and synthesized themes.
    *   Communicates with the backend via HTTP API calls.
*   **Backend API:** FastAPI application (`backend/app/main.py`) serving requests from the frontend.
    *   **API Routers (`query_api.py`, `documents_api.py`):** Define and handle incoming HTTP requests for querying, document uploads, and document listing.
    *   **Core Services (`backend/app/services/`):** Encapsulate the business logic:
        *   `DocumentIngestorService`: Handles reading and parsing the initial CSV dataset.
        *   `DocumentProcessorService`: Responsible for text cleaning, chunking documents into manageable pieces, and generating vector embeddings using the Google Gemini API.
        *   `VectorDBService`: Manages interactions with the ChromaDB vector store (storing, querying embeddings).
        *   `QueryProcessorService`: Processes user queries by generating query embeddings, performing semantic search against the vector store, and using the Gemini chat model to extract answers from relevant document chunks.
        *   `ThemeAnalyzerService`: Takes extracted answers and uses the Gemini chat model to identify and summarize common themes.
    *   **Pydantic Models (`schemas.py`):** Define data structures for request and response validation, ensuring data integrity.
    *   **Configuration (`config.py`, `.env`):** Manages application settings, including API keys and model IDs, loaded from environment variables.
*   **Vector Database:** ChromaDB, used as a persistent store for document embeddings, enabling efficient semantic search.
*   **LLM (Large Language Model):** Google Gemini API, accessed via the `google-generativeai` SDK.
    *   `gemini-embedding-exp-03-07` (or configured model): Used for generating embeddings for document chunks (`RETRIEVAL_DOCUMENT` task type) and user queries (`RETRIEVAL_QUERY` task type).
    *   `gemini-1.5-flash-latest` (or configured model): Used for extracting answers from text chunks and for identifying/summarizing themes.
*   **Data Storage:**
    *   `legal_text_classification.csv`: The initial dataset of legal documents.
    *   `backend/data/vector_store/`: Directory for ChromaDB's persistent storage.
    *   `backend/data/uploads/`: Directory where uploaded TXT/PDF files are temporarily saved for processing.

## 4. Technology Stack

*   **Backend:**
    *   Python 3.x
    *   FastAPI (Web framework)
    *   Uvicorn (ASGI server)
    *   Pydantic (Data validation)
    *   `google-generativeai` (Python SDK for Google Gemini API)
    *   ChromaDB (Vector database)
    *   python-dotenv (Environment variable management)
*   **Frontend:**
    *   Streamlit (Web framework)
    *   Requests (HTTP client for backend communication)
*   **LLM:**
    *   Google Gemini
        *   Chat Model: `gemini-1.5-flash-latest` (or as configured in `.env`)
        *   Embedding Model: `gemini-embedding-exp-03-07` (or as configured in `.env`)
    *   SDK: `google-generativeai`
*   **Version Control:** Git

## 5. Project Structure

Please refer to `structure.md` for a detailed explanation of the directory layout and key files.

## 6. Setup and Installation Instructions

*(To be detailed once initial setup scripts and procedures are finalized)*

**Prerequisites:**
*   Python (version 3.8+ recommended)
*   Git
*   Access to the internet for downloading dependencies and LLM API calls.

**General Steps (Backend):**
1.  Clone the repository: `git clone <repository-url>`
2.  Navigate to the backend directory: `cd chatbot_theme_identifier/backend`
3.  Create a Python virtual environment: `python -m venv venv`
4.  Activate the virtual environment:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
5.  Install dependencies: `pip install -r requirements.txt`
6.  Create a `.env` file in the `backend` directory by copying `.env.example`.
7.  Fill in your `LLM_API_KEY` in the `.env` file.
8.  Run the FastAPI server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (Adjust host/port as needed from `.env`)

**General Steps (Frontend):**
1.  Navigate to the frontend directory: `cd chatbot_theme_identifier/frontend`
2.  (If not using the same venv as backend) Create and activate a virtual environment and install dependencies: `pip install -r requirements.txt`
3.  Run the Streamlit app: `streamlit run app.py`

## 7. Usage Guide

1.  **Prerequisites:**
    *   Ensure Python 3.8+ and Git are installed.
    *   Have your Google Gemini API key ready.

2.  **Setup Backend:**
    *   Clone the repository.
    *   Navigate to `chatbot_theme_identifier/backend`.
    *   Create and activate a Python virtual environment (e.g., `python -m venv venv`, then `source venv/bin/activate` or `venv\Scripts\activate`).
    *   Install dependencies: `pip install -r requirements.txt`.
    *   Create a `.env` file by copying `.env.example` and add your `GEMINI_API_KEY`.
    *   The `legal_text_classification.csv` file should be in the project root (e.g., `c:/kaggle/Project/`).
    *   Run the backend server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
    *   On first startup, if the vector database is empty, the backend will ingest and process `legal_text_classification.csv`. This may take some time. Monitor backend logs. You can also trigger this manually via the `/api/v1/admin/ingest-csv` endpoint.

3.  **Setup Frontend:**
    *   Navigate to `chatbot_theme_identifier/frontend`.
    *   If using a separate virtual environment, create and activate it.
    *   Install dependencies: `pip install -r requirements.txt`.
    *   Run the Streamlit app: `streamlit run app.py`.

4.  **Using the Application:**
    *   The Streamlit application will open in your web browser.
    *   **Document Upload (Optional):** Use the sidebar to upload TXT or PDF files. Click "Process" to have them ingested into the system. The backend will save, extract text, process, and store these documents.
    *   **View Processed Documents:** The sidebar will list documents currently in the system (from CSV and uploads), fetched from the backend.
    *   **Ask Questions:** Enter your natural language question about the documents in the main text area. Adjust the "Number of relevant chunks to retrieve" in the sidebar if needed.
    *   **Get Results:** Click "Get Answers & Themes". The application will display:
        *   Individual answers extracted from relevant document chunks, along with citations (document ID, source file, chunk number).
        *   Common themes identified across the answers, with summaries and supporting document IDs.

## 8. Methodology/Approach

*   **Document Ingestion & Processing:**
    *   **Sources:** The system ingests documents from two primary sources:
        *   A large pre-existing CSV file (`legal_text_classification.csv`) containing legal cases, processed on initial backend startup if the database is empty, or via an admin endpoint.
        *   User-uploaded TXT and PDF files via a dedicated API endpoint.
    *   **Text Extraction:**
        *   For CSVs, text is directly read from the specified column (`case_text`).
        *   For TXT files, content is read directly.
        *   For PDFs, `PyPDF2` library is used to extract text page by page. Image-based PDFs without embedded text will not yield content with this method (OCR is a future enhancement).
    *   **Cleaning:** Basic text cleaning (e.g., normalizing whitespace) is applied to the extracted text.
    *   **Chunking:** The cleaned text is divided into smaller, semantically coherent chunks. Currently, this is done by splitting the text into segments of approximately 5 sentences. This helps in managing token limits for embedding models and allows for more targeted retrieval.
    *   **Embedding Generation:** Each text chunk is converted into a high-dimensional vector (embedding) using Google's `gemini-embedding-exp-03-07` model (task type: `RETRIEVAL_DOCUMENT`). These embeddings capture the semantic meaning of the chunks.
    *   **Vector Storage:** The embeddings, along with the corresponding text chunks and metadata (original document ID, source filename, chunk number, etc.), are stored in a ChromaDB persistent vector database. This allows for efficient similarity searches.

*   **Querying & Answer Extraction:**
    *   **Query Embedding:** When a user submits a query, it is also converted into an embedding using the same `gemini-embedding-exp-03-07` model (task type: `RETRIEVAL_QUERY`).
    *   **Semantic Search:** The query embedding is used to search the ChromaDB vector store for the most similar document chunks (based on cosine similarity or other distance metrics). The top N relevant chunks are retrieved.
    *   **Answer Extraction:** For each retrieved relevant chunk, the user's original query and the chunk's text are passed to the Google Gemini chat model (`gemini-1.5-flash-latest`). A carefully crafted prompt instructs the model to extract a concise answer to the query strictly from the provided chunk text. If no relevant information is found, it's instructed to indicate so.
    *   **Citation:** Each extracted answer is accompanied by a citation, typically including the original document ID, the source filename, and the chunk number from which the answer was derived.

*   **Theme Identification & Synthesis:**
    *   **Input:** The list of individual answers extracted in the previous step, along with the original user query.
    *   **LLM-Powered Analysis:** These collected answers are passed to the Gemini chat model (`gemini-1.5-flash-latest`) with a detailed prompt. This prompt asks the model to:
        *   Identify common themes or topics present across the provided set of answers.
        *   For each identified theme, generate a concise name and a summary.
        *   List the document IDs that support each theme.
    *   **Structured Output:** The prompt explicitly requests the LLM to return this information in a JSON format to simplify parsing. The backend service then parses this JSON to create structured `Theme` objects.
    *   **Presentation:** The identified themes, their summaries, and supporting document IDs are then presented to the user in the frontend.

## 9. Error Handling

The application will aim for robust error handling, providing informative messages for issues such as:
*   CSV parsing errors.
*   LLM API connectivity or processing errors.
*   Vector database errors.
*   Invalid user inputs.

## 10. Known Issues/Limitations

*   **OCR Accuracy:** Text extraction from images using Tesseract OCR may vary in accuracy depending on image quality, font, and layout. Complex or noisy images might yield suboptimal results.
*   **PDF Text Extraction:** `PyPDF2` may not be able to extract text from all PDFs, especially scanned/image-based PDFs or those with complex encodings or security restrictions. The current implementation does not fall back to OCR for PDFs if direct text extraction fails, though OCR is used for direct image uploads.
*   **Citation Granularity:** Citations currently include Document ID, source file, and chunk number. Page and paragraph-level citations, especially for PDFs, are not yet precisely implemented and rely on the granularity of chunks.
*   **Scalability of Initial CSV Ingestion:** The initial ingestion of a very large CSV (25,000+ documents) on application startup can be time-consuming. While an admin endpoint exists for manual re-ingestion, a fully asynchronous background processing pipeline would be better for production scenarios.
*   **Service Instantiation:** There's a slight difference in service instantiation: `main.py` uses global instances for some services during startup, while API modules like `query_api.py` and `documents_api.py` use their own lazy-loaded instances. This is functional but could be unified with a dependency injection framework for larger applications.
*   **Error Handling in Frontend:** While the backend has custom exceptions, the Streamlit frontend's error display for backend issues is generic. More specific error messages could be propagated.
*   **`upload-multiple` Response:** The `/documents/upload-multiple` endpoint currently returns a general success message. It does not yet provide a detailed status or list of processed IDs for each individual file in the batch.

## 11. Future Enhancements

*   **Support for More Document Types:** Add parsers for DOCX (e.g., using `python-docx`), HTML, etc.
*   **Enhanced Citation Precision:**
    *   Integrate PDF parsing libraries that provide more detailed layout information (page, paragraph, bounding boxes) to enable more precise citations.
    *   Refine chunking strategy to better align with document structure for improved citation context.
*   **Improved Scalability:**
    *   Implement an asynchronous task queue (e.g., Celery) for document ingestion and processing, especially for large batches or the initial CSV load.
    *   Add an API to monitor the status of background ingestion tasks.
*   **Advanced Frontend Features:**
    *   Implement user accounts and document ownership/permissions if needed.
    *   Add more sophisticated filtering and sorting options for documents and query results.
    *   Visual interface for mapping citations (e.g., clickable links from themes/answers to document highlights).
*   **Refined LLM Interaction:**
    *   Systematic prompt engineering and A/B testing for answer extraction and theme analysis prompts to improve accuracy and relevance.
    *   Option to choose different LLM models or configurations.
    *   Implement a mechanism for users to provide feedback on answer quality.
*   **Consolidated Answer Generation:** Explicitly generate a single, overarching "consolidated answer" for the user's query, in addition to themes, if desired.
*   **Comprehensive API Integration Testing:** Expand test coverage for all API endpoints, including more edge cases and error conditions.
*   **Deployment:** Dockerize the backend and frontend for easier deployment to cloud platforms (e.g., Vercel, AWS, Google Cloud Run).
*   **Security Hardening:** Review and implement security best practices, especially for file uploads and API endpoints.
*   **Configuration Management:** Move more hardcoded values (like UPLOAD_DIRECTORY in `documents_api.py`) to the central configuration (`config.py` / `.env`).



---
