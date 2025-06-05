import logging
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from typing import List, Optional

from ..models import schemas # Pydantic models for response
from ..services.document_processor import DocumentProcessorService # Still needed for type hinting
from ..services.vector_db_service import VectorDBService # Still needed for type hinting
from ..core.exceptions import DocumentProcessingError, LLMError 
from ..dependencies import get_doc_processor_serv, get_vector_db_serv # Import dependency providers from dependencies.py

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Remove local lazy initialization ---
# _document_processor_service_instance: Optional[DocumentProcessorService] = None
# _vector_db_service_instance: Optional[VectorDBService] = None

# def get_doc_processor_service() -> DocumentProcessorService:
#     global _document_processor_service_instance
#     if _document_processor_service_instance is None:
#         _document_processor_service_instance = DocumentProcessorService()
#         logger.info("DocumentProcessorService instance created for documents_api.")
#     return _document_processor_service_instance

# def get_vector_db_service() -> VectorDBService:
#     global _vector_db_service_instance
#     if _vector_db_service_instance is None:
#         _vector_db_service_instance = VectorDBService()
#         if not _vector_db_service_instance.collection:
#             logger.error("Failed to initialize VectorDBService for documents_api.")
#         logger.info("VectorDBService instance created for documents_api.")
#     return _vector_db_service_instance
# --- End Service Instantiation ---

# Placeholder for where uploaded files could be saved or processed
# In a real scenario, this would be configurable and more robust.
import os
import shutil
UPLOAD_DIRECTORY = "./backend/data/uploads"
# Ensure upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@router.post("/documents/upload", response_model=schemas.DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessorService = Depends(get_doc_processor_serv),
    vector_db: VectorDBService = Depends(get_vector_db_serv)
):
    """
    Endpoint for uploading a single document.
    It saves the file, extracts text based on content type,
    processes it (chunks, embeds), and adds to the vector DB.
    """
    logger.info(f"Received file upload request for: {file.filename}, content type: {file.content_type}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"File '{file.filename}' saved to '{file_location}'")
    except Exception as e:
        logger.error(f"Could not save file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    extracted_text = ""
    processing_message = f"File '{file.filename}' saved. "

    # Services (doc_processor, vector_db) are now injected via Depends.
    
    if file.content_type == "text/plain":
        try:
            with open(file_location, "r", encoding="utf-8") as f:
                extracted_text = f.read()
            logger.info(f"Extracted text from TXT file: {file.filename[:50]}...")
            
            if not vector_db.collection:
                 raise HTTPException(status_code=503, detail="Vector DB service is not available.")

            cleaned_text = doc_processor.clean_text(extracted_text)
            text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

            if not text_chunks_content:
                processing_message += "Text extracted but resulted in no processable chunks."
            else:
                doc_id_for_upload = f"upload_{file.filename}"
                chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                
                processed_chunks_for_db = []
                for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                    chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                    embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                    if embedding:
                        processed_chunks_for_db.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id_for_upload,
                            "text_chunk": chunk_text_item,
                            "embedding": embedding,
                            "metadata": {
                                "source_file": file.filename,
                                "original_content_type": file.content_type,
                                "upload_source": "api_upload",
                                "chunk_number": chunk_idx
                            }
                        })
                    else:
                        logger.warning(f"Skipping chunk {chunk_id} due to missing embedding.")
                
                if processed_chunks_for_db:
                    vector_db.add_documents(processed_chunks_for_db)
                    processing_message += f"Text extracted, processed into {len(processed_chunks_for_db)} chunks, and stored."
                    logger.info(f"Processed and stored {len(processed_chunks_for_db)} chunks from TXT file {file.filename}.")
                else:
                    processing_message += "Text extracted, but no chunks were successfully processed for storage."

        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for TXT file {file.filename}: {proc_err.message} - {proc_err.details}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {proc_err.message}")
        except Exception as e:
            logger.error(f"Could not read or process TXT file {file.filename}: {e}", exc_info=True)
            processing_message += f"Could not read/process TXT content: {str(e)}."
    elif file.content_type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_location)
            pdf_text_content = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_text_content.append(page.extract_text())
            extracted_text = "\n".join(pdf_text_content)
            if not extracted_text.strip(): # Check if any text was actually extracted
                 logger.warning(f"PyPDF2 extracted no text from PDF: {file.filename}. The PDF might be image-based or protected.")
                 processing_message += f"PDF file '{file.filename}' processed, but no text could be extracted (it might be an image-based PDF or protected). OCR not yet implemented."
            else:
                logger.info(f"Extracted text from PDF file: {file.filename[:50]}...")
                # doc_processor and vector_db are injected
                if not vector_db.collection:
                    raise HTTPException(status_code=503, detail="Vector DB service is not available.")

                cleaned_text = doc_processor.clean_text(extracted_text)
                text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

                if not text_chunks_content:
                    processing_message += "Text extracted from PDF but resulted in no processable chunks."
                else:
                    doc_id_for_upload = f"upload_{file.filename}"
                    chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                    
                    processed_chunks_for_db = []
                    for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                        chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                        embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                        if embedding:
                            processed_chunks_for_db.append({
                                "chunk_id": chunk_id,
                                "doc_id": doc_id_for_upload,
                                "text_chunk": chunk_text_item,
                                "embedding": embedding,
                                "metadata": {
                                    "source_file": file.filename,
                                    "original_content_type": file.content_type,
                                    "upload_source": "api_upload",
                                    "chunk_number": chunk_idx
                                }
                            })
                        else:
                            logger.warning(f"Skipping chunk {chunk_id} from PDF due to missing embedding.")
                    
                    if processed_chunks_for_db:
                        vector_db.add_documents(processed_chunks_for_db)
                        processing_message += f"Text extracted from PDF, processed into {len(processed_chunks_for_db)} chunks, and stored."
                        logger.info(f"Processed and stored {len(processed_chunks_for_db)} chunks from PDF file {file.filename}.")
                    else:
                        processing_message += "Text extracted from PDF, but no chunks were successfully processed for storage."
        except ImportError:
            logger.error("PyPDF2 library is not installed. Cannot process PDF files.")
            processing_message += "PDF processing skipped: PyPDF2 library not found."
        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for PDF file {file.filename}: {proc_err.message} - {proc_err.details}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF document: {proc_err.message}")
        except Exception as e:
            logger.error(f"Could not read or process PDF file {file.filename}: {e}", exc_info=True)
            processing_message += f"Could not read/process PDF content: {str(e)}."
    elif file.content_type in ["image/png", "image/jpeg", "image/tiff"]:
        try:
            import pytesseract
            from PIL import Image

            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example path for Windows
            # User might need to set this path if Tesseract is not in PATH

            img = Image.open(file_location)
            extracted_text = pytesseract.image_to_string(img)
            
            if not extracted_text.strip():
                logger.warning(f"Pytesseract extracted no text from image: {file.filename}.")
                processing_message += f"Image file '{file.filename}' processed, but no text could be extracted by OCR."
            else:
                logger.info(f"Extracted text from image file using OCR: {file.filename[:50]}...")
                # doc_processor and vector_db are injected
                if not vector_db.collection:
                    raise HTTPException(status_code=503, detail="Vector DB service is not available.")

                cleaned_text = doc_processor.clean_text(extracted_text)
                text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

                if not text_chunks_content:
                    processing_message += "Text extracted from image by OCR but resulted in no processable chunks."
                else:
                    doc_id_for_upload = f"upload_{file.filename}" # Define here for use in response
                    chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                    
                    processed_chunks_for_db = []
                    for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                        chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                        embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                        if embedding:
                            processed_chunks_for_db.append({
                                "chunk_id": chunk_id, "doc_id": doc_id_for_upload,
                                "text_chunk": chunk_text_item, "embedding": embedding,
                                "metadata": {
                                    "source_file": file.filename, "original_content_type": file.content_type,
                                    "upload_source": "api_upload_ocr", "chunk_number": chunk_idx
                                }
                            })
                        else:
                            logger.warning(f"Skipping chunk {chunk_id} from OCR'd image due to missing embedding.")
                    
                    if processed_chunks_for_db:
                        vector_db.add_documents(processed_chunks_for_db)
                        processing_message += f"Text extracted from image by OCR, processed into {len(processed_chunks_for_db)} chunks, and stored."
                        logger.info(f"Processed and stored {len(processed_chunks_for_db)} chunks from OCR'd image {file.filename}.")
                    else:
                        processing_message += "Text extracted from image by OCR, but no chunks were successfully processed for storage."
        except ImportError:
            logger.error("Pytesseract or Pillow library is not installed. Cannot process image files with OCR.")
            processing_message += "Image OCR processing skipped: Pytesseract or Pillow library not found."
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR engine not found. Ensure it's installed and in PATH, or tesseract_cmd is set.")
            processing_message += "Image OCR processing failed: Tesseract OCR engine not found."
            # Consider raising HTTPException here if OCR is critical and fails due to setup
            # raise HTTPException(status_code=500, detail="OCR engine not found. Please configure Tesseract.")
        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for OCR'd image {file.filename}: {proc_err.message} - {proc_err.details}")
            raise HTTPException(status_code=500, detail=f"Error processing OCR'd document: {proc_err.message}")
        except Exception as e:
            logger.error(f"Could not process image file {file.filename} with OCR: {e}", exc_info=True)
            processing_message += f"Could not process image with OCR: {str(e)}."
    elif file.content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        try:
            import pandas as pd
            logger.info(f"Attempting to process Excel file: {file.filename} with content type: {file.content_type}")
            
            # Read all sheets into a dictionary of DataFrames
            excel_file = pd.ExcelFile(file_location)
            all_text_content = []

            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name, header=None) # Read without assuming a header
                # Iterate over all cells in the DataFrame and convert to string
                for r_idx in range(len(df)):
                    for c_idx in range(len(df.columns)):
                        cell_value = df.iat[r_idx, c_idx]
                        if pd.notna(cell_value): # Check if cell is not NaN
                            all_text_content.append(str(cell_value))
            
            extracted_text = "\n".join(all_text_content)

            if not extracted_text.strip():
                logger.warning(f"Pandas extracted no text from Excel file: {file.filename}.")
                processing_message += f"Excel file '{file.filename}' processed, but no text could be extracted."
            else:
                logger.info(f"Extracted text from Excel file: {file.filename[:50]}...")
                # doc_processor and vector_db are injected
                if not vector_db.collection:
                    raise HTTPException(status_code=503, detail="Vector DB service is not available.")

                cleaned_text = doc_processor.clean_text(extracted_text)
                text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

                if not text_chunks_content:
                    processing_message += "Text extracted from Excel but resulted in no processable chunks."
                else:
                    doc_id_for_upload = f"upload_{file.filename}"
                    chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                    
                    processed_chunks_for_db = []
                    for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                        chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                        embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                        if embedding:
                            processed_chunks_for_db.append({
                                "chunk_id": chunk_id, "doc_id": doc_id_for_upload,
                                "text_chunk": chunk_text_item, "embedding": embedding,
                                "metadata": {
                                    "source_file": file.filename, "original_content_type": file.content_type,
                                    "upload_source": "api_upload_excel", "chunk_number": chunk_idx
                                }
                            })
                        else:
                            logger.warning(f"Skipping chunk {chunk_id} from Excel file due to missing embedding.")
                    
                    if processed_chunks_for_db:
                        vector_db.add_documents(processed_chunks_for_db)
                        processing_message += f"Text extracted from Excel, processed into {len(processed_chunks_for_db)} chunks, and stored."
                        logger.info(f"Processed and stored {len(processed_chunks_for_db)} chunks from Excel file {file.filename}.")
                    else:
                        processing_message += "Text extracted from Excel, but no chunks were successfully processed for storage."
        except ImportError:
            logger.error("Pandas or its dependency (e.g., openpyxl for .xlsx, xlrd for .xls) is not installed. Cannot process Excel files.")
            processing_message += "Excel processing skipped: Pandas or required Excel library not found."
            # Consider raising HTTPException here if Excel processing is critical
            # raise HTTPException(status_code=501, detail="Excel processing library not installed.")
        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for Excel file {file.filename}: {proc_err.message} - {proc_err.details}")
            raise HTTPException(status_code=500, detail=f"Error processing Excel document: {proc_err.message}")
        except Exception as e:
            logger.error(f"Could not read or process Excel file {file.filename}: {e}", exc_info=True)
            processing_message += f"Could not read/process Excel content: {str(e)}."
    elif file.content_type in ["text/csv", "application/csv"]:
        try:
            import pandas as pd
            logger.info(f"Attempting to process CSV file: {file.filename} with content type: {file.content_type}")

            df = pd.read_csv(file_location, header=None) # Read without assuming a header
            all_text_content = []
            for r_idx in range(len(df)):
                for c_idx in range(len(df.columns)):
                    cell_value = df.iat[r_idx, c_idx]
                    if pd.notna(cell_value):
                        all_text_content.append(str(cell_value))
            
            extracted_text = "\n".join(all_text_content)

            if not extracted_text.strip():
                logger.warning(f"Pandas extracted no text from CSV file: {file.filename}.")
                processing_message += f"CSV file '{file.filename}' processed, but no text could be extracted."
            else:
                logger.info(f"Extracted text from CSV file: {file.filename[:50]}...")
                # doc_processor and vector_db are injected
                if not vector_db.collection:
                    raise HTTPException(status_code=503, detail="Vector DB service is not available.")

                cleaned_text = doc_processor.clean_text(extracted_text)
                text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

                if not text_chunks_content:
                    processing_message += "Text extracted from CSV but resulted in no processable chunks."
                else:
                    doc_id_for_upload = f"upload_{file.filename}"
                    chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                    
                    processed_chunks_for_db = []
                    for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                        chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                        embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                        if embedding:
                            processed_chunks_for_db.append({
                                "chunk_id": chunk_id, "doc_id": doc_id_for_upload,
                                "text_chunk": chunk_text_item, "embedding": embedding,
                                "metadata": {
                                    "source_file": file.filename, "original_content_type": file.content_type,
                                    "upload_source": "api_upload_csv", "chunk_number": chunk_idx
                                }
                            })
                        else:
                            logger.warning(f"Skipping chunk {chunk_id} from CSV file due to missing embedding.")
                    
                    if processed_chunks_for_db:
                        vector_db.add_documents(processed_chunks_for_db)
                        processing_message += f"Text extracted from CSV, processed into {len(processed_chunks_for_db)} chunks, and stored."
                        logger.info(f"Processed and stored {len(processed_chunks_for_db)} chunks from CSV file {file.filename}.")
                    else:
                        processing_message += "Text extracted from CSV, but no chunks were successfully processed for storage."
        except ImportError:
            logger.error("Pandas library is not installed. Cannot process CSV files.")
            # processing_message += "CSV processing skipped: Pandas library not found." # Replaced by HTTPException
            raise HTTPException(status_code=501, detail="CSV processing error: Pandas library not installed on server.")
        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for CSV file {file.filename}: {proc_err.message} - {proc_err.details}")
            raise HTTPException(status_code=500, detail=f"Error processing CSV document: {proc_err.message}")
        except Exception as e: # Catch other errors like pd.read_csv failing
            logger.error(f"Could not read or process CSV file {file.filename}: {e}", exc_info=True)
            # processing_message += f"Could not read/process CSV content: {str(e)}." # Replaced by HTTPException
            raise HTTPException(status_code=422, detail=f"Could not parse CSV file {file.filename}: {str(e)}")
    else:
        processing_message += f"Unsupported file type: {file.content_type}. Processing not implemented for {file.filename}."
        logger.warning(f"Unsupported file type for {file.filename}: {file.content_type}")


    processed_doc_id_list = []
    if "stored" in processing_message and "upload_" in processing_message: # A bit heuristic, better to use a flag
        # Attempt to extract the doc_id if it was successfully processed and stored
        # This assumes doc_id_for_upload was set and used in a success message part
        # A more robust way would be to set a flag or return the doc_id from a helper function
        try:
            # Example: if doc_id_for_upload was set before this return
            # This is a simplification; actual doc_id should be confirmed from processing logic
            if 'doc_id_for_upload' in locals() and doc_id_for_upload:
                 processed_doc_id_list.append(doc_id_for_upload)
        except NameError: # doc_id_for_upload might not be defined if processing failed early
            pass


    return schemas.DocumentUploadResponse(
        message=processing_message,
        uploaded_files=[file.filename],
        processed_doc_ids=processed_doc_id_list
    )

@router.post("/documents/upload-multiple", response_model=schemas.DocumentUploadResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    doc_processor: DocumentProcessorService = Depends(get_doc_processor_serv),
    vector_db: VectorDBService = Depends(get_vector_db_serv)
):
    """
    Endpoint for uploading multiple documents.
    """
    uploaded_filenames = []
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    for file in files:
        logger.info(f"Received file for multiple upload: {file.filename}, content type: {file.content_type}")
        if not file.filename:
            logger.warning("A file without a filename was provided in multiple upload, skipping.")
            continue
        
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        doc_id_for_upload = f"upload_{file.filename}"
        current_file_processed_ids = []
        current_file_message = f"File '{file.filename}'"

        try:
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
            logger.info(f"File '{file.filename}' saved to '{file_location}' for multiple upload.")
            current_file_message += " saved. "

            extracted_text = ""
            if file.content_type == "text/plain":
                with open(file_location, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                logger.info(f"Extracted text from TXT file: {file.filename}")
                current_file_message += "TXT content extracted. "
            elif file.content_type == "application/pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(file_location)
                pdf_text_content = [page.extract_text() for page in reader.pages]
                extracted_text = "\n".join(pdf_text_content)
                if not extracted_text.strip():
                    logger.warning(f"PyPDF2 extracted no text from PDF: {file.filename} during multiple upload.")
                    current_file_message += "PDF processed, but no text extracted. "
                else:
                    logger.info(f"Extracted text from PDF file: {file.filename}")
                    current_file_message += "PDF content extracted. "
            elif file.content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                import pandas as pd
                excel_file_pd = pd.ExcelFile(file_location)
                all_text_content_pd = []
                for sheet_name in excel_file_pd.sheet_names:
                    df_pd = excel_file_pd.parse(sheet_name, header=None)
                    for r_idx in range(len(df_pd)):
                        for c_idx in range(len(df_pd.columns)):
                            cell_value = df_pd.iat[r_idx, c_idx]
                            if pd.notna(cell_value):
                                all_text_content_pd.append(str(cell_value))
                extracted_text = "\n".join(all_text_content_pd)
                if not extracted_text.strip():
                    logger.warning(f"Pandas extracted no text from Excel: {file.filename} during multiple upload.")
                    current_file_message += "Excel processed, but no text extracted. "
                else:
                        logger.info(f"Extracted text from Excel file: {file.filename}")
                        current_file_message += "Excel content extracted. "
            elif file.content_type in ["text/csv", "application/csv"]:
                import pandas as pd
                logger.info(f"Attempting to process CSV file in multiple upload: {file.filename} with content type: {file.content_type}")
                df_csv = pd.read_csv(file_location, header=None)
                all_text_content_csv = []
                for r_idx in range(len(df_csv)):
                    for c_idx in range(len(df_csv.columns)):
                        cell_value = df_csv.iat[r_idx, c_idx]
                        if pd.notna(cell_value):
                            all_text_content_csv.append(str(cell_value))
                extracted_text = "\n".join(all_text_content_csv)
                if not extracted_text.strip():
                    logger.warning(f"Pandas extracted no text from CSV: {file.filename} during multiple upload.")
                    current_file_message += "CSV processed, but no text extracted. "
                else:
                    logger.info(f"Extracted text from CSV file: {file.filename}")
                    current_file_message += "CSV content extracted. "
            else:
                logger.warning(f"Unsupported file type for {file.filename} in multiple upload: {file.content_type}")
                current_file_message += f"Unsupported file type ({file.content_type}). Skipped processing. "
                uploaded_filenames.append(file.filename) # Still count as received
                continue # Skip to next file if unsupported for processing

            if extracted_text.strip():
                # doc_processor and vector_db are now injected
                if not vector_db.collection:
                    logger.error(f"Vector DB service not available for {file.filename} in multiple upload. Skipping DB storage.")
                    current_file_message += "DB service unavailable. Skipped storage. "
                else:
                    cleaned_text = doc_processor.clean_text(extracted_text)
                    text_chunks_content = list(doc_processor.chunk_text_by_sentences(cleaned_text))

                    if text_chunks_content:
                        chunk_embeddings = doc_processor.get_embeddings(text_chunks_content)
                        processed_chunks_for_db = []
                        for chunk_idx, chunk_text_item in enumerate(text_chunks_content):
                            chunk_id = f"{doc_id_for_upload}_chunk_{chunk_idx}"
                            embedding = chunk_embeddings[chunk_idx] if chunk_idx < len(chunk_embeddings) else None
                            if embedding:
                                processed_chunks_for_db.append({
                                    "chunk_id": chunk_id, "doc_id": doc_id_for_upload,
                                    "text_chunk": chunk_text_item, "embedding": embedding,
                                    "metadata": {
                                        "source_file": file.filename, "original_content_type": file.content_type,
                                        "upload_source": "api_upload_multiple", "chunk_number": chunk_idx
                                    }
                                })
                        if processed_chunks_for_db:
                            vector_db.add_documents(processed_chunks_for_db)
                            current_file_message += f"Processed into {len(processed_chunks_for_db)} chunks and stored. "
                            current_file_processed_ids.append(doc_id_for_upload)
                        else:
                            current_file_message += "No chunks stored. "
                    else:
                        current_file_message += "No processable chunks from text. "
            else:
                 if file.content_type in ["text/plain", "application/pdf"]: # If it was a type we tried to extract from
                    current_file_message += "No text content to process. "


        except (DocumentProcessingError, LLMError) as proc_err:
            logger.error(f"Processing/LLM error for {file.filename} in multiple upload: {proc_err.message}")
            current_file_message += f"Processing error: {proc_err.message}. "
            # Consider re-raising or setting a file-specific error status if returning detailed per-file status
        except ImportError:
            logger.error(f"Pandas library not installed. Cannot process CSV {file.filename} in multiple upload.")
            current_file_message += "CSV processing skipped: Pandas not found. "
        except Exception as e: # Catch other errors like pd.read_csv failing for a specific file
            logger.error(f"Error processing file {file.filename} in multiple upload: {e}", exc_info=True)
            current_file_message += f"General error processing file: {str(e)}. "
        
        logger.info(f"Multiple upload status for {file.filename}: {current_file_message}")
        uploaded_filenames.append(file.filename) # Add to list of received files regardless of processing outcome

    if not uploaded_filenames: # Should not happen if initial check for files list passed
        raise HTTPException(status_code=400, detail="No valid files were processed in the multiple upload request.")

    # Consolidate processed IDs. For simplicity, just using the last one if multiple were processed.
    # A more robust approach would collect all successfully processed doc_ids.
    # For now, the response schema expects a list, so we can append all successful ones.
    # The current_file_processed_ids is reset per file, so this needs adjustment.
    # Let's assume for now the response will just list filenames and a general message.
    # A proper implementation would return a list of dicts, each with file status.

    final_processed_doc_ids = [] # This should be populated correctly based on successful processing of each file
    # For now, this remains a simplification. The `current_file_processed_ids` logic was per-file.
    # A better way is to accumulate `doc_id_for_upload` into `final_processed_doc_ids` if that file's processing was successful.
    # This requires a flag per file.

    # Simplified message for now:
    overall_message = f"{len(uploaded_filenames)} file(s) received. Check logs for individual processing status."
    if any("stored" in current_file_message for _ in range(len(files))): # Heuristic
        # This check is flawed as current_file_message is from the last file.
        # A better summary message would require tracking success per file.
        pass


    return schemas.DocumentUploadResponse(
        message=overall_message, # General message
        uploaded_files=uploaded_filenames, # All files received
        processed_doc_ids=[] # Placeholder: Needs proper aggregation of successfully processed IDs
    )

# Placeholder for listing documents (part of Document Management API)
@router.get("/documents", response_model=schemas.ListDocumentsResponse)
async def list_uploaded_documents(
    vector_db: VectorDBService = Depends(get_vector_db_serv)
):
    """
    Endpoint to list documents.
    Queries metadata from the vector DB.
    """
    # vector_db is now injected
    if not vector_db.collection:
        raise HTTPException(status_code=503, detail="Vector DB service is not available.")

    doc_list = []
    try:
        # Fetch all items to get their metadata. This could be inefficient for very large DBs.
        # Consider adding pagination or more specific querying to VectorDBService if needed.
        # The `get` method in ChromaDB can retrieve items.
        # We need to ensure we don't fetch embeddings unless necessary.
        # We are interested in unique doc_ids and their associated metadata.
        
        # Get all metadatas and ids. This might be large.
        # A more optimized way would be to have a separate metadata store or query distinct doc_ids.
        # For now, let's try to work with what ChromaDB's `get` offers.
        # The `get()` method without IDs fetches all items.
        all_items = vector_db.collection.get(include=["metadatas"]) # Only fetch metadatas
        
        unique_docs = {} # To store unique documents by doc_id
        if all_items and all_items.get('metadatas'):
            for metadata_item in all_items['metadatas']:
                if metadata_item: # Ensure metadata_item is not None
                    doc_id = metadata_item.get('doc_id')
                    if doc_id and doc_id not in unique_docs:
                        # Try to get a meaningful title
                        title = metadata_item.get('case_title') # From CSV
                        if not title:
                            title = metadata_item.get('source_file') # From upload or CSV
                        
                        source = metadata_item.get('upload_source') # e.g., 'api_upload', 'api_upload_multiple'
                        if not source:
                             source = "csv_ingest" if "csv" in (metadata_item.get('source_file') or "").lower() else "unknown"
                        
                        unique_docs[doc_id] = schemas.DocumentInfo(
                            doc_id=doc_id,
                            title=title or doc_id, # Fallback to doc_id if no title
                            source=source
                        )
            doc_list = list(unique_docs.values())
            logger.info(f"Retrieved {len(doc_list)} unique documents from VectorDB for listing.")
        else:
            logger.info("No items or metadatas found in VectorDB for listing.")

    except Exception as e:
        logger.error(f"Error retrieving document list from VectorDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list from database.")

    return schemas.ListDocumentsResponse(
        total_documents=len(doc_list),
        documents=doc_list
    )

logger.info("Document API router created with placeholder upload and list endpoints.")