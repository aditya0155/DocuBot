import streamlit as st
import requests
import json # For pretty printing JSON if needed for debugging
import os # Import os to access environment variables

# Configuration for Backend API URL
# Use the environment variable if available (set on Render), otherwise default to localhost for local dev
BASE_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000") # Render provides the full URL e.g. https://docubot-backend-feda.onrender.com/

# Construct specific endpoint URLs
# .rstrip('/') ensures that if the BASE_API_URL has a trailing slash, it's removed before appending the specific path,
# preventing double slashes like 'https://...//api/v1'.
QUERY_API_URL = f"{BASE_API_URL.rstrip('/')}/api/v1/query"
DOCUMENTS_API_URL = f"{BASE_API_URL.rstrip('/')}/api/v1/documents"
UPLOAD_MULTIPLE_API_URL = f"{DOCUMENTS_API_URL}/upload-multiple"

# BACKEND_URL is now replaced by specific endpoint URLs like QUERY_API_URL

st.set_page_config(page_title="DocuBot - Document Research", layout="wide")

st.title("📚 Document Research & Theme Identifier")
st.markdown("""
Welcome to DocuBot! Ask questions about the loaded documents, and I'll try to find relevant information and common themes.
""")

# --- Helper function to call backend ---
def query_backend(query_text: str, n_results: int = 5):
    """
    Sends a query to the backend API and returns the response.
    """
    payload = {
        "query_text": query_text,
        "n_results": n_results
    }
    try:
        response = requests.post(QUERY_API_URL, json=payload, timeout=120) # Increased timeout for potentially long LLM calls
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error: Could not connect to the backend at {QUERY_API_URL}. Ensure the backend is running.")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error: The request to the backend timed out.")
    except requests.exceptions.RequestException as req_err:
        st.error(f"An unexpected error occurred with the request: {req_err}")
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON response from backend: {response.text}")
    return None

# --- Main Application UI ---
st.sidebar.header("Query Options")
num_results = st.sidebar.slider("Number of relevant chunks to retrieve:", min_value=1, max_value=20, value=5, step=1, key="num_results_slider")

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload document(s) (TXT, PDF, CSV, XLS, XLSX, PNG, JPG, TIFF)", 
    type=["txt", "pdf", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "tiff", "tif"],
    accept_multiple_files=True, 
    key="doc_uploader_multiple"
)

if uploaded_files: # This will be a list of UploadedFile objects
    if st.sidebar.button(f"Process {len(uploaded_files)} selected file(s)", key="process_upload_button_multiple"):
        if not uploaded_files: # Should not happen if button is shown, but good practice
            st.sidebar.warning("No files selected.")
        else:
            # Prepare files for the 'upload-multiple' endpoint
            files_payload = []
            for uploaded_file_item in uploaded_files:
                files_payload.append(('files', (uploaded_file_item.name, uploaded_file_item.getvalue(), uploaded_file_item.type)))
            
            # upload_url = "http://localhost:8000/api/v1/documents/upload-multiple" # Now using UPLOAD_MULTIPLE_API_URL
            
            with st.spinner(f"Uploading and processing {len(uploaded_files)} file(s)..."):
                try:
                    response = requests.post(UPLOAD_MULTIPLE_API_URL, files=files_payload, timeout=300) # Increased timeout for multiple files
                    response.raise_for_status()
                    upload_response_data = response.json()
                    st.sidebar.success(f"{len(uploaded_files)} file(s) processed!")
                    st.sidebar.write("Response from backend:")
                    st.sidebar.json(upload_response_data) 
                    # Consider refreshing the document list here if desired, e.g., by re-running or using st.experimental_rerun
                except requests.exceptions.HTTPError as http_err:
                    st.sidebar.error(f"Upload HTTP error: {http_err} - {response.text}")
                except requests.exceptions.RequestException as req_err:
                    st.sidebar.error(f"Upload error: {req_err}")
                except Exception as e:
                    st.sidebar.error(f"An unexpected error during upload: {e}")

st.sidebar.markdown("---")
# Display list of documents
st.sidebar.header("📚 Processed Documents")
# documents_list_url = "http://localhost:8000/api/v1/documents" # Now using DOCUMENTS_API_URL
try:
    doc_response = requests.get(DOCUMENTS_API_URL, timeout=30)
    doc_response.raise_for_status()
    docs_data = doc_response.json()
    if docs_data and docs_data.get("documents"):
        # Sort documents, perhaps by source then title/doc_id
        sorted_docs = sorted(docs_data["documents"], key=lambda x: (x.get("source", "z"), x.get("title", "")))
        for doc in sorted_docs:
            st.sidebar.markdown(f"- **{doc.get('title', doc.get('doc_id', 'Unknown Document'))}** (`ID: {doc.get('doc_id', 'N/A')}` Source: *{doc.get('source', 'N/A')}*)")
    else:
        st.sidebar.caption("No documents found or an error occurred.")
except requests.exceptions.RequestException as e:
    st.sidebar.caption(f"Could not fetch document list: {e}")
except Exception as e: # Catch any other unexpected error during doc list fetching
    st.sidebar.caption(f"Error displaying documents: {e}")


st.sidebar.markdown("---")
st.sidebar.info("This is a demo application for document research and theme identification using AI.")


st.header("🔍 Ask a Question")
user_query = st.text_area("Enter your question about the documents:", height=100, key="user_query_input")

if st.button("🚀 Get Answers & Themes", key="submit_query_button"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your query... This may take a moment. 🧠"):
            backend_response = query_backend(user_query, n_results=num_results)

        if backend_response:
            st.success("Query processed successfully!")
            
            # Display Original Query
            st.subheader(f"🔎 Your Query:")
            st.markdown(f"> {backend_response.get('original_query', 'N/A')}")
            st.divider()

            # Display Individual Answers
            st.subheader("📄 Individual Answers from Documents")
            individual_answers = backend_response.get("individual_answers", [])
            if individual_answers:
                answers_for_table = []
                for ans in individual_answers:
                    answers_for_table.append({
                        "Document ID": ans.get('doc_id', 'N/A'),
                        "Extracted Answer": ans.get('extracted_answer', 'No answer extracted.'),
                        "Citation": ans.get('citation_source', 'No citation provided.'),
                        # "Relevance": f"{ans.get('relevance_score', 'N/A'):.4f}" if isinstance(ans.get('relevance_score'), float) else ans.get('relevance_score', 'N/A'),
                        # "Source Chunk": ans.get('text_chunk', '') # Optional: too long for table
                    })
                
                # Using st.dataframe for better interactivity if needed, or st.table for simpler static table
                st.dataframe(answers_for_table, use_container_width=True)
            else:
                st.info("No individual answers were found for your query.")
            
            st.divider()

            # Display Themes
            st.subheader("💡 Identified Themes")
            themes = backend_response.get("themes", [])
            if themes:
                for i, theme in enumerate(themes):
                    st.markdown(f"#### Theme {i+1}: {theme.get('theme_name', 'Unnamed Theme')}")
                    st.markdown(f"**Summary:** {theme.get('theme_summary', 'No summary provided.')}")
                    st.markdown(f"**Supporting Document IDs:** `{', '.join(theme.get('supporting_doc_ids', []))}`")
                    st.markdown("---")
            else:
                st.info("No common themes were identified for your query.")
        else:
            st.error("Failed to get a response from the backend. Please check the logs or try again later.")

# To run this Streamlit app:
# 1. Ensure the FastAPI backend is running.
# 2. Navigate to the `frontend` directory.
# 3. Run `streamlit run app.py` in your terminal.
