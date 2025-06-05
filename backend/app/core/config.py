import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

# Determine the path to the .env file relative to this config.py file
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

class Settings(BaseSettings):
    # Google Gemini API Configuration
    gemini_api_key: str = Field(..., validation_alias='GEMINI_API_KEY')
    gemini_chat_model_id: str = Field(default="gemini-1.5-flash-latest", validation_alias='GEMINI_CHAT_MODEL_ID')
    gemini_embedding_model_id: str = Field(default="gemini-embedding-exp-03-07", validation_alias='GEMINI_EMBEDDING_MODEL_ID')
    
    # Database Configuration
    chroma_db_path: str = Field(default="./backend/data/vector_store", validation_alias='CHROMA_DB_PATH')
    chroma_db_collection_name: str = Field(default="legal_documents", validation_alias='CHROMA_DB_COLLECTION_NAME')

    # Application Settings
    host: str = Field(default="0.0.0.0", validation_alias='HOST')
    port: int = Field(default=8000, validation_alias='PORT')
    log_level: str = Field(default="INFO", validation_alias='LOG_LEVEL')

    # Optional: Project name and API prefix (these are not from .env, so direct assignment is fine)
    project_name: str = "Document Research & Theme Identification Chatbot"
    api_v1_prefix: str = "/api/v1"

    model_config = {
        "env_file": ENV_PATH,
        "env_file_encoding": 'utf-8',
        "case_sensitive": True, # This means Pydantic looks for env var names matching field names or aliases case-sensitively.
        "extra": 'ignore'
    }

settings = Settings()

# Validation print statement (optional, for debugging)
if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
    print(f"Warning: GEMINI_API_KEY is not set correctly in {ENV_PATH} or environment variables.")
elif settings.gemini_api_key == "AIzaSyDvC7I7UjkP2ZlgSqeZ_ZyiAvJm5FQp0Z4":
     print(f"GEMINI_API_KEY loaded successfully via Pydantic settings: {settings.gemini_api_key[:10]}...") # Print part of the key
else:
    print(f"GEMINI_API_KEY loaded via Pydantic settings, but it does not match the expected test key.")

print(f"Loaded settings via Pydantic: Port={settings.port}, LogLevel={settings.log_level}, Chroma Path={settings.chroma_db_path}")