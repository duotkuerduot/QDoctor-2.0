import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GROQ_DECISION_API_KEY = os.getenv("GROQ_DECISION_API_KEY", "")
    GROQ_GENERATOR_API_KEY = os.getenv("GROQ_GENERATOR_API_KEY", "")
    GROQ_VALIDATOR_API_KEY = os.getenv("GROQ_VALIDATOR_API_KEY", "")
    
    # Paths
    KB_PATH = "QBrain" 
    VECTOR_DB_PATH = "storage/qbrain_faiss_index"
    CACHE_PATH = "storage/cache_data"

    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.3-70b-versatile" 
    
    # Retrieval Settings
    TOP_K = 6
    SCORE_THRESHOLD = 0.4 

settings = Settings()