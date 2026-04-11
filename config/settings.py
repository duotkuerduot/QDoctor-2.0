# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_DECISION_API_KEY = os.getenv("GROQ_DECISION_API_KEY")
    GROQ_GENERATOR_API_KEY = os.getenv("GROQ_GENERATOR_API_KEY")
    GROQ_VALIDATOR_API_KEY = os.getenv("GROQ_VALIDATOR_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    KB_PATH = os.getenv("KB_PATH", "QBrain")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "storage", "qbrain_faiss_index")
    BACKEND_URL = os.getenv("BACKEND_URL", "https://duotkuerduot-qdoctor.hf.space")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = "llama-3.1-8b-instant" 

    try:
        TOP_K = int(os.getenv("TOP_K", 5))
    except (ValueError, TypeError):
        TOP_K = 5

settings = Settings()
