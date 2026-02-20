import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_DECISION_API_KEY = os.getenv("GROQ_DECISION_API_KEY")
    GROQ_GENERATOR_API_KEY = os.getenv("GROQ_GENERATOR_API_KEY")
    GROQ_VALIDATOR_API_KEY = os.getenv("GROQ_VALIDATOR_API_KEY")
    _raw_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_BOT_TOKEN = _raw_token[3:] if _raw_token.startswith("bot") else _raw_token
    
    TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    KB_PATH = os.getenv("KB_PATH", "QBrain")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "storage", "qbrain_faiss_index")
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
    LLM_MODEL = "llama-3.1-8b-instant" 

    try:
        TOP_K = int(os.getenv("TOP_K", 5))
    except (ValueError, TypeError):
        TOP_K = 5

settings = Settings()