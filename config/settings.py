import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_DECISION_API_KEY = os.getenv("GROQ_DECISION_API_KEY")
    GROQ_GENERATOR_API_KEY = os.getenv("GROQ_GENERATOR_API_KEY")
    GROQ_VALIDATOR_API_KEY = os.getenv("GROQ_VALIDATOR_API_KEY")


    KB_PATH = os.getenv("KB_PATH", "QBrain")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "storage/qbrain_faiss_index")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    

    try:
        TOP_K = int(os.getenv("TOP_K", 5))
    except (ValueError, TypeError):
        TOP_K = 5