import os
from storage.vector_store import QBrainVectorStore 

if __name__ == "__main__":
    print("Starting Hybrid Ingestion...")
    os.makedirs("storage/qbrain_faiss_index", exist_ok=True)
    
    # This will now create BOTH the FAISS index and the docs_cache.pkl for BM25
    kb = QBrainVectorStore()
    kb.ingest_data()
    
    print("Ingestion Done! FAISS index and BM25 cache created.")