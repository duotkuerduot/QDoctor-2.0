import os
from storage.vector_store import QBrainVectorStore 

if __name__ == "__main__":
    print("Starting ingestion...")
    os.makedirs("storage/qbrain_faiss_index", exist_ok=True)
    kb = QBrainVectorStore()
    kb.ingest_data()
    print("Done!")