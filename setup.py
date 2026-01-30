from storage.vector_store import QBrainVectorStore 
if __name__ == "__main__":
    print("Starting ingestion...")
    kb = QBrainVectorStore()
    kb.ingest_data()
    print("Done!")