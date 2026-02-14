import os
import pickle
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from config.settings import settings

VECTOR_STORE_PATH = settings.VECTOR_DB_PATH
DOCS_CACHE_PATH = os.path.join(settings.BASE_DIR, "storage", "docs_cache.pkl")

class QBrainVectorStore:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def ingest_data(self):
        """Rebuilds the index (FAISS + BM25) from QBrain folder."""
        if not os.path.exists(settings.KB_PATH):
            print(f"Directory {settings.KB_PATH} not found.")
            return

        print("Loading documents...")
        loaders = [
            DirectoryLoader(settings.KB_PATH, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(settings.KB_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        ]
        
        docs = []
        for loader in loaders:
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading docs: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        if chunks:
            print(f"Embedding {len(chunks)} chunks...")
            
            # 1. Build & Save FAISS (Vector)
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
            self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # 2. Save Chunks for BM25 (Keyword)
            # We pickle the chunks so we can rebuild BM25 quickly on restart
            with open(DOCS_CACHE_PATH, "wb") as f:
                pickle.dump(chunks, f)
                
            print("Ingestion complete (Vector + Keyword Data saved).")
        else:
            print("No documents found to ingest.")

    def load_retrievers(self):
        """Loads both FAISS and BM25 retrievers."""
        if self.ensemble_retriever:
            return

        print("Loading retrievers...")
        
        # 1. Load FAISS
        self.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # 2. Load BM25
        if os.path.exists(DOCS_CACHE_PATH):
            with open(DOCS_CACHE_PATH, "rb") as f:
                chunks = pickle.load(f)
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = 4
        else:
            print("Warning: No BM25 cache found. Run setup.py again.")
            # Fallback to just FAISS if BM25 fails
            self.ensemble_retriever = faiss_retriever
            return

        # 3. Create Ensemble (Hybrid RRF)
        # weights=[0.4, 0.6] means we trust Vector (0.6) slightly more than Keyword (0.4)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]
        )

    def retrieve(self, query: str):
        """Retrieve relevant chunks using Hybrid Search."""
        if not self.ensemble_retriever:
            self.load_retrievers()
        return self.ensemble_retriever.invoke(query)