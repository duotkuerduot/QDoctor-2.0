import os
import pickle
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from config.settings import settings

# Paths
VECTOR_STORE_PATH = settings.VECTOR_DB_PATH
BM25_STORE_PATH = os.path.join(os.path.dirname(VECTOR_STORE_PATH), "bm25_chunks.pkl")

class QBrainVectorStore:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = None
        self.bm25_retriever = None
        
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(BM25_STORE_PATH):
            self.load()

    def ingest_data(self):
        """Rebuilds the index from QBrain folder."""
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
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
            self.vector_store.save_local(VECTOR_STORE_PATH)
            with open(BM25_STORE_PATH, "wb") as f:
                pickle.dump(chunks, f)
            self.load()
            print("Ingestion complete. Hybrid System Ready.")
        else:
            print("No documents found to ingest.")

    def load(self):
        """Load the FAISS vector store and BM25 index from disk."""
        try:
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            with open(BM25_STORE_PATH, "rb") as f:
                chunks = pickle.load(f)
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = 5  # Default top k for keyword
            
            print("Successfully loaded Hybrid Retrievers.")
        except Exception as e:
            print(f"Error loading indices: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Custom Weighted Ensemble Retrieval (RRF).
        Merges Semantic (FAISS) and Keyword (BM25) results.
        """
        if not self.vector_store or not self.bm25_retriever:
            print("Indices not loaded. Attempting load...")
            self.load()
            if not self.vector_store: return []

        semantic_docs = self.vector_store.similarity_search(query, k=k)
        keyword_docs = self.bm25_retriever.invoke(query)
        

        fused_scores = {}
        
        def apply_rrf(docs, weight):
            for rank, doc in enumerate(docs):
                doc_content = doc.page_content
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = {"doc": doc, "score": 0.0}
                
                score = weight * (1 / (rank + 60))
                fused_scores[doc_content]["score"] += score

        apply_rrf(semantic_docs, weight=0.6)
        apply_rrf(keyword_docs, weight=0.4)

        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return [item["doc"] for item in sorted_results[:k]]