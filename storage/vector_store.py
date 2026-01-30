import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

VECTOR_STORE_PATH = settings.VECTOR_DB_PATH

class QBrainVectorStore:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.store = None

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
            self.store = FAISS.from_documents(chunks, self.embedding_model)
            self.store.save_local(VECTOR_STORE_PATH)
            print("Ingestion complete.")
        else:
            print("No documents found to ingest.")

    def load(self):
        """Load the FAISS vector store from disk."""
        self.store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 3):
        """Retrieve relevant chunks."""
        if not self.store:
            self.load()
        return self.store.similarity_search(query, k=k)