from core.decision_engine import DecisionEngine
from core.query_expander import QueryExpander  # <-- NEW IMPORT
from storage.vector_store import QBrainVectorStore 
from storage.cache import QCache
from synthesis.generator import AnswerGenerator 
from evaluation.validator import HallucinationChecker

class Orchestrator:
    def __init__(self):
        self.classifier = DecisionEngine()
        self.expander = QueryExpander()  # <-- Initialize Expander
        self.kb = QBrainVectorStore() 
        self.cache = QCache()
        self.generator = AnswerGenerator()
        self.validator = HallucinationChecker()

    def process_query(self, user_query: str):
        print(f"\n--- Processing: {user_query} ---")
        
        # 1. Mental Health Check
        print("1. Checking Intent...")
        is_mh = self.classifier.is_mental_health_related(user_query)
        if not is_mh:
            return "I only specialize in answering mental health–related questions."

        # 2. Cache Check
        print("2. Checking Cache...")
        cached_answer = self.cache.get(user_query)
        if cached_answer:
            print("-> Cache Hit")
            return cached_answer

        # 3. Query Expansion (The Scenario Solver)
        print("3. Expanding Query...")
        search_queries = self.expander.generate_variations(user_query)
        print(f"   -> Generated Queries: {search_queries}")

        # 4. Hybrid Retrieval Loop
        print("4. Retrieving Context (Hybrid)...")
        all_docs = []
        for q in search_queries:
            # This now calls the Ensemble (BM25 + FAISS) retriever
            docs = self.kb.retrieve(q) 
            all_docs.extend(docs)

        # Deduplicate documents based on content to avoid processing duplicates
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        context_docs = list(unique_docs)
        print(f"   -> Retrieved {len(context_docs)} unique chunks.")
        
        # 5. Synthesis
        print("5. Generating Answer...")
        raw_answer = self.generator.generate_answer(user_query, context_docs)

        # 6. Hallucination Check
        print("6. Validating Answer...")
        is_valid = self.validator.check(context_docs, raw_answer)
        
        if not is_valid:
            print("-> Hallucination Detected.")
            return "I apologize, but I cannot find verified information to answer this safely."

        # 7. Cache & Return
        print("7. Caching and Returning...")
        self.cache.set(user_query, raw_answer)
        return raw_answer