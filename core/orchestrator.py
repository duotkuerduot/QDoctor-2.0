from core.analyzer import QueryAnalyzer 
from storage.vector_store import QBrainVectorStore 
from evaluation.validator import HallucinationChecker
from storage.cache import QCache
from synthesis.generator import AnswerGenerator 

class Orchestrator:
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.kb = QBrainVectorStore() 
        self.cache = QCache()
        self.validator = HallucinationChecker()
        self.generator = AnswerGenerator()

    def process_query(self, user_query: str):
        print(f"\n--- Production Flow: {user_query} ---")
        
        # 1. Cache Check
        cached_answer = self.cache.get(user_query)
        if cached_answer:
            print("-> Cache Hit")
            return cached_answer

        # 2. Analyze Intent & Optimize Query
        analysis = self.analyzer.analyze(user_query)
        intent = analysis.get("intent", "MENTAL_HEALTH")
        search_query = analysis.get("search_query", user_query)

        if intent == "INVALID":
            return "I only specialize in answering mental health–related questions. How can I support your mental well-being today?"
        
        # 3. Hybrid Retrieval
        context_docs = []
        if intent == "MENTAL_HEALTH":
            print(f"-> Retrieving clinical context for: {search_query}")
            context_docs = self.kb.retrieve(search_query)
        else:
            print("-> Detected Greeting. Skipping Vector Search.")

        # 4. Synthesis (Llama 3 Summarization)
        raw_answer = self.generator.generate_answer(user_query, context_docs)

        # 5. VALIDATE
        if intent == "MENTAL_HEALTH" and context_docs:
            try:
                is_valid = self.validator.check(context_docs, raw_answer)
                if not is_valid:
                    print(f"CRITICAL: Hallucination blocked for query: {user_query}")
                    return "I'm sorry, I couldn't verify that information against clinical protocols. Please consult the official Kenya Mental Health Policy 2015-2030."
            except Exception as e:
                # Fail-safe: If the validator rate-limits, block the output for safety
                print(f"Validation Error: {e}")
                return "The clinical verification system is temporarily overloaded. For safety, please rephrase your question or try again in a moment."

        # 6. Cache & Return
        self.cache.set(user_query, raw_answer)
        return raw_answer