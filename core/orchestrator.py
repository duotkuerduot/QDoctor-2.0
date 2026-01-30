from core.decision_engine import DecisionEngine
from storage.vector_store import QBrainVectorStore 
from storage.cache import QCache
from synthesis.generator import AnswerGenerator 
from evaluation.validator import HallucinationChecker

class Orchestrator:
    def __init__(self):
        # Initializing all defined classes
        self.classifier = DecisionEngine()
        self.kb = QBrainVectorStore() 
        self.cache = QCache()
        self.generator = AnswerGenerator()
        self.validator = HallucinationChecker()

    def process_query(self, user_query: str):
        print(f"\n--- Processing: {user_query} ---")
        
        # 1. Mental Health Check (uses DecisionEngine)
        print("1. Checking Intent...")
        is_mh = self.classifier.is_mental_health_related(user_query)
        if not is_mh:
            return "I only specialize in answering mental health–related questions."

        # 2. Cache Check (uses QCache)
        print("2. Checking Cache...")
        cached_answer = self.cache.get(user_query)
        if cached_answer:
            print("-> Cache Hit")
            return cached_answer

        # 3. Retrieval (uses QBrainVectorStore)
        print("3. Retrieving Context...")
        context = self.kb.retrieve(user_query)
        
        # 4. Synthesis (uses AnswerGenerator)
        print("4. Generating Answer...")
        raw_answer = self.generator.generate_answer(user_query, context)

        # 5. Hallucination Check (uses HallucinationChecker)
        print("5. Validating Answer...")
        is_valid = self.validator.check(context, raw_answer)
        
        if not is_valid:
            print("-> Hallucination Detected or No Context Found.")
            return "I apologize, but I cannot find any verified information within my sources to answer this safely."

        # 6. Cache Write & Return
        print("6. Caching and Returning...")
        self.cache.set(user_query, raw_answer)
        return raw_answer