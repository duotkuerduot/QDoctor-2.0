import json
from groq import Groq
from config.settings import settings

class QueryAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_DECISION_API_KEY)

    def analyze(self, user_input: str) -> dict:
        """
        Analyzes intent and generates a single optimized search query.
        Returns JSON: {"intent": "GREETING"|"MENTAL_HEALTH"|"INVALID", "search_query": "..."}
        """
        system_prompt = """
        You are the core analyzer for QDoctor, a mental health AI.
        Analyze the user's input.
        Return a JSON object with exactly two keys:
        1. "intent": Must be "GREETING", "MENTAL_HEALTH", or "INVALID" (for non-mental health medical or random topics like cars/math).
        2. "search_query": If "MENTAL_HEALTH", provide ONE highly optimized search query combining clinical terms and local context. If "GREETING" or "INVALID", leave as an empty string "".
        Respond ONLY with valid JSON.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                model=settings.LLM_MODEL,
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Analyzer Error: {e}")
            return {"intent": "MENTAL_HEALTH", "search_query": user_input}