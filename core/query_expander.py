from groq import Groq
from config.settings import settings

class QueryExpander:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    def generate_variations(self, original_query: str) -> list:
        """
        Generates 3 distinct search queries based on the user's input.
        1. Keyword-heavy (for BM25)
        2. Semantic/Concept-based (for Vector)
        3. Specific Form/Legal Reference (if applicable)
        """
        system_prompt = (
            "You are a search optimization assistant for a Kenyan Mental Health RAG system. "
            "Your goal is to break down the User's complex scenario into 3 specific, distinct search queries "
            "that will help retrieve the correct legal forms, clinical guidelines, or policy documents. "
            "Output ONLY the 3 queries, separated by newlines. No numbering."
        )

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_query}
                ],
                model=settings.LLM_MODEL,
                temperature=0.3,
            )
            # Split by newline and clean up
            queries = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
            # Ensure we always have at least the original query if something fails
            return queries[:3] if queries else [original_query]
        except Exception as e:
            print(f"Query Expansion failed: {e}")
            return [original_query]