from groq import Groq
from config.settings import settings

class QueryExpander:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    def generate_variations(self, original_query: str) -> list:
        """
        Generates 3 distinct search queries based on the user's input.
        1. GLOBAL CLINICAL: For symptoms, definitions, and standard medical facts (WHO, NICE).
        2. LOCAL PROTOCOL: For Kenya-specific laws, forms, or guidelines.
        3. KEYWORD: A simplified search string.
        """
        system_prompt = (
            "You are a medical search optimization assistant. "
            "Your goal is to break down the User's query into 3 distinct search queries "
            "to ensure we find both global clinical facts AND local Kenyan context.\n\n"
            "Generate these 3 queries:\n"
            "1. GLOBAL_CLINICAL: Search for medical signs, symptoms, or definitions using international standards (WHO, ICD-11, DSM-5). Do NOT add 'Kenya' here.\n"
            "2. LOCAL_PROTOCOL: Search for Kenya-specific guidelines, Mental Health Act, legal forms, or referral pathways.\n"
            "3. KEYWORD_SEARCH: A simple, keyword-heavy version of the query.\n\n"
            "Output ONLY the 3 queries, separated by newlines. No numbering, no labels."
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
            
            # Robust parsing: Split by newline and remove empty strings
            content = response.choices[0].message.content.strip()
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Fallback if the model returns nothing useful
            return queries[:3] if queries else [original_query]
            
        except Exception as e:
            print(f"Query Expansion failed: {e}")
            return [original_query]