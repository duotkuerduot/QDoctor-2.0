from groq import Groq
from config.settings import settings

class DecisionEngine:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_DECISION_API_KEY)

    def is_mental_health_related(self, question: str) -> bool:
        """
        Determines if the question is related to mental health.
        Returns: True (Yes) or False (No).
        """
        system_prompt = (
            "You are a strict classifier. Your ONLY job is to determine if the user input "
            "is related to mental health, psychology, or psychiatry. "
            "Reply with exactly one word: 'YES' or 'NO'."
        )
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                model=settings.LLM_MODEL,
                temperature=0, 
            )
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
        except Exception as e:
            print(f"Classification Error: {e}")
            return False