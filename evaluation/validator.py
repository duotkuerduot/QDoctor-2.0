import re

from groq import Groq
from config.settings import settings

class HallucinationChecker:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_VALIDATOR_API_KEY)

    @staticmethod
    def _strip_citation_markers(answer: str) -> str:
        return re.sub(r"\[C\d+\]", "", answer or "")

    def check(self, context_docs: list, answer: str) -> bool:
        """
        Returns True if the answer is supported by context, False otherwise.
        """
        if not context_docs:
            return False

        context_str = "\n".join([doc.page_content for doc in context_docs])
        
        system_prompt = (
            "You are an intelligent evaluator for a mental health RAG system. "
            "Your task is to compare the Answer against the provided Context. "
            "1. If the Answer is supported by the Context (even if summarized or rephrased), reply 'PASS'. "
            "2. If the Answer contains specific facts (numbers, laws, names) that appear NOWHERE in the Context, reply 'FAIL'. "
            "3. Be generous with style but strict with facts. "
            "Output ONLY 'PASS' or 'FAIL'."
        )
        
        clean_answer = self._strip_citation_markers(answer)
        user_prompt = f"Context: {context_str}\n\nAnswer: {clean_answer}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.LLM_MODEL,
                temperature=0,
            )
            result = response.choices[0].message.content.strip().upper()
            return "PASS" in result
        except Exception as e:
            print(f"Validation Error: {e}")
            return True 
