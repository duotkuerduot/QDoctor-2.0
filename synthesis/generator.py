from groq import Groq
from config.settings import settings

class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    def generate_answer(self, question: str, context_docs: list) -> str:
        if not context_docs:
            return "Based on my available sources I could not find specific information regarding your query."

        # Format context
        context_str = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in context_docs])

        system_prompt = (
            "You are QDoctor, a specialized mental health assistant. "
            "Answer the user question strictly using the provided context below. "
            "Do not use outside knowledge. If the answer is not in the context, say you don't know. "
            "Be empathetic but professional."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.LLM_MODEL,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"