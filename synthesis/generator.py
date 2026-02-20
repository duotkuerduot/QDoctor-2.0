import re
from groq import Groq
from config.settings import settings

class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    def generate_answer(self, question: str, context_docs: list) -> str:
        if not context_docs:
            context_str = "No medical context required. This is a conversational greeting."
        else:
            # Format context, remove .pdf, and add page numbers
            cleaned_docs = []
            for doc in context_docs:
                source = doc.metadata.get('source', 'Unknown')
                source_clean = re.sub(r'\.pdf$', '', source, flags=re.IGNORECASE)
                page = doc.metadata.get('page', '')
                page_str = f", Page {page}" if page else ""
                
                cleaned_docs.append(f"Source: {source_clean}{page_str}\nContent: {doc.page_content}")
            context_str = "\n\n".join(cleaned_docs)

        system_prompt = (
            "You are QDoctor, an empathetic and specialized mental health assistant. "
            "1. If the user is greeting you or making small talk, respond warmly and naturally. "
            "2. If answering a medical/mental health question, use ONLY the provided context. "
            "Do not hallucinate. If the answer is not in the context, say you are unable to best answer the question based on your sources. "
            "3. CITATION RULES: For medical answers, you MUST cite your sources at the end of the information using the provided document name and page number (e.g., 'Source: Kenya Mental Health Policy, Page 12')."
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Input: {question}"

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