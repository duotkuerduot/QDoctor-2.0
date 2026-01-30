"""
Answer synthesis for QBrain RAG system.
"""
from typing import List
from langchain_core.documents import Document
from groq import Groq
from config.settings import settings

FALLBACK_MESSAGE = "Sorry, I am unable to answer this question at the moment."

class QBrainAnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    async def synthesize_answer_llm(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generate an answer from retrieved QBrain chunks using Groq LLM. If none, return fallback.
        """
        if not retrieved_docs:
            return FALLBACK_MESSAGE
        context = "\n\n".join([doc.page_content.strip() for doc in retrieved_docs])
        prompt = f"Answer the following question using ONLY the provided context.\n\nQuestion: {query}\n\nContext:\n{context}\n\nIf the answer is not in the context, say 'Sorry, this information is not available in the knowledge base.'"
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a clinical mental health assistant. Cite sources and use supportive, professional language."},
                    {"role": "user", "content": prompt}
                ],
                model=settings.LLM_MODEL,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq answer generation error: {e}")
            return FALLBACK_MESSAGE