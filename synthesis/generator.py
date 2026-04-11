# synthesis/generator.py
import os
import re
from typing import List, Dict
from groq import Groq
from config.settings import settings
from core.schemas import CitationReference


class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_GENERATOR_API_KEY)

    def _build_prompts(
        self,
        question: str,
        context_docs: list,
        citation_catalog: List[CitationReference] = None,
    ) -> tuple[str, str]:
        """Build system and user prompts."""
        citation_catalog = citation_catalog or []
        citation_lookup = {citation.chunk_id: citation for citation in citation_catalog}

        if not context_docs:
            context_str = "No medical context required. This is a conversational greeting."
        else:
            cleaned_docs = []
            for doc in context_docs:
                source = doc.metadata.get("source", "Unknown Document")
                source_base = os.path.basename(source)
                source_clean = re.sub(r"\.pdf$", "", source_base, flags=re.IGNORECASE)
                page = doc.metadata.get("page", "")
                page_str = f", Page {page}" if page else ""
                chunk_id = doc.metadata.get("chunk_id", "")
                citation = citation_lookup.get(chunk_id)
                citation_lines = []
                if citation and citation.tier in {"tier_1", "tier_2"}:
                    citation_lines.extend(
                        [
                            f"Citation Marker: {citation.marker}",
                            f"Chunk ID: {citation.chunk_id}",
                            f"Evidence Tier: {citation.tier}",
                        ]
                    )
                cleaned_docs.append(
                    "\n".join(
                        [
                            f"Document: {source_clean}{page_str}",
                            *citation_lines,
                            f"Content: {doc.page_content}",
                        ]
                    )
                )
            context_str = "\n\n".join(cleaned_docs)

        system_prompt = (
            "You are an elite Clinical Evidence Assistant. Your goal is to provide high-precision, "
            "medically responsible information grounded strictly in provided clinical guidelines."

            # --- IDENTITY & SCOPE ---
            "1. SCOPE: Respond professionally to greetings. For medical/mental health queries, "
            "synthesize answers ONLY from provided context. If the answer is missing from the context, "
            "state: 'This information is not available in the current clinical guidelines.' Do not hallucinate."

            # --- DYNAMIC FORMATTING RULES ---
            "2. ADAPTIVE STRUCTURE: "
            "- SHORT DEFINITIONS: For simple definitions, provide a concise 1-2 paragraph response WITHOUT subheadings. "
            "- COMPLEX QUERIES: For management or legal questions, use **Bold Subheadings**. "
            "- MEDICAL STEPS: For step-by-step guidance, use indented bulleted or numbered lists."

            # --- CITATION RULES (OPEN-EVIDENCE STYLE) ---
            "3. CITATION CHIPS: "
            "- Every factual claim supported by Tier 1 or Tier 2 evidence MUST be followed by the exact citation marker supplied in the context, such as [C1]. "
            "- Never invent a citation marker or change its spelling. "
            "- If multiple sources apply, append multiple markers with no prose between them, e.g. [C1][C2]. "
            "- If no Tier 1 or Tier 2 context supports a statement, do not cite it and do not fabricate support. "
            "- These markers act as the trigger for the UI's link cards. "
            "- DO NOT output a 'References' or 'Sources' section at the end."
        
            # --- TONALITY & CONTEXT ---
            "4. RIGOR: Use formal, clinical language. Avoid conversational filler."
            "5. CONTINUITY: Use provided conversation history to maintain context."
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Input: {question}"
        return system_prompt, user_prompt

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: List[Dict[str, str]],
    ) -> list:
        """
        Build the full messages array for the LLM call.
        
        Layout:
          [system] -> [history msg 1] -> [history msg 2] -> ... -> [current user prompt with context]
        
        The history contains prior user/assistant turns.
        The final user message includes the RAG context + current question.
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (skip the last user message — that's the current one)
        # History from frontend includes all messages including the current user message,
        # so we take all but the last (which is the current query)
        prior_turns = history[:-1] if history else []

        for turn in prior_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        # Add the current user message with context
        messages.append({"role": "user", "content": user_prompt})

        return messages

    def generate_answer(
        self,
        question: str,
        context_docs: list,
        citation_catalog: List[CitationReference] = None,
        history: List[Dict[str, str]] = None,
    ) -> str:
        """Non-streaming method with conversation history."""
        system_prompt, user_prompt = self._build_prompts(
            question, context_docs, citation_catalog=citation_catalog
        )
        messages = self._build_messages(system_prompt, user_prompt, history or [])

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=settings.LLM_MODEL,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"

    def generate_answer_stream(
        self,
        question: str,
        context_docs: list,
        citation_catalog: List[CitationReference] = None,
        history: List[Dict[str, str]] = None,
    ):
        """Streaming method with conversation history."""
        system_prompt, user_prompt = self._build_prompts(
            question, context_docs, citation_catalog=citation_catalog
        )
        messages = self._build_messages(system_prompt, user_prompt, history or [])

        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=settings.LLM_MODEL,
                temperature=0.1,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"Error generating answer: {e}"
