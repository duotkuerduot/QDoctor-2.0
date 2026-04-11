import json
import os
import re
from urllib.parse import quote
from typing import Dict, List, Optional

from core.analyzer import QueryAnalyzer
from core.schemas import AnswerPayload, CitationReference
from evaluation.validator import HallucinationChecker
from storage.cache import QCache
from storage.vector_store import QBrainVectorStore
from synthesis.generator import AnswerGenerator


class Orchestrator:
    CITABLE_TIERS = {"tier_1", "tier_2"}

    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.kb = QBrainVectorStore()
        self.cache = QCache()
        self.validator = HallucinationChecker()
        self.generator = AnswerGenerator()

    @staticmethod
    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def _stream_text(text: str):
        """Yields SSE token events word-by-word."""
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield f"event: token\ndata: {json.dumps(chunk)}\n\n"

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return (base_url or "https://qdoctor-ai.vercel.app/").rstrip("/")

    @staticmethod
    def _clean_doc_names(file_name: str) -> tuple[str, str]:
        base_name = os.path.basename(file_name or "Unknown.pdf")
        clean_name = re.sub(r"\.pdf$", "", base_name, flags=re.IGNORECASE)
        return base_name, clean_name

    @staticmethod
    def _build_pdf_path(file_name: str, page_num: int) -> str:
        from config.settings import settings

        kb_root = settings.KB_PATH
        if not os.path.isabs(kb_root):
            kb_root = os.path.join(settings.BASE_DIR, kb_root)

        absolute_file = file_name
        if not os.path.isabs(absolute_file):
            absolute_file = os.path.join(settings.BASE_DIR, file_name)

        try:
            relative_path = os.path.relpath(absolute_file, kb_root)
        except ValueError:
            relative_path = os.path.basename(file_name)

        encoded_path = quote(relative_path.replace("\\", "/"))
        return f"/pdfs/{encoded_path}#page={page_num}"

    @staticmethod
    def _safe_page(page_value) -> Optional[int]:
        try:
            return int(page_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _guess_tier(source_name: str) -> str:
        normalized = (source_name or "").replace("\\", "/").lower()
        if "kenya moh mental health resources" in normalized or "kenya legal & policy framework" in normalized:
            return "tier_1"
        if "nice guidelines" in normalized or "who mhgap" in normalized:
            return "tier_2"
        if "kenya" in normalized:
            return "tier_1"
        return "tier_3"

    @staticmethod
    def _normalize_answer_text(answer_text: str) -> str:
        return (answer_text or "").strip()

    @classmethod
    def _is_citation_ready_answer(cls, answer_text: str) -> bool:
        normalized = cls._normalize_answer_text(answer_text).lower()
        if not normalized:
            return False

        blocked_fragments = (
            "error generating answer:",
            "i only specialize in answering mental health-related questions",
            "i couldn't verify that information clinically",
            "temporarily overloaded",
            "I cannot find this information in the current clinical guidelinesas per my sources",
        )
        return not any(fragment in normalized for fragment in blocked_fragments)

    @classmethod
    def _build_citation_catalog(self, context_docs: list) -> List[CitationReference]:
        from config.settings import settings

        base_url = self._normalize_base_url(settings.BACKEND_URL)
        citations: List[CitationReference] = []
        seen_chunk_ids = set()

        for index, doc in enumerate(context_docs, 1):
            file_name = doc.metadata.get("source", "Unknown.pdf")
            base_name, clean_name = self._clean_doc_names(file_name)
            page_num = self._safe_page(doc.metadata.get("page")) or 1
            tier = doc.metadata.get("tier") or self._guess_tier(file_name)
            if tier not in self.CITABLE_TIERS:
                continue

            chunk_id = doc.metadata.get("chunk_id") or f"{clean_name.lower().replace(' ', '_')}-{page_num}-{index}"
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            pdf_path = self._build_pdf_path(file_name, page_num)
            citation = CitationReference(
                citation_id=f"cite_{len(citations) + 1}",
                marker=f"[C{len(citations) + 1}]",
                title=clean_name,
                tier=tier,
                chunk_id=chunk_id,
                page=page_num,
                source_url=f"{base_url}{pdf_path}",
                pdf_path=pdf_path,
                snippet=(doc.page_content or "")[:250].strip(),
            )
            citations.append(citation)

            doc.metadata["chunk_id"] = chunk_id
            doc.metadata["tier"] = tier
            doc.metadata["citation_marker"] = citation.marker
            doc.metadata["citation_id"] = citation.citation_id

        return citations

    @classmethod
    def _append_fallback_citations(
        cls,
        answer_text: str,
        citation_catalog: List[CitationReference],
    ) -> str:
        if not cls._is_citation_ready_answer(answer_text):
            return answer_text

        if any(citation.marker in answer_text for citation in citation_catalog):
            return answer_text

        if not citation_catalog:
            return answer_text

        fallback_markers = "".join(citation.marker for citation in citation_catalog[:2])
        if not fallback_markers:
            return answer_text

        separator = "" if answer_text.endswith((" ", "\n")) else " "
        return f"{answer_text}{separator}{fallback_markers}"

    @staticmethod
    def _extract_used_citations(
        answer_text: str,
        citation_catalog: List[CitationReference],
    ) -> List[CitationReference]:
        positions = []
        for citation in citation_catalog:
            index = answer_text.find(citation.marker)
            if index != -1:
                positions.append((index, citation))

        positions.sort(key=lambda item: item[0])
        return [citation for _, citation in positions]

    @staticmethod
    def _coerce_citation_item(item, fallback_index: int) -> Optional[CitationReference]:
        if isinstance(item, CitationReference):
            return item

        if isinstance(item, str):
            title = item.split("/")[-1].split("#")[0] or f"Source {fallback_index}"
            clean_title = re.sub(r"\.pdf$", "", title, flags=re.IGNORECASE)
            return CitationReference(
                citation_id=f"cached_{fallback_index}",
                marker=f"[C{fallback_index}]",
                title=clean_title,
                tier="tier_1",
                chunk_id=f"cached_{fallback_index}",
                source_url=item,
                pdf_path=item if item.startswith("/pdfs/") else None,
            )

        if not isinstance(item, dict):
            return None

        title = item.get("title") or item.get("name") or f"Source {fallback_index}"
        source_url = item.get("source_url") or item.get("url")
        pdf_path = item.get("pdf_path")
        return CitationReference(
            citation_id=item.get("citation_id") or f"cached_{fallback_index}",
            marker=item.get("marker") or f"[C{fallback_index}]",
            title=title,
            tier=item.get("tier") or Orchestrator._guess_tier(title),
            chunk_id=item.get("chunk_id") or f"cached_{fallback_index}",
            page=Orchestrator._safe_page(item.get("page")),
            source_url=source_url,
            pdf_path=pdf_path,
            snippet=item.get("snippet"),
        )

    @classmethod
    def _normalize_payload(cls, payload_data) -> AnswerPayload:
        if isinstance(payload_data, AnswerPayload):
            return payload_data

        if isinstance(payload_data, str):
            return AnswerPayload(answer_text=payload_data, citations=[])

        if not isinstance(payload_data, dict):
            return AnswerPayload(answer_text="", citations=[])

        answer_text = payload_data.get("answer_text") or payload_data.get("answer") or ""
        raw_citations = payload_data.get("citations")
        if raw_citations is None:
            raw_citations = payload_data.get("sources") or []

        citations: List[CitationReference] = []
        if isinstance(raw_citations, dict):
            for index, (key, meta) in enumerate(raw_citations.items(), 1):
                meta = meta if isinstance(meta, dict) else {"source_url": meta}
                citation = cls._coerce_citation_item(
                    {
                        "title": key,
                        "source_url": meta.get("source_url") or meta.get("url"),
                        "pdf_path": meta.get("pdf_path"),
                        "chunk_id": meta.get("chunk_id") or key,
                        "page": meta.get("page"),
                        "tier": meta.get("tier"),
                        "marker": meta.get("marker") or f"[C{index}]",
                        "snippet": meta.get("snippet"),
                    },
                    fallback_index=index,
                )
                if citation:
                    citations.append(citation)
        elif isinstance(raw_citations, list):
            for index, item in enumerate(raw_citations, 1):
                citation = cls._coerce_citation_item(item, fallback_index=index)
                if citation:
                    citations.append(citation)

        return AnswerPayload(answer_text=answer_text, citations=citations)

    @staticmethod
    def _build_response(answer_text: str, citations: List[CitationReference]) -> Dict:
        payload = AnswerPayload(answer_text=answer_text, citations=citations)
        return payload.to_dict(include_legacy=True)

    @staticmethod
    def _empty_response(answer_text: str) -> Dict:
        return AnswerPayload(answer_text=answer_text, citations=[]).to_dict(include_legacy=True)

    def process_query(
        self, user_query: str, user_id: str = None,
        intent_override: str = None, history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        print(f"\n--- Production Flow: {user_query} ---")

        if not history:
            cached_data = self.cache.get(user_query)
            if cached_data:
                return self._normalize_payload(cached_data).to_dict(include_legacy=True)

        analysis = self.analyzer.analyze(user_query)
        intent = analysis.get("intent", "MENTAL_HEALTH")
        search_query = analysis.get("search_query", user_query)

        if intent == "INVALID":
            return self._empty_response("I only specialize in answering mental health-related questions.")

        context_docs = []
        if intent == "MENTAL_HEALTH":
            context_docs = self.kb.retrieve(search_query)

        citation_catalog = self._build_citation_catalog(context_docs)
        raw_answer = self.generator.generate_answer(
            user_query,
            context_docs,
            citation_catalog=citation_catalog,
            history=history or [],
        )

        if intent == "MENTAL_HEALTH" and context_docs:
            try:
                if not self.validator.check(context_docs, raw_answer):
                    return self._empty_response("I couldn't verify that information clinically.")
            except Exception as e:
                print(f"Validation Error: {e}")
                return self._empty_response("The clinical verification system is temporarily overloaded. Please try again.")

        final_answer = self._append_fallback_citations(raw_answer, citation_catalog)
        used_citations = self._extract_used_citations(final_answer, citation_catalog)
        final_response = self._build_response(final_answer, used_citations)

        if not history:
            self.cache.set(user_query, final_response)
        return final_response

    def process_query_stream(
        self, user_query: str, user_id: str = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Yields SSE events: token, citations, response, replace.
        Does NOT yield 'done' because the caller (main.py) handles that
        after saving messages to the database.
        """
        print(f"\n--- Streaming Flow: {user_query} ---")

        if not history:
            cached_answer = self.cache.get(user_query)
            if cached_answer:
                print("-> Cache Hit (streaming)")
                payload = self._normalize_payload(cached_answer)
                if payload.answer_text:
                    yield from self._stream_text(payload.answer_text)
                if payload.citations:
                    citations_payload = [citation.to_dict() for citation in payload.citations]
                    yield self._sse("citations", citations_payload)
                    yield self._sse("sources", [citation.title for citation in payload.citations])
                yield self._sse("response", payload.to_dict(include_legacy=True))
                return

        analysis = self.analyzer.analyze(user_query)
        intent = analysis.get("intent", "MENTAL_HEALTH")
        search_query = analysis.get("search_query", user_query)

        if intent == "INVALID":
            msg = "I only specialize in answering mental health-related questions. How can I support your mental well-being today?"
            yield from self._stream_text(msg)
            yield self._sse("response", self._empty_response(msg))
            return

        context_docs = []
        if intent == "MENTAL_HEALTH":
            print(f"-> Retrieving clinical context for: {search_query}")
            context_docs = self.kb.retrieve(search_query)
        else:
            print("-> Detected Greeting. Skipping Vector Search.")

        citation_catalog = self._build_citation_catalog(context_docs)
        answer_parts: List[str] = []

        for chunk in self.generator.generate_answer_stream(
            user_query,
            context_docs,
            citation_catalog=citation_catalog,
            history=history or [],
        ):
            answer_parts.append(chunk)
            yield self._sse("token", chunk)

        full_answer = "".join(answer_parts)

        if intent == "MENTAL_HEALTH" and context_docs:
            try:
                if not self.validator.check(context_docs, full_answer):
                    print(f"CRITICAL: Hallucination blocked for: {user_query}")
                    blocked_message = "I'm sorry, I couldn't verify that information. Please consult a healthcare professional."
                    yield self._sse("replace", blocked_message)
                    yield self._sse("response", self._empty_response(blocked_message))
                    return
            except Exception as e:
                print(f"Validation Error: {e}")
                overload_message = "The clinical verification system is temporarily overloaded. Please try again."
                yield self._sse("replace", overload_message)
                yield self._sse("response", self._empty_response(overload_message))
                return

        final_answer = self._append_fallback_citations(full_answer, citation_catalog)
        if final_answer != full_answer:
            suffix = final_answer[len(full_answer):]
            if suffix:
                yield self._sse("token", suffix)

        used_citations = self._extract_used_citations(final_answer, citation_catalog)
        response_payload = self._build_response(final_answer, used_citations)

        if used_citations:
            citations_payload = [citation.to_dict() for citation in used_citations]
            yield self._sse("citations", citations_payload)
            yield self._sse("sources", [citation.title for citation in used_citations])

        yield self._sse("response", response_payload)

        if not history:
            self.cache.set(user_query, response_payload)