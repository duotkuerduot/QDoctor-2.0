from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CitationReference(BaseModel):
    citation_id: str
    marker: str
    title: str
    tier: str
    chunk_id: str
    page: Optional[int] = None
    source_url: Optional[str] = None
    pdf_path: Optional[str] = None
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "marker": self.marker,
            "title": self.title,
            "tier": self.tier,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "source_url": self.source_url,
            "pdf_path": self.pdf_path,
            "snippet": self.snippet,
        }


class AnswerPayload(BaseModel):
    answer_text: str = ""
    citations: List[CitationReference] = Field(default_factory=list)

    def to_dict(self, include_legacy: bool = True) -> Dict[str, object]:
        payload = {
            "answer_text": self.answer_text,
            "citations": [citation.to_dict() for citation in self.citations],
        }
        if include_legacy:
            payload["answer"] = self.answer_text
            payload["sources"] = [citation.to_dict() for citation in self.citations]
        return payload

    def source_links(self) -> List[str]:
        links: List[str] = []
        for citation in self.citations:
            link = citation.source_url or citation.pdf_path
            if link:
                links.append(link)
        return links
