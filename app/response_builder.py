from __future__ import annotations
from typing import Dict, List, Any
from retrieval_agent import Passage


EMPTY_RESULT = {
    "summary_text": "I could not find that in the indexed CMI/PI documents. Please check the official product materials or speak with a pharmacist.",
    "bullets": [],
    "citations": [],
    "disclaimer": "General information only, not medical advice.",
}


def synthesize_answer(query: str, passages: List["Passage"], summary_text: str | None = None) -> Dict:
    """Turn retrieved passages into a UI-friendly answer payload."""
    if not passages:
        return EMPTY_RESULT.copy()

    bullets: List[Dict[str, str | float]] = []
    citations: List[Dict[str, str]] = []

    for passage in passages:
        snippet = passage.text.strip().split("\n")[0][:800]
        if snippet:
            bullets.append({
                "text": snippet,
                "score": float(passage.score),
                "drug_name": getattr(passage, "drug_name", None),
                "active_ingridients": getattr(passage, "active_ingridients", None),
            })
        citations.append({"url": passage.url or "", "section": passage.section or ""})

    return {
        "summary_text": summary_text,
        "bullets": bullets,
        "citations": citations,
        "disclaimer": "General information only, not medical advice.",
    }
