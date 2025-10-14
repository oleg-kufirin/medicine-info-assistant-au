from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import quote


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from langchain_community.utilities import WikipediaAPIWrapper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        from langchain.utilities import WikipediaAPIWrapper  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        WikipediaAPIWrapper = None  # type: ignore[misc,assignment]


@dataclass
class WikipediaResult:
    """Structured representation of a Wikipedia lookup."""

    query: str
    title: str
    summary: str
    url: str


class WikipediaLookupTool:
    """Thin wrapper around LangChain's WikipediaAPIWrapper."""

    def __init__(self, language: str = "en", top_k: int = 2) -> None:
        if WikipediaAPIWrapper is None:
            raise RuntimeError(
                "LangChain WikipediaAPIWrapper is unavailable. Install `langchain-community` "
                "or `langchain` with Wikipedia support."
            )
        try:
            self._wrapper = WikipediaAPIWrapper(lang=language, top_k_results=top_k)  # type: ignore[arg-type]
        except TypeError:
            # Older signatures do not accept keyword params.
            self._wrapper = WikipediaAPIWrapper()  # type: ignore[call-arg]

    def lookup(self, query: str) -> Optional[WikipediaResult]:
        """Run a single Wikipedia query."""
        normalized = (query or "").strip()
        if not normalized:
            return None

        try:
            documents = self._wrapper.load(normalized)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - network/config dependent
            logger.debug("LangChain Wikipedia lookup failed", exc_info=exc)
            return None

        if not documents:
            return None

        document = documents[0]
        metadata = getattr(document, "metadata", {}) or {}
        summary = str(metadata.get("summary") or getattr(document, "page_content", "") or "").strip()
        if not summary:
            return None

        title = str(
            metadata.get("title")
            or metadata.get("displaytitle")
            or metadata.get("query")
            or metadata.get("pageid")
            or normalized
        ).strip()

        url = str(metadata.get("source") or metadata.get("url") or "").strip()
        if not url:
            url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

        return WikipediaResult(
            query=normalized,
            title=title,
            summary=summary,
            url=url,
        )

    def batch_lookup(self, queries: Sequence[str]) -> List[WikipediaResult]:
        """Run multiple lookups, skipping duplicates and empty queries."""
        seen: set[str] = set()
        results: List[WikipediaResult] = []
        for query in queries:
            normalized = (query or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)

            result = self.lookup(normalized)
            if result:
                results.append(result)
        return results

