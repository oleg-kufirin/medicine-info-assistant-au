from __future__ import annotations
import os
import logging

from typing import Any, Dict, List, Sequence
from dotenv import load_dotenv
from utils import load_prompt

_LC_AVAILABLE = True
try:
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - optional dependency
    _LC_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

class SummaryWritingAgent:
    """Agent for summarizing text passages using a language model."""

    def __init__(self) -> None:
        self._chain: Any = None
        self._rewrite_chain: Any = None

    @staticmethod
    def _format_passages_for_summary(passages: Sequence[Any]) -> str:
        """Format passages into a single string for the summarization prompt."""
        chunks: List[str] = []
        passage_scores: List[float] = []

        # Per-passage snippet cap (normalized chars) used when formatting LLM context
        # Large values risk exceeding model limits; set to 0 to disable
        per_snippet_limit = int(os.getenv("SUMMARY_PASSAGE_SNIPPET_LIMIT", "4000"))
        context_budget = int(os.getenv("SUMMARY_MAX_CONTEXT_CHARS", "20000"))


        if context_budget == 0:
            context_budget = 1_000_000_000  # effectively unlimited
        header_overhead = 64  # rough allowance per chunk for labels/meta

        for idx, passage in enumerate(passages, start=1):
            # Capture passage score when available (object or dict)
            raw_score = passage.get("score") if isinstance(passage, dict) else getattr(passage, "score", None)
            passage_scores.append(float(raw_score))

            text = getattr(passage, "text", "") or ""
            snippet = str(text).strip()
            if not snippet:
                continue

            snippet = " ".join(snippet.split())

            # Enforce per-snippet and total-context limits
            current_len = sum(len(c) for c in chunks)
            remaining = context_budget - current_len - header_overhead
            if remaining <= 0:
                break
            per_limit = remaining if per_snippet_limit == 0 else per_snippet_limit
            allowed = min(per_limit, max(0, remaining))
            snippet = snippet[:allowed]

            meta: List[str] = []
            section = getattr(passage, "section", None)
            if section:
                meta.append(f"section: {section}")

            url = getattr(passage, "url", None)
            if url:
                meta.append(f"url: {url}")

            meta_suffix = f" ({'; '.join(meta)})" if meta else ""
            chunks.append(f"Passage {idx}{meta_suffix}:\n{snippet}")

        result = "\n\n".join(chunks)
        # Log all passage scores for visibility
        logger.info("Summary passages scores: %s", ", ".join(f"{s:.4f}" for s in passage_scores))

        return result

    def _get_summary_chain(self):
        """Get or create the summarization chain."""
        if self._chain is not None:
            return self._chain
        if not _LC_AVAILABLE:
            return None

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model = os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant")
        llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)

        system_prompt = load_prompt("system_summary")
        if not system_prompt:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    (
                        "Answer the following question using the text passages.\n\n"
                        "Question: {query}\n\n"
                        "Passages:\n{context}"
                    ),
                ),
            ]
        )

        self._chain = prompt | llm
        return self._chain

    def _get_rewrite_chain(self):
        """Get or create the rewrite chain used for revisions."""
        if self._rewrite_chain is not None:
            return self._rewrite_chain
        if not _LC_AVAILABLE:
            return None

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model = os.getenv("SUMMARY_REWRITE_MODEL", os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant"))
        llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)

        system_prompt = load_prompt("system_summary_rewrite")
        if not system_prompt:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    (
                        "Question: {query}\n\n"
                        "Retrieved Passages:\n{context}\n\n"
                        "Original Draft Summary:\n{draft}\n\n"
                        "Critique Notes:\n{critique}\n\n"
                        "External Context:\n{external_context}"
                    ),
                ),
            ]
        )

        parser = StrOutputParser()
        self._rewrite_chain = prompt | llm | parser
        return self._rewrite_chain

    def summarize_passages(self, query: str, passages: Sequence[Any]) -> str | None:
        """Summarize the given passages in the context of the query."""
        if not passages:
            return None

        chain = self._get_summary_chain()
        if chain is None:
            return None

        context = self._format_passages_for_summary(passages)
        if not context:
            return None

        try:
            res = chain.invoke({"query": query, "context": context})
        except Exception:
            logger.warning("Summarization failed; returning None", exc_info=True)
            return None

        # Log token usage if available
        meta = getattr(res, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        logger.info(f"Number of tokens used: prompt - {prompt_tokens}, completion - {completion_tokens}")

        # Extract summary text
        summary = StrOutputParser().invoke(res)

        if isinstance(summary, str):
            summary_text = summary.strip()
        else:
            summary_text = getattr(summary, "content", "").strip()
        return summary_text or None

    @staticmethod
    def _format_critique(critique: Dict[str, Any] | None) -> str:
        if not critique:
            return "No critique provided."

        parts: List[str] = []
        issues = critique.get("issues") or []
        if issues:
            parts.append("Issues:")
            parts.extend(f"- {str(issue).strip()}" for issue in issues if str(issue).strip())

        instructions = str(critique.get("revision_instructions", "") or "").strip()
        if instructions:
            parts.append("\nRevision Instructions:")
            parts.append(instructions)

        needs_context = critique.get("needs_additional_context", False)
        parts.append(f"\nNeeds Additional Context: {bool(needs_context)}")
        return "\n".join(part for part in parts if part.strip()) or "No critique provided."

    @staticmethod
    def _format_external_context(external_context: List[Dict[str, Any]] | None) -> str:
        if not external_context:
            return "No external context supplied."
        sections: List[str] = []
        for item in external_context:
            title = str(item.get("title", "") or "").strip()
            summary = str(item.get("summary", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            query = str(item.get("query", "") or "").strip()
            details = []
            if query:
                details.append(f"Search query: {query}")
            if url:
                details.append(f"URL: {url}")
            header = title or query or "Wikipedia Entry"
            meta = f" ({'; '.join(details)})" if details else ""
            body = summary if summary else "No summary available."
            sections.append(f"{header}{meta}\n{body}")
        return "\n\n".join(sections)
    
    def rewrite_summary(
        self,
        query: str,
        passages: Sequence[Any],
        draft_summary: str | None,
        critique: Dict[str, Any] | None,
        external_context: List[Dict[str, Any]] | None = None,
    ) -> str | None:
        """Generate a revised summary using critique notes and external context."""
        draft = (draft_summary or "").strip()
        if not draft:
            return None

        chain = self._get_rewrite_chain()
        if chain is None:
            return None

        context = self._format_passages_for_summary(passages) if passages else ""
        critique_text = self._format_critique(critique)
        external_text = self._format_external_context(external_context)

        try:
            revision = chain.invoke(
                {
                    "query": query,
                    "context": context,
                    "draft": draft,
                    "critique": critique_text,
                    "external_context": external_text,
                }
            )
        except Exception:
            return None

        if isinstance(revision, str):
            revised_text = revision.strip()
        else:
            revised_text = getattr(revision, "content", "").strip()
        return revised_text or None
