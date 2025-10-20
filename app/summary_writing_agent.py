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
        self._last_error: Dict[str, Any] | None = None


    def get_last_error(self) -> Dict[str, Any] | None:
        """Return the last error encountered during summarization/rewrite."""
        return self._last_error


    def write_summary(self, query: str, preformatted_passages: str | None = None) -> str | None:
        """Summarize the given passages in the context of the query."""
        # Reset last error on each call
        self._last_error = None

        if not preformatted_passages:
            return None

        chain = self._get_summary_chain()
        if chain is None:
            return None

        # Invoke the chain and handle exceptions
        try:
            result = chain.invoke({"query": query, "context": preformatted_passages})
        except Exception as e:
            # Capture and log the error
            err_msg = str(e)
            error_payload: Dict[str, Any] = {"kind": "unknown", "message": err_msg}
            # Check for Groq API-specific errors
            import groq
            if isinstance(e, groq.APIStatusError):
                status = getattr(e, "status_code", None)
                error_payload = {"kind": "groq_api_error", "status_code": status, "message": err_msg, }
            logger.warning("Summarization failed: %s; returning None", err_msg)
            self._last_error = error_payload
            return None

        if isinstance(result, str):
            summary_text = result.strip()
        else:
            summary_text = getattr(result, "content", "").strip()
        return summary_text or None


    def rewrite_summary(self, query: str, draft_summary: str | None, critique: Dict[str, Any] | None, preformatted_passages: str | None = None,) -> str | None:
        """Generate a revised summary using critique notes."""
        draft = (draft_summary or "").strip()
        if not draft:
            return None

        chain = self._get_rewrite_chain()
        if chain is None:
            return None

        critique_text = self._format_critique(critique)

        try:
            revision = chain.invoke({"query": query, "context": preformatted_passages, "draft": draft, "critique": critique_text,})
        except Exception:
            return None

        if isinstance(revision, str):
            revised_text = revision.strip()
        else:
            revised_text = getattr(revision, "content", "").strip()
        return revised_text or None


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

        parser = StrOutputParser()
        self._chain = prompt | llm | parser
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
                        "Critique Notes:\n{critique}"
                    ),
                ),
            ]
        )

        parser = StrOutputParser()
        self._rewrite_chain = prompt | llm | parser
        return self._rewrite_chain

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

        return "\n".join(part for part in parts if part.strip()) or "No critique provided."



    
