from __future__ import annotations

import os
import logging

from typing import Any, Dict, List, Sequence
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from utils import load_prompt

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logger = logging.getLogger(__name__)

# Define the structured output model for reflection
class ReflectionPayload(BaseModel):
    revision_instructions: str = Field(
        default="", 
        description="Guidance for improving the summary."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues spotted in the draft summary.",
    )


class ReflectionAgent:
    """Agent that critiques a summary draft and identifies missing context."""

    PASSAGE_SNIPPET_LIMIT = 1000

    def __init__(self) -> None:
        self._chain: Any = None
        self._last_error: Dict[str, Any] | None = None


    def get_last_error(self) -> Dict[str, Any] | None:
        """Return the last error encountered during reflection, if any."""
        return self._last_error


    def review_summary(self, query: str, passages: Sequence[Any], summary_text: str | None, ) -> Dict[str, Any]:
        """Critique the summary and recommend follow-up actions."""
        if not summary_text:
            return self._default_response()

        # Get or create the chain
        chain = self._get_chain()
        if chain is None:
            return self._default_response()

        # Format the context passages for the prompt
        context = self._format_passages(passages) if passages else ""

        # Run the reflection chain
        try:
            result = chain.invoke({"query": query, "context": context, "summary": summary_text})
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
            return self._default_response()
            
         # Extract parsed payload and validate the output
        reflection = result.model_dump()

        return reflection or self._default_response()


    def _get_chain(self) -> Any:
        """Create (or reuse) the reflection chain."""
        if self._chain is not None:
            return self._chain

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model = os.getenv("REFLECTION_MODEL", os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant"))
        llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)
        structured_llm = llm.with_structured_output(ReflectionPayload, method="json_mode")

        system_prompt = load_prompt("system_reflection")
        if not system_prompt:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    (
                        "Question: {query}\n\n"
                        "Draft Summary:\n{summary}\n\n"
                        "Retrieved Passages:\n{context}"
                    ),
                ),
            ]
        )

        self._chain = prompt | structured_llm
        return self._chain


    @staticmethod
    def _format_passages(passages: Sequence[Any]) -> str:
        """Create a condensed text block describing the retrieved passages."""
        chunks: List[str] = []
        for idx, passage in enumerate(passages, start=1):
            if isinstance(passage, dict):
                text = passage.get("text", "") or ""
                url = passage.get("url")
                section = passage.get("section")
            else:
                text = getattr(passage, "text", "") or ""
                url = getattr(passage, "url", None)
                section = getattr(passage, "section", None)

            snippet = " ".join(str(text).strip().split())
            if not snippet:
                continue
            # snippet = snippet[: ReflectionAgent.PASSAGE_SNIPPET_LIMIT]

            meta: List[str] = []
            if section:
                meta.append(f"section: {section}")
            if url:
                meta.append(f"url: {url}")

            meta_suffix = f" ({'; '.join(meta)})" if meta else ""
            chunks.append(f"Passage {idx}{meta_suffix}:\n{snippet}")
        return "\n\n".join(chunks)


    @staticmethod
    def _default_response() -> Dict[str, Any]:
        """Fallback response when reflection cannot run."""
        return ReflectionPayload().model_dump()
