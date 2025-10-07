from __future__ import annotations
import os

from typing import Any, List, Sequence
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

class SummarizationAgent:
    """Agent for summarizing text passages using a language model."""

    PASSAGE_SNIPPET_LIMIT = 800

    def __init__(self) -> None:
        self._chain: Any = None

    @staticmethod
    def _format_passages_for_summary(passages: Sequence[Any]) -> str:
        """Format passages into a single string for the summarization prompt."""
        chunks: List[str] = []
        for idx, passage in enumerate(passages, start=1):
            text = getattr(passage, "text", "") or ""
            snippet = str(text).strip()
            if not snippet:
                continue

            snippet = " ".join(snippet.split())
            snippet = snippet[: SummarizationAgent.PASSAGE_SNIPPET_LIMIT]

            meta: List[str] = []
            section = getattr(passage, "section", None)
            if section:
                meta.append(f"section: {section}")

            url = getattr(passage, "url", None)
            if url:
                meta.append(f"url: {url}")

            meta_suffix = f" ({'; '.join(meta)})" if meta else ""
            chunks.append(f"Passage {idx}{meta_suffix}:\n{snippet}")
        return "\n\n".join(chunks)

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
            summary = chain.invoke({"query": query, "context": context})
        except Exception:
            return None

        if isinstance(summary, str):
            summary_text = summary.strip()
        else:
            summary_text = getattr(summary, "content", "").strip()
        return summary_text or None
