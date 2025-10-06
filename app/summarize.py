from __future__ import annotations

import os
from typing import Any, List, Sequence
from dotenv import load_dotenv
from utils import load_prompt

_LC_AVAILABLE = True
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    _LC_AVAILABLE = False

# Number of characters to keep from each passage for summarization
PASSAGE_SNIPPET_LIMIT = 800

load_dotenv()

def _format_passages_for_summary(passages: Sequence[Any]) -> str:
    chunks: List[str] = []
    # loop through passages and format them
    for idx, passage in enumerate(passages, start=1):
        # extract text
        text = getattr(passage, "text", "") or ""
        snippet = str(text).strip() #strip leading and trailing whitespaces
        if not snippet:
            continue
        snippet = " ".join(snippet.split()) # collapse extra whitespace
        snippet = snippet[:PASSAGE_SNIPPET_LIMIT] # keep only first 800 chars

        # collect metadata
        meta: List[str] = []
        section = getattr(passage, "section", None)
        if section:
            meta.append(f"section: {section}")

        url = getattr(passage, "url", None)
        if url:
            meta.append(f"url: {url}")

        # build suffix with metadata
        meta_suffix = f" ({'; '.join(meta)})" if meta else ""
        # build formatted passage entry
        chunks.append(f"Passage {idx}{meta_suffix}:\n{snippet}")
    return "\n\n".join(chunks)


def _get_summary_chain():
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
            ("user", 
                (
                    "Answer the following question using the text passages.\n\n"
                    "Question: {query}\n\n"
                    "Passages:\n{context}"    
                )
            ),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


def _summarize_passages(query: str, passages: Sequence[Any]) -> str | None:
    chain = _get_summary_chain()
    if chain is None:
        return None

    context = _format_passages_for_summary(passages)
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


def summarize_passages(query: str, passages: Sequence[Any]) -> str | None:
    """Public wrapper to summarize passages via the LLM chain."""
    if not passages:
        return None
    return _summarize_passages(query, passages)


