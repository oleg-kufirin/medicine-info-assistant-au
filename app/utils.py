import os
from typing import Any, List, Optional, Sequence
from dotenv import load_dotenv

load_dotenv()

def load_prompt(name: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Load a prompt text by name from app/prompts.

    - Accepts either a bare name (e.g., "safety_moderation") or a filename
      (e.g., "safety_moderation.txt").
    - Returns the file contents stripped, or None if missing/empty.
    """
    if not name:
        return None

    # Ensure .txt suffix
    filename = name if name.endswith(".txt") else f"{name}.txt"

    # Resolve prompts directory
    if base_dir is None:
        # Default to the app/prompts directory so callers only pass the prompt name.
        base_dir = os.path.join(os.path.dirname(__file__), 'prompts')

    path = os.path.join(base_dir, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception:
        return None


def to_web_url(source_url: str | None) -> str | None:
    """Convert a stored source path into a shareable URL when possible."""
    if not source_url:
        return None

    # Trim whitespace
    url = source_url.strip()
    if not url:
        return None

    # If already a web URL, return as-is
    if url.startswith(("http://", "https://")):
        return url

    normalized = url.replace("\\", "/")
    basename = os.path.basename(normalized)
    identifier, _ = os.path.splitext(basename)
    if identifier:
        base = "https://www.ebs.tga.gov.au/ebs/picmi/picmirepository.nsf/pdf?OpenAgent=&id="
        return f"{base}{identifier}"
    return url


def format_passages_for_context(passages: Sequence[Any]) -> str:
    """Format passages into a unified LLM-ready context block.

    - Supports passages objects with attributes: text, section.
    - Normalizes whitespace and applies per-snippet and total context budgets.
    - Returns a single string with formatted passages.
    """
    # Default limits in characters
    PER_PASSAGE_LIMIT = 2000
    PASSAGE_CONTEXT_BUDGET = 20000
    HEADER_OVERHEAD = 64

    # Initialize limits from environment variables
    try:
        env_per_passage_limit = os.getenv("SINGLE_PASSAGE_CONTEXT_LIMIT", PER_PASSAGE_LIMIT)
        if env_per_passage_limit is not None:
            per_passage_limit = int(env_per_passage_limit)
            if per_passage_limit <= 0:
                per_passage_limit = 0  # interpret <=0 as unlimited
    except Exception:
        per_passage_limit = PER_PASSAGE_LIMIT
        pass

    try:
        env_passage_context_budget = os.getenv("TOTAL_PASSAGE_CONTEXT_BUDGET", PASSAGE_CONTEXT_BUDGET)
        if env_passage_context_budget is not None:
            passage_context_budget = int(env_passage_context_budget)
            if passage_context_budget <= 0:
                passage_context_budget = 0  # interpret <=0 as unlimited
    except Exception:
        passage_context_budget = PASSAGE_CONTEXT_BUDGET
        pass

    if passage_context_budget == 0:
        passage_context_budget = 1_000_000_000  # effectively unlimited

    chunks: List[str] = []

    # Process each passage
    for idx, passage in enumerate(passages, start=1):
        text = getattr(passage, "text", "") or ""
        section = getattr(passage, "section", None)

        snippet = " ".join(str(text).strip().split())
        if not snippet: continue

        # Enforce limits
        current_len = sum(len(c) for c in chunks)
        remaining = passage_context_budget - current_len - HEADER_OVERHEAD
        if remaining <= 0:
            break
        per_limit = remaining if per_passage_limit == 0 else per_passage_limit
        allowed = min(per_limit, max(0, remaining))
        snippet = snippet[:allowed]

        section_suffix = f" - {section}" if section else ""
        chunks.append(f"Passage {idx}{section_suffix}:\n{snippet}")

    return "\n\n".join(chunks)
