import os
from typing import Optional

def load_prompt(name: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Load a prompt text by name from app/prompts.

    - Accepts either a bare name (e.g., "system_safety_classifier") or a filename
      (e.g., "system_safety_classifier.txt").
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
