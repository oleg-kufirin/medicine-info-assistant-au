"""PBS/TGA ingestion and indexing.

- Fetches PBS pages from the web using URLs in 'data/seed_urls.yaml'.
- Loads TGA CMI/PI documents from local PDFs under data/downloads'.
- Chunks extracted text and writes 'data/index/docs.jsonl' (if missing).
- Builds a FAISS index at 'data/index/faiss.index' from the chunked docs.
- If 'docs.jsonl' already exists, reuses it and only rebuilds the index.
- Supports optional external drug metadata mapping via DRUG_MAPPING_PATH.
"""

import os, json, yaml, re, time, io
import requests
import faiss
import numpy as np
import asyncio, json, httpx

from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "index")
SEED_URLS = os.path.join(DATA_DIR, "seed_urls.yaml")
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
TGA_PDF_DIR = os.path.join(DATA_DIR, "downloads")  # Local folder with TGA PDFs
DRUG_MAPPING_PATH = os.getenv("DRUG_MAPPING_PATH", os.path.join(DATA_DIR, "drug_mapping.yaml"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL = SentenceTransformer(EMBED_MODEL_NAME)


def load_seed_urls() -> List[str]:
    """Load seed URLs from YAML config.

    Returns:
        List[str]: URLs to ingest
    """
    with open(SEED_URLS,"r") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("urls", []))


def fetch(url: str) -> str:
    """Fetch a URL and extract visible text. Handles HTML and PDF responses. 

    Args:
        url: The URL to fetch.

    Returns:
        str: Extracted plain text.
    """
    client = httpx.Client(follow_redirects=True, timeout=30)
    r = client.get(url)
    # Fallback to requests for robustness
    try:
        r.raise_for_status()
    except Exception:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    ctype = r.headers.get("content-type","").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return extract_pdf_text(r.content)
    else:
        return extract_html_text(r.text)


def _load_drug_mapping(path: str) -> Dict[str, Dict[str, Any]]:
    """Load external drug metadata mapping.

    Supports JSON or YAML based on filename extension. The mapping format is a
    flat dictionary where keys are identifiers (exact URL, filename, or a
    substring to match) and values are objects with:
      - "drug_name": str
      - "active_ingridients": List[str]

    Example (YAML):
      "paracetamol_cmi.pdf":
        drug_name: "Panadol"
        active_ingridients: ["paracetamol"]

    Returns an empty mapping if the file is missing or malformed.
    """
    # Load mapping file
    if not path or not os.path.exists(path):
        print(f"[INFO] Drug mapping not found at {path}. Proceeding without metadata.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                raw = yaml.safe_load(f) or {}
            else:
                raw = json.load(f) or {}
    except Exception as e:
        print(f"[WARN] Failed to load drug mapping ({path}): {e}")
        return {}

    # Normalize mapping structure
    mapping: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            dn = v.get("drug_name")
            ai = v.get("active_ingridients")

            # drug_name: single non-empty string or None
            if isinstance(dn, (str, int, float)):
                dn_norm: Optional[str] = str(dn).strip() or None
            else:
                dn_norm = None

            # active_ingridients: list of non-empty strings
            if isinstance(ai, list):
                ai_norm = [str(x).strip() for x in ai if str(x).strip()]
            elif isinstance(ai, (str, int, float)):
                ai_norm = [str(ai).strip()] if str(ai).strip() else []
            else:
                ai_norm = []

            mapping[str(k).strip()] = {
                "drug_name": dn_norm,
                "active_ingridients": ai_norm,
            }
    else:
        print(f"[WARN] Unsupported mapping structure in {path}; expecting a dict.")
    return mapping


def _resolve_metadata(source_url: str, title: str, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve metadata for a document from an external mapping.

    Matching priority:
      1) Exact key match on full source_url
      2) Exact key match on filename (basename of source_url or local path)
      3) Case-insensitive substring match of any mapping key within URL, filename, or title
    Returns a dict with keys: "drug_name" (str|None), "active_ingridients" (List[str]).
    """
    if not mapping:
        return {"drug_name": None, "active_ingridients": []}

    url = str(source_url or "")
    title_s = str(title or "")
    filename = os.path.basename(url) if "://" in url else os.path.basename(source_url)

    # Prepare for case-insensitive matching
    url_l = url.lower()
    fn_l = str(filename or "").lower()
    title_l = title_s.lower()

    # 1) Exact URL
    if url in mapping:
        return mapping[url]

    # 2) Exact filename
    if filename and filename in mapping:
        return mapping[filename]

    # 3) Substring fallback
    for key, meta in mapping.items():
        k = str(key).strip()
        if not k:
            continue
        k_l = k.lower()
        if k_l in url_l or (fn_l and k_l in fn_l) or (title_l and k_l in title_l):
            return meta

    return {"drug_name": None, "active_ingridients": []}


def extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF byte stream using pdfminer.six.

    Args:
        content: Raw PDF bytes.

    Returns:
        str: Extracted text.
    """
    return pdf_extract_text(io.BytesIO(content))


def extract_html_text(html: str) -> str:
    """Extract visible text from HTML, stripping scripts/styles.

    Args:
        html: HTML content as a string.

    Returns:
        str: Concatenated visible text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()    
    text = "\n".join(t.strip() for t in soup.stripped_strings if t.strip())
    return text


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List[str]: List of chunk strings.
    """
    words = text.split()
    print(f"Text length: {len(words)}")
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def list_local_tga_pdfs(directory: str) -> List[str]:
    """Recursively list local TGA PDF files under a directory.

    Args:
        directory: Root directory to search

    Returns:
        List[str]: Sorted absolute file paths to PDFs.
    """
    if not os.path.isdir(directory):
        return []
    pdfs = []
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    return sorted(pdfs)


def read_pdf_file(path: str) -> str:
    """Open a local PDF and extract text.

    Args:
        path: Path to a PDF file on disk.

    Returns:
        str: Extracted text.
    """
    with open(path, "rb") as f:
        content = f.read()
    return extract_pdf_text(content)


def create_docs(url_ignore: bool = False, mapping_path: Optional[str] = None) -> List[Dict]:
    """Ingest TGA (local PDFs) and PBS (web) and write docs.jsonl.

    - If not URL ignore,
        fetches PBS/other non-TGA URLs from `data/seed_urls.yaml`.
    - Loads TGA CMI/PI content from local PDFs under `data/downloads`.
    - Chunks extracted text into sections and writes to `data/index/docs.jsonl`.

    Returns:
        List[Dict]: The list of document chunks written to docs.jsonl, each with
        keys:   "source_url" (URL or file path) 
                "title" (hostname or filename)
                "section" (chunk index)
                "text" (chunk content).
    """
    # Prepare dirs
    urls = load_seed_urls()
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = []
    # Load external mapping (optional)
    mapping = _load_drug_mapping(mapping_path or DRUG_MAPPING_PATH)

    # Ingest web sources (PBS)
    if not url_ignore:
        for url in urls:
            try:
                print(f"[INGEST: WEB] {url}")
                text = fetch(url)
                chunks = chunk_text(text)
                print(f"Chunks: {len(chunks)}")
                for i, ch in enumerate(chunks):
                    meta = _resolve_metadata(url, urlparse(url).netloc, mapping)
                    docs.append({
                        "source_url": url,
                        "title": urlparse(url).netloc,
                        "section": f"section-{i+1}",
                        "drug_name": meta.get("drug_name"),
                        "active_ingridients": meta.get("active_ingridients", []),
                        "text": ch,
                    })
            except Exception as e:
                print(f"[WARN][WEB] Failed {url}: {e}")

    # Ingest local PDFs (TGA)
    tga_pdfs = list_local_tga_pdfs(TGA_PDF_DIR)
    if not tga_pdfs:
        print(f"[WARN] No TGA PDFs found in {TGA_PDF_DIR}")
    for pdf_path in tga_pdfs:
        try:
            print(f"[INGEST: PDF] {pdf_path}")
            text = read_pdf_file(pdf_path)
            chunks = chunk_text(text)
            print(f"Chunks: {len(chunks)}")
            for i, ch in enumerate(chunks):
                title = os.path.basename(pdf_path)
                meta = _resolve_metadata(pdf_path, title, mapping)
                docs.append({
                    "source_url": pdf_path,
                    "title": title,
                    "section": f"section-{i+1}",
                    "drug_name": meta.get("drug_name"),
                    "active_ingridients": meta.get("active_ingridients", []),
                    "text": ch,
                })
        except Exception as e:
            print(f"[WARN][TGA] Failed {pdf_path}: {e}")

    # Write docs
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Simple summary
    with_meta = sum(1 for d in docs if d.get("drug_name") or d.get("active_ingridients"))
    print(f"[DONE] Created {len(docs)} docs; {with_meta} annotated with drug metadata")

    return docs


def build_embeddings_index():
    """Build and save a FAISS index from docs.jsonl.

    - Loads existing docs from 'data/index/docs.jsonl' if present; 
      otherwise generates them by calling 'create_docs().
    - Encodes doc texts with the SentenceTransformer specified by EMBED_MODEL
    - Builds an inner-product FAISS index and saves it to 'data/index/faiss.index'.
    """
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        print(f"[DONE] Loaded {len(docs)} documents from {DOCS_PATH}")
    else:
        docs = create_docs(url_ignore=True, mapping_path=DRUG_MAPPING_PATH)

    # Build embeddings + FAISS
    texts = [d["text"] for d in docs]
    embs = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    print(f"[DONE] Indexed {len(docs)} chunks. Saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_embeddings_index()
