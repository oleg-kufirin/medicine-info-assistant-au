import os, json, yaml, re, time, io
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "docs_raw")
INDEX_DIR = os.path.join(DATA_DIR, "index")
SEED = os.path.join(DATA_DIR, "seed_urls.yaml")
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def load_seed_urls() -> List[str]:
    with open(SEED,"r") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("urls", []))

def fetch(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    ctype = r.headers.get("content-type","").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return extract_pdf_text(r.content)
    else:
        return extract_html_text(r.text)

def extract_pdf_text(content: bytes) -> str:
    return pdf_extract_text(io.BytesIO(content))

def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    text = "\n".join(t.strip() for t in soup.stripped_strings if t.strip())
    return text

def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build():
    urls = load_seed_urls()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = []
    for url in urls:
        try:
            print(f"[INGEST] {url}")
            text = fetch(url)
            chunks = chunk_text(text)
            for i, ch in enumerate(chunks):
                docs.append({
                    "source_url": url,
                    "title": urlparse(url).netloc,
                    "section": f"section-{i+1}",
                    "text": ch
                })
        except Exception as e:
            print(f"[WARN] Failed {url}: {e}")

    # Write docs
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Build embeddings + FAISS
    texts = [d["text"] for d in docs]
    embs = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    print(f"[DONE] Indexed {len(docs)} chunks. Saved to {INDEX_PATH}")

if __name__ == "__main__":
    build()
