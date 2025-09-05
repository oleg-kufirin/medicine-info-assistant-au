from typing import List, Dict
import os, json
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Simple local vector store wrapper for MVP
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL = SentenceTransformer(EMBED_MODEL)

INDEX_DIR = "data/index"
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

@dataclass
class Passage:
    text: str
    url: str | None = None
    section: str | None = None
    score: float = 0.0

def _load_index():
    if not (os.path.exists(DOCS_PATH) and os.path.exists(INDEX_PATH)):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    docs = [json.loads(line) for line in open(DOCS_PATH,"r",encoding="utf-8")]
    return index, docs

INDEX, DOCS = _load_index()

def _embed(texts: List[str]) -> np.ndarray:
    embs = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def retrieve(query: str, top_k: int = 20) -> List[Passage]:
    if INDEX is None:
        return []
    q = _embed([query])
    D, I = INDEX.search(q, top_k)
    idxs = I[0].tolist()
    ds = D[0].tolist()
    passages: List[Passage] = []
    for i, d in zip(idxs, ds):
        if i == -1: 
            continue
        doc = DOCS[i]
        passages.append(Passage(text=doc.get("text","")[:4000],
                                url=doc.get("source_url"),
                                section=doc.get("section"),
                                score=float(d)))
    # Naive rerank: keep top 5
    passages = sorted(passages, key=lambda p: p.score, reverse=True)[:5]
    return passages

def synthesize_answer(query: str, passages: List[Passage]) -> Dict:
    if not passages:
        return {
            "answer_text": "I couldn’t find that in my indexed PBS/TGA sources. Try searching the official PBS medicine finder.",
            "bullets": [],
            "citations": [],
            "disclaimer": "General information only, not medical advice."
        }
    # For MVP, just stitch together a compact answer with top passage snippets
    bullets = []
    citations = []
    for p in passages:
        snippet = p.text.strip().split("\n")[0][:300]
        bullets.append(snippet)
        citations.append({"url": p.url or "", "section": p.section or ""})
    return {
        "answer_text": f"Here’s what I found about your question: {query}",
        "bullets": bullets[:3],
        "citations": citations[:3],
        "disclaimer": "General information only, not medical advice."
    }
