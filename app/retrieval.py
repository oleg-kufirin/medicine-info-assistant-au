import os, json
import numpy as np
import faiss

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer
from utils import to_web_url
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL = SentenceTransformer(EMBED_MODEL)

INDEX_DIR = "data/index"
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
MIN_SIMILARITY = 0.4
MIN_PASSAGES = 3
MAX_PASSAGES = 5


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
    with open(DOCS_PATH, "r", encoding="utf-8") as handle:
        docs = [json.loads(line) for line in handle]
    return index, docs


INDEX, DOCS = _load_index()


def _embed(texts: List[str]) -> np.ndarray:
    embeddings = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def retrieve(query: str, top_k: int = 20) -> List[Passage]:
    if INDEX is None:
        return []

    query_vector = _embed([query])
    distances, indices = INDEX.search(query_vector, top_k)
    idxs = indices[0].tolist()
    dists = distances[0].tolist()

    passages: List[Passage] = []
    for idx, score in zip(idxs, dists):
        if idx == -1:
            continue
        doc = DOCS[idx]
        passages.append(
            Passage(
                text=doc.get("text", "")[:4000],
                url=to_web_url(doc.get("source_url")),
                section=doc.get("section"),
                score=float(score),
            )
        )
    passages = sorted(passages, key=lambda item: item.score, reverse=True)
    filtered = [p for p in passages if p.score >= MIN_SIMILARITY]
    if len(filtered) >= MIN_PASSAGES:
        return filtered[:MAX_PASSAGES]
    return passages[:MIN_PASSAGES]


