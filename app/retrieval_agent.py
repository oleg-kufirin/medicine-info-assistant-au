import os, json
import numpy as np
import faiss

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from typing import List, Sequence, Tuple
from sentence_transformers import SentenceTransformer
from utils import to_web_url

load_dotenv()


@dataclass
class Passage:
    text: str
    url: str | None = None
    section: str | None = None
    score: float = 0.0
    # Optional metadata carried through from the source doc for filtering/debugging
    drug_name: Optional[str] = None
    active_ingridients: Optional[List[str]] = None


class RetrievalAgent:
    """Agent for retrieving relevant passages from a FAISS index based on a query."""

    def __init__(self, index_dir: str = "data/index", min_similarity: float = 0.4, min_passages: int = 3, max_passages: int = 5) -> None:
        self.index_dir = index_dir
        self.docs_path = os.path.join(index_dir, "docs.jsonl")
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.min_similarity = min_similarity
        self.min_passages = min_passages
        self.max_passages = max_passages

        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(model_name)

        self._index: faiss.Index | None = None
        self._docs: Sequence[dict] | None = None

    def _ensure_index(self) -> Tuple[faiss.Index | None, Sequence[dict] | None]:
        """Load or create the FAISS index and associated documents."""
        if self._index is not None and self._docs is not None:
            return self._index, self._docs

        if not (os.path.exists(self.docs_path) and os.path.exists(self.index_path)):
            self._index, self._docs = None, None
            return self._index, self._docs

        self._index = faiss.read_index(self.index_path)
        with open(self.docs_path, "r", encoding="utf-8") as handle:
            self._docs = [json.loads(line) for line in handle]
        return self._index, self._docs

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def retrieve(self, query: str, top_k: int = 20, restrict_names: List[str] | None = None) -> List[Passage]:
        """Retrieve relevant passages for the given query."""
        index, docs = self._ensure_index()
        if index is None or docs is None:
            return []

        query_vector = self._embed([query])
        distances, indices = index.search(query_vector, top_k)

        idxs = indices[0].tolist()
        dists = distances[0].tolist()

        passages: List[Passage] = []
        
        # Safe per-passage character limit (env MAX_PASSAGE_CHARS, 0 disables)
        max_chars_env = int(os.getenv("MAX_PASSAGE_CHARS", "0"))
        max_chars = max(0, max_chars_env)

        for idx, score in zip(idxs, dists):
            if idx == -1:
                continue

            doc = docs[idx]
            
            # If restriction is provided, ensure the document matches the detected names
            if restrict_names:
                # Case-insensitive matching against drug_name and active_ingridients
                dn = str(doc.get("drug_name", "") or "").casefold()
                ingr_list = doc.get("active_ingridients") or []
                try:
                    ingr_list_cf = [str(x or "").casefold() for x in ingr_list]
                except Exception:
                    ingr_list_cf = []

                # Check if any of the restrict_names match
                match = False
                for name in restrict_names:
                    key = str(name or "").casefold()                    #
                    if not key:
                        continue
                    # Direct match against drug_name or any active ingredient
                    if key in dn:
                        match = True
                        break
                    # Check against active ingredients
                    if any(key in ing for ing in ingr_list_cf):
                        match = True
                        break
                if not match:
                    continue

            # Trim text to avoid excessively large prompts downstream
            full_text = doc.get("text", "") or ""
            text = full_text if max_chars == 0 else full_text[:max_chars]

            passages.append(
                Passage(
                    text=text,
                    url=to_web_url(doc.get("source_url")),
                    section=doc.get("section"),
                    score=float(score),
                    drug_name=doc.get("drug_name"),
                    active_ingridients=doc.get("active_ingridients"),
                )
            )

        passages.sort(key=lambda item: item.score, reverse=True)

        filtered = [p for p in passages if p.score >= self.min_similarity]
        if len(filtered) >= self.min_passages:
            return filtered[: self.max_passages]
        return passages[: self.min_passages]
