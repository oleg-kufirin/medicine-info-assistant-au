from __future__ import annotations

import os
from typing import Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from utils import load_prompt

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class DrugDetectionResult(BaseModel):
    names: List[str] = Field(default_factory=list, 
                             description="List of drug or active-ingredient names explicitly present in the query."
    )


class DrugDetectionAgent:
    """Agent that extracts drug/ingredient names from a user query using an LLM with structured output.

    Public methods:
    - extract_drug_names(query: str) -> List[str]
    - extract_drug_name(query: str) -> str  (compat: first name or empty string)
    """

    def __init__(self) -> None:
        self._chain: Any = None

    def _get_chain(self):
        """Create (or reuse) the extraction chain that returns structured output."""
        if self._chain is not None:
            return self._chain

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model = os.getenv("DETECTION_MODEL", os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant"))
        llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0)
        structured_llm = llm.with_structured_output(DrugDetectionResult)

        system_prompt = load_prompt("system_drug_detection")
        if not system_prompt:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "Query: {query}"),
            ]
        )

        self._chain = prompt | structured_llm
        return self._chain

    @staticmethod
    def _clean_names(items: List[str]) -> List[str]:
        """Clean and deduplicate extracted names. Returns [] on failure or no valid names."""
        cleaned: List[str] = []
        seen = set()
        # Simple cleaning: trim whitespace and common surrounding punctuation
        for raw in items or []:
            if not isinstance(raw, str):
                continue
            name = raw.strip().strip("\"'`").strip()
            name = name.strip(" .:;,-")

            if not name or len(name) > 80:
                continue
            key = name.casefold()
            
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(name)
        return cleaned

    def extract_drug_names(self, query: str) -> List[str]:
        """Extract a list of explicit drug/ingredient names from the query. Returns [] on failure."""
        q = (query or "").strip()
        if not q:
            return []

        chain = self._get_chain()
        if chain is None:
            return []

        try:
            result: DrugDetectionResult = chain.invoke({"query": q})
        except Exception:
            return []

        names = list(result.names or [])
        return self._clean_names(names)

    # Backwards-compatible helper: return the first name or empty string
    def extract_drug_name(self, query: str) -> str:
        """Extract the first explicit drug/ingredient name from the query, or return an empty string."""
        names = self.extract_drug_names(query)
        return names[0] if names else ""
