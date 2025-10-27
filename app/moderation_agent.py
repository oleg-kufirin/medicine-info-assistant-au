import os

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from dotenv import load_dotenv
from utils import load_prompt

_LC_AVAILABLE = True
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
except Exception:  # pragma: no cover - dependency not available at runtime
    _LC_AVAILABLE = False

load_dotenv()

REFUSAL_EMPTY = (
    "The query is empty. Please ask a question about a medicine that can be answered using official Consumer "
    "Medicine Information or Product Information documents."
)

REFUSAL_ERROR = (
    "There was an error processing your request. Please try again later."
)

REFUSAL_UNSUPPORTED = (
    "I can only provide general information that appears in official Consumer Medicine Information or Product "
    "Information documents."
)

REFUSAL_EMERGENCY = (
    "I cannot help with emergencies. If you or someone else is at risk, contact local emergency services immediately."
)

REFUSAL_SELF_HARM = (
    "I am sorry you are feeling distressed. I cannot help with self-harm or suicide related requests. Please reach out "
    "to emergency services or a trusted professional right away."
)

REFUSAL_MEDICAL_ADVICE = (
    "I cannot give personalised medical advice or dosing guidance. Please speak with your pharmacist or doctor."
)


@dataclass
class SafetyIntentDecision:
    safety_label: Optional[str]
    safety_allow: bool
    intent_label: Optional[str]
    intent_allow: bool
    message: Optional[str] = None


class ModerationAgent:
    """
    Agent for moderating user queries. 
    If a query is deemed unsafe or inappropriate, the agent will refuse to answer.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._chain: Any = None


    def classify_safety_and_intent(self, query: str) -> SafetyIntentDecision:
        """Classify the safety and intent of the given query."""
        q = (query or "").strip()
        if not q:
            return SafetyIntentDecision("empty", False, "empty", False, REFUSAL_EMPTY)

        decision = self._classify_via_llm(q)
        if decision is None:
            return SafetyIntentDecision("error", False, "error", False, REFUSAL_ERROR)

        safety_label, safety_allow, intent_label, intent_allow = decision
        return SafetyIntentDecision(
            safety_label,
            safety_allow,
            intent_label,
            intent_allow,
            self._message_for_label(safety_label, intent_label),
        )


    def _get_llm(self):
        """Get the language model instance."""
        if self._llm is not None:
            return self._llm

        if not _LC_AVAILABLE:
            return None

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        model = os.getenv("SAFETY_MODEL", "llama-3.1-8b-instant")
        self._llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0)
        return self._llm


    def _get_classifier_chain(self):
        """Get or create the classification chain."""
        if self._chain is not None:
            return self._chain

        llm = self._get_llm()
        if llm is None:
            return None

        system_prompt = load_prompt("safety_moderation")
        if not system_prompt:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "Classify the following user query for safety and intent. Only output JSON. Query: {query}",
                ),
            ]
        )

        parser = JsonOutputParser()
        self._chain = prompt | llm | parser
        return self._chain


    def _classify_via_llm(self, query: str) -> Optional[Tuple[str, bool, str, bool]]:
        """Classify the query using the LLM chain."""
        try:
            chain = self._get_classifier_chain()
            obj = chain.invoke({"query": query}) if chain else None
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        safety_label = str(obj.get("safety_label", "other")).lower()
        intent_label = str(obj.get("intent_label", "other")).lower()

        valid_safety_labels = {"safe", "medical_advice", "emergency", "self_harm"}
        if safety_label not in valid_safety_labels:
            safety_label = "other"
        safety_allow = safety_label == "safe"

        valid_intent_labels = {"pi_cmi", "other"}
        if intent_label not in valid_intent_labels:
            intent_label = "other"
        intent_allow = intent_label == "pi_cmi"

        return safety_label, safety_allow, intent_label, intent_allow


    @staticmethod
    def _message_for_label(safety_label: str, intent_label: str) -> str:
        """Get a refusal message based on the safety and intent labels."""
        if safety_label == "emergency":
            return REFUSAL_EMERGENCY
        if safety_label == "self_harm":
            return REFUSAL_SELF_HARM
        if safety_label == "medical_advice":
            return REFUSAL_MEDICAL_ADVICE
        if intent_label == "other":
            return REFUSAL_UNSUPPORTED
        return "The query appears safe and appropriate."
