from typing import Any, Dict, List, TypedDict

import moderation, summarize, retrieval, response_builder

class AgentResponse(TypedDict, total=False):
    query: str
    safety_intent_decision: Dict[str, Any]
    passages: List[Dict[str, Any]]
    answer: Dict[str, Any]


def _serialize_passages(passages: List[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for p in passages:
        serialized.append(
            {
                "text": getattr(p, "text", ""),
                "url": getattr(p, "url", None),
                "section": getattr(p, "section", None),
                "score": float(getattr(p, "score", 0.0) or 0.0),
            }
        )
    return serialized


def run_graph(query: str, cfg: Dict[str, Any] | None = None, **_: Any) -> AgentResponse:
    """Run the simplified CMI/PI assistant pipeline for a single query."""
    decision = moderation.classify_safety_and_intent(query)
    allow = bool(decision.safety_allow and decision.intent_allow)

    response: AgentResponse = {
        "query": query,
        "safety_intent_decision": {
            "allow": allow,
            "safety_label": decision.safety_label,
            "safety_allow": decision.safety_allow,
            "intent_label": decision.intent_label,
            "intent_allow": decision.intent_allow,
            "message": decision.message,
        },
    }

    if not allow:
        response["final_message"] = decision.message or (
            "I can only provide information that appears in official Consumer Medicine Information "
            "or Product Information documents."
        )
        return response

    passages = retrieval.retrieve(query)

    serialized_passages = _serialize_passages(passages)
    if serialized_passages:
        response["passages"] = serialized_passages

    summary_text = summarize.summarize_passages(query, passages) if passages else None

    answer = response_builder.synthesize_answer(query, passages, summary_text=summary_text)
    response["answer"] = answer

    return response
