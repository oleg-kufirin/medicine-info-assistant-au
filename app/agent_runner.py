from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from moderation_agent import ModerationAgent
from retrieval_agent import RetrievalAgent, Passage
from summarize_agent import SummarizationAgent

import response_builder

class AgentState(TypedDict, total=False):
    query: str
    safety_intent_decision: Dict[str, Any]
    passages: List[Passage] | List[Dict[str, Any]]
    answer: Dict[str, Any]

class AgentWorkflow:
    def __init__(self) -> None:
        """
        Initialize the multi-agent workflow with moderation, retrieval, and summarization agents.
        """
        self.moderation = ModerationAgent() 
        self.retriever = RetrievalAgent()
        self.summarizer = SummarizationAgent()
        self.compiled_workflow = self.build_workflow()


    def build_workflow(self) -> Any:
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("moderation", self._moderation_step)
        workflow.add_node("retrieval", self._retrieval_step)
        workflow.add_node("summarization", self._summarization_step)

        # Add edges
        workflow.set_entry_point("moderation")
        workflow.add_conditional_edges(
            "moderation",
            self._decide_after_moderation,
            {
                True: "retrieval",
                False: END,
            },
        )
        workflow.add_edge("retrieval", "summarization")
        workflow.add_edge("summarization", END)

        return workflow.compile()


    def _moderation_step(self, state: AgentState) -> AgentState:
        """Perform moderation on the query."""
        query = state.get("query", "")
        decision = self.moderation.classify_safety_and_intent(query)

        # Determine if the query is allowed based on moderation results
        allow = bool(decision.safety_allow and decision.intent_allow)
    
        state["safety_intent_decision"] = {
            "allow": allow,
            "safety_label": decision.safety_label,
            "safety_allow": decision.safety_allow,
            "intent_label": decision.intent_label,
            "intent_allow": decision.intent_allow,
            "message": decision.message,
        }
        return state


    def _decide_after_moderation(self, state: AgentState) -> bool:
        """Decide whether to proceed based on moderation results."""
        decision = state.get("safety_intent_decision", {})
        return bool(decision.get("allow", False))
    

    def _serialize_passages(self, passages: List[Any]) -> List[Dict[str, Any]]:
        """Serialize Passage objects into dictionaries."""
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

    def _retrieval_step(self, state: AgentState) -> AgentState:
        """Retrieve relevant passages based on the query."""
        query = state.get("query", "")
        passages = self.retriever.retrieve(query)
        state["passages"] = passages
        return state


    def _summarization_step(self, state: AgentState) -> AgentState:
        """Summarize the retrieved passages and synthesize an answer."""
        query = state.get("query", "")
        passages = state.get("passages", [])
        summary_text = self.summarizer.summarize_passages(query, passages) if passages else None
        answer = response_builder.synthesize_answer(query, passages, summary_text=summary_text)
        state["answer"] = answer
        return state
    

    def run(self, query: str) -> AgentState:
        """Run the compiled workflow with the given query."""
        try:
            print(f"[DEBUG] Starting full_pipeline with question='{query}'")
            initial_state: AgentState = {
                "query": query,
                "safety_intent_decision": {},
                "passages": [],
                "answer": {},
            }
            final_state = self.compiled_workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            print(f"[ERROR] Exception during workflow execution: {e}")
            raise
