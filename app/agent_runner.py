import logging
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from moderation_agent import ModerationAgent
from retrieval_agent import RetrievalAgent, Passage
from summary_writing_agent import SummaryWritingAgent

import response_builder


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    safety_intent_decision: Dict[str, Any]
    passages: List[Passage] | List[Dict[str, Any]]
    summary_draft: str | None
    summary_critique: str | None
    answer: Dict[str, Any]


class AgentWorkflow:
    def __init__(self) -> None:
        """
        Initialize the multi-agent workflow with moderation, retrieval, and summarization agents.
        """
        self.moderation = ModerationAgent() 
        self.retriever = RetrievalAgent()
        self.summary_writer = SummaryWritingAgent()
        self.compiled_workflow = self.build_workflow()


    def build_workflow(self) -> Any:
        """Create and compile the multi-agent workflow."""
        logger.debug("Compiling agent workflow graph")
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("moderation", self._moderation_step)
        workflow.add_node("retrieval", self._retrieval_step)
        workflow.add_node("summary_writing", self._summary_writing_step)
        workflow.add_node("response_building", self._response_building_step)

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
        workflow.add_edge("retrieval", "summary_writing")
        workflow.add_edge("summary_writing", "response_building")
        workflow.add_edge("response_building", END)
        
        return workflow.compile()


    def _moderation_step(self, state: AgentState) -> AgentState:
        """Perform moderation on the query."""
        query = state.get("query", "")
        logger.debug("Running moderation step", extra={"query_preview": query[:80]})
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
        logger.debug(
            "Moderation decision completed",
            extra={
                "allow": allow,
                "safety_label": decision.safety_label,
                "intent_label": decision.intent_label,
            },
        )
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
        logger.debug("Running retrieval step", extra={"query_preview": query[:80]})
        passages = self.retriever.retrieve(query)
        state["passages"] = passages
        logger.debug(
            "Retrieval step completed",
            extra={
                "passage_count": len(passages),
                "top_score": passages[0].score if passages else None,
            },
        )
        return state


    def _summary_writing_step(self, state: AgentState) -> AgentState:
        """Summarize the retrieved passages and synthesize an answer."""
        query = state.get("query", "")
        passages = state.get("passages", [])
        logger.debug(
            "Running summary writing step",
            extra={"query_preview": query[:80], "passage_count": len(passages)},
        )
        summary_text = self.summary_writer.summarize_passages(query, passages) if passages else None
        self.summary_draft = summary_text if summary_text else None
        logger.debug(
            "Summary writing step completed",
            extra={"draft_summary_available": bool(summary_text), "draft_summary_length": len(summary_text) if summary_text else 0},
        )
        return state


    def _response_building_step(self, state: AgentState) -> AgentState:
        """Build the final answer response."""
        query = state.get("query", "")
        passages = state.get("passages", [])
        summary_text = self.summary_draft if hasattr(self, 'summary_draft') else None
        logger.debug(
            "Running response builder step",
            extra={"query_preview": query[:80], "passage_count": len(passages), "draft_summary_available": bool(summary_text)},
        )
        answer = response_builder.synthesize_answer(query, passages, summary_text=summary_text)
        state["answer"] = answer
        logger.debug(
            "Response builder step completed",
            extra={"answer_built": bool(answer)},
        )
        return state


    def run(self, query: str) -> AgentState:
        """Run the compiled workflow with the given query."""
        try:
            logger.debug("Starting agent workflow", extra={"query": query})
            initial_state: AgentState = {
                "query": query,
                "safety_intent_decision": {},
                "passages": [],
                "answer": {},
            }
            final_state = self.compiled_workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.exception("Agent workflow execution failed", extra={"query": query})
            raise
