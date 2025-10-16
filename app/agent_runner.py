import logging

from time import perf_counter
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from moderation_agent import ModerationAgent
from drug_detection_agent import DrugDetectionAgent
from reflection_agent import ReflectionAgent
from retrieval_agent import Passage, RetrievalAgent
from summary_writing_agent import SummaryWritingAgent
# from wikipedia_tool import WikipediaLookupTool

import response_builder


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    # Updated by ModerationAgent
    safety_intent_decision: Dict[str, Any]
    # Updated by DrugDetectionAgent
    detected_drug_names: List[str]
    # Updated by RetrievalAgent
    passages: List[Passage] | List[Dict[str, Any]]
    # Updated by SummaryWritingAgent
    summary_initial: str | None
    summary_draft: str | None
    # Updated by ReflectionAgent
    summary_critique: Dict[str, Any] | None
    needs_external_context: bool
    wikipedia_queries: List[str]
    # Updated by WikipediaLookupTool
    wikipedia_context: List[Dict[str, Any]]
    # Updated by ResponseBuilder
    answer: Dict[str, Any]


class AgentWorkflow:
    def __init__(self) -> None:
        """
        Initialize the multi-agent workflow with moderation, retrieval, and summarization agents.
        """
        self.moderation = ModerationAgent() 
        self.drug_detector = DrugDetectionAgent()
        self.retriever = RetrievalAgent()
        self.summary_writer = SummaryWritingAgent()
        self.reflection_reviewer = ReflectionAgent()
        # self.wikipedia_tool = WikipediaLookupTool()
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self) -> Any:
        """Create and compile the multi-agent workflow."""
        logger.debug("Compiling agent workflow graph")
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("moderation", self._moderation_step)
        workflow.add_node("drug_detection", self._drug_detection_step)
        workflow.add_node("retrieval", self._retrieval_step)
        workflow.add_node("summary_writing", self._summary_writing_step)
        workflow.add_node("reflection", self._reflection_step)
        workflow.add_node("wikipedia_lookup", self._wikipedia_lookup_step)
        workflow.add_node("summary_revision", self._summary_revision_step)
        workflow.add_node("response_building", self._response_building_step)

        # Add edges
        workflow.set_entry_point("moderation")
        workflow.add_conditional_edges(
            "moderation",
            self._decide_after_moderation,
            {
                True: "drug_detection",
                False: END,
            },
        )
        workflow.add_conditional_edges(
            "drug_detection",
            self._decide_after_drug_detection,
            {
                True: "retrieval",
                False: "response_building",
            },
        )
        workflow.add_edge("retrieval", "summary_writing")
        workflow.add_edge("summary_writing", "reflection")
        workflow.add_conditional_edges(
            "reflection",
            self._decide_after_reflection,
            {
                "fetch_external": "wikipedia_lookup",
                "skip_external": "summary_revision",
            },
        )
        workflow.add_edge("wikipedia_lookup", "summary_revision")
        workflow.add_edge("summary_revision", "response_building")
        workflow.add_edge("response_building", END)
        
        return workflow.compile()


    def _moderation_step(self, state: AgentState) -> AgentState:
        """Perform moderation on the query."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            logger.debug("[START] Running moderation step", extra={"query_preview": query[:80]})
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
                "[END] Moderation completed",
                extra={
                    "allow": allow,
                    "safety_label": decision.safety_label,
                    "intent_label": decision.intent_label,
                },
            )
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Moderation: %.2f s", elapsed_s)


    def _decide_after_moderation(self, state: AgentState) -> bool:
        """Decide whether to proceed based on moderation results."""
        decision = state.get("safety_intent_decision", {})
        return bool(decision.get("allow", False))


    def _drug_detection_step(self, state: AgentState) -> AgentState:
        """Perform drug detection on the query."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            logger.debug("[START] Running drug detection step", extra={"query_preview": query[:80]})

            extracted_drug_names = self.drug_detector.extract_drug_names(query)

            validated_drug_names: List[str] = []
            for name in extracted_drug_names:
                if name.lower() in query.lower():
                    validated_drug_names.append(name)

            state["detected_drug_names"] = validated_drug_names

            logger.debug("[END] Drug detection completed", extra={"detected_drug_names": validated_drug_names,},)
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Drug Detection: %.2f s", elapsed_s)


    def _decide_after_drug_detection(self, state: AgentState) -> bool:
        """Decide whether to proceed after drug detection."""
        detected_drug_names = state.get("detected_drug_names", [])
        return bool(detected_drug_names)


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
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            detected_names = state.get("detected_drug_names") or []
            restrict = detected_names if isinstance(detected_names, list) and detected_names else None

            logger.debug(
                "[START] Running retrieval step",
                extra={
                    "query_preview": query[:80],
                    "restrict_names": ", ".join(restrict) if restrict else "",
                },
            )

            if restrict:
                passages = self.retriever.retrieve(query, top_k=20, restrict_names=restrict)
            else:
                passages = self.retriever.retrieve(query)

            state["passages"] = passages
            logger.debug(
                "[END] Retrieval completed",
                extra={
                    "passage_count": len(passages),
                    "top_score": passages[0].score if passages else None,
                },
            )
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Retrieval: %.2f s", elapsed_s)


    def _summary_writing_step(self, state: AgentState) -> AgentState:
        """Summarize the retrieved passages and synthesize an answer."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            passages = state.get("passages", [])

            logger.debug(
                "[START] Running summary writing step",
                extra={"query_preview": query[:80], "passage_count": len(passages)},
            )

            summary_text = self.summary_writer.summarize_passages(query, passages) if passages else None

            state["summary_initial"] = summary_text
            state["summary_draft"] = summary_text
            state["summary_critique"] = None
            state["needs_external_context"] = False
            state["wikipedia_queries"] = []
            state["wikipedia_context"] = []

            logger.debug(
                "[END] Summary writing completed",
                extra={"draft_summary_available": bool(summary_text), "draft_summary_length": len(summary_text) if summary_text else 0},
            )

            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Summary Writing: %.2f s", elapsed_s)


    def _reflection_step(self, state: AgentState) -> AgentState:
        """Critique the summary draft and recommend follow-up actions."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            passages = state.get("passages", [])
            summary_text = state.get("summary_draft")

            logger.debug(
                "[START] Running reflection review step",
                extra={
                    "query_preview": query[:80],
                    "has_summary": bool(summary_text),
                    "passage_count": len(passages),
                },
            )

            critique = self.reflection_reviewer.review_summary(query, passages, summary_text)
            needs_context = bool(critique.get("needs_additional_context"))
            wiki_queries = critique.get("wikipedia_queries") or []

            state["summary_critique"] = critique
            state["needs_external_context"] = needs_context and bool(wiki_queries)
            state["wikipedia_queries"] = wiki_queries

            logger.debug(
                "[END] Reflection review completed",
                extra={
                    "needs_external_context": state["needs_external_context"],
                    "wikipedia_query_count": len(wiki_queries),
                },
            )
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Reflection: %.2f s", elapsed_s)
            logger.info("Critique notes: \n %s", critique)
            logger.info("Needs additional context: %s", needs_context)
            if wiki_queries:
                logger.info("Wikipedia queries recommended: %s", wiki_queries)
            

    def _decide_after_reflection(self, state: AgentState) -> str:
        """Determine whether to perform an external Wikipedia lookup."""
        # needs_context = bool(state.get("needs_external_context"))
        # if needs_context:
        #     return "fetch_external"
        return "skip_external"


    def _wikipedia_lookup_step(self, state: AgentState) -> AgentState:
        """Fetch supplemental context from Wikipedia when required."""
        queries = state.get("wikipedia_queries", []) or []
        logger.debug(
            "[START] Running Wikipedia lookup step",
            extra={"query_count": len(queries)},
        )
        results = self.wikipedia_tool.batch_lookup(queries)
        state["wikipedia_context"] = [
            {
                "query": result.query,
                "title": result.title,
                "summary": result.summary,
                "url": result.url,
            }
            for result in results
        ]
        logger.debug(
            "[END] Wikipedia lookup completed",
            extra={"results_found": len(state["wikipedia_context"])},
        )
        return state


    def _summary_revision_step(self, state: AgentState) -> AgentState:
        """Revise the summary using critique notes and external context if available."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            passages = state.get("passages", [])
            draft = state.get("summary_draft")
            critique = state.get("summary_critique")
            external_context = state.get("wikipedia_context", [])
            logger.debug(
                "[START] Running summary revision step",
                extra={
                    "query_preview": query[:80],
                    "has_draft": bool(draft),
                    "critique_present": bool(critique),
                    "external_context_count": len(external_context),
                },
            )
            revised = self.summary_writer.rewrite_summary(
                query,
                passages,
                draft,
                critique,
                external_context,
            )
            if revised:
                state["summary_draft"] = revised
                logger.debug("[END] Summary revision completed")
            else:
                logger.debug("Revision failed; retaining original draft summary")
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Summary Revision: %.2f s", elapsed_s)


    def _response_building_step(self, state: AgentState) -> AgentState:
        """Build the final answer response."""
        start_time = perf_counter()
        try:
            query = state.get("query", "")
            passages = state.get("passages", [])
            summary_text = state.get("summary_draft")
            logger.debug(
                "[START] Running response builder step",
                extra={"query_preview": query[:80], "passage_count": len(passages), "draft_summary_available": bool(summary_text)},
            )
            answer = response_builder.synthesize_answer(query, passages, summary_text=summary_text)
            state["answer"] = answer
            logger.debug(
                "[END] Response building completed",
                extra={"answer_built": bool(answer)},
            )
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Response Builder: %.2f s", elapsed_s)


    def run(self, query: str) -> AgentState:
        """Run the compiled workflow with the given query."""
        start_time = perf_counter()
        try:
            logger.debug("Starting agent workflow", extra={"query": query})
            initial_state: AgentState = {
                "query": query,
                "safety_intent_decision": {},
                "passages": [],
                "summary_initial": None,
                "summary_draft": None,
                "summary_critique": None,
                "needs_external_context": False,
                "wikipedia_queries": [],
                "wikipedia_context": [],
                "answer": {},
            }
            final_state = self.compiled_workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.exception("Agent workflow execution failed", extra={"query": query})
            raise
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Total inference latency: %.2f s", elapsed_s)
