import logging

from time import perf_counter
from typing import Any, Dict, List, TypedDict, Callable, Optional
from langgraph.graph import StateGraph, END
from utils import format_passages_for_context
from moderation_agent import ModerationAgent
from drug_detection_agent import DrugDetectionAgent
from reflection_agent import ReflectionAgent
from retrieval_agent import Passage, RetrievalAgent
from summary_writing_agent import SummaryWritingAgent

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
    # Preformatted passages context for prompts
    passages_context: str
    # Updated by SummaryWritingAgent
    summary_initial: str | None
    summary_draft: str | None
    # Updated by ReflectionAgent
    summary_critique: Dict[str, Any] | None
    # Updated by ResponseBuilder
    answer: Dict[str, Any]
    # Error info from SummaryWritingAgent
    summary_error: Dict[str, Any] | None


class AgentWorkflow:
    def __init__(self, on_event: Optional[Callable[[str, str, str | None], None]] = None, mode: str = "advanced") -> None:
        """
        Initialize the multi-agent workflow with moderation, retrieval, and summarization agents.
        """
        # Run mode: "advanced" runs reflection + revision; "light" skips them
        self.mode = mode
        self.on_event = on_event
        self.moderation = ModerationAgent() 
        self.drug_detector = DrugDetectionAgent()
        self.retriever = RetrievalAgent()
        self.summary_writer = SummaryWritingAgent()
        self.reflection_reviewer = ReflectionAgent()
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
        workflow.add_node("summary_revision", self._summary_revision_step)
        workflow.add_node("response_building", self._response_building_step)

        # Add edges
        workflow.set_entry_point("moderation")
        # After moderation, branch based on allow decision
        workflow.add_conditional_edges(
            "moderation",
            self._decide_after_moderation,
            {
                True: "drug_detection",
                False: END,
            },
        )
        # After drug detection, branch based on detected drugs
        workflow.add_conditional_edges(
            "drug_detection",
            self._decide_after_drug_detection,
            {
                True: "retrieval",
                False: "response_building",
            },
        )
        workflow.add_edge("retrieval", "summary_writing")
        # After summary, branch based on mode
        workflow.add_conditional_edges(
            "summary_writing",
            self._decide_after_summary_writing,
            {
                "advanced": "reflection",
                "light": "response_building",
            },
        )
        workflow.add_edge("reflection", "summary_revision")
        workflow.add_edge("summary_revision", "response_building")
        workflow.add_edge("response_building", END)
        
        return workflow.compile()


    def run(self, query: str) -> AgentState:
        """Run the compiled workflow with the given query."""
        start_time = perf_counter()
        try:
            logger.debug("Starting agent workflow \n", extra={"query": query})
            self._ui("workflow", "start", "Starting analysis…")
            initial_state: AgentState = {
                "query": query,
                "safety_intent_decision": {},
                "detected_drug_names": [],
                "passages": [],
                "passages_context": "",
                "summary_initial": None,
                "summary_draft": None,
                "summary_critique": None,
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
            print('-'*50)
            self._ui("workflow", "end", "Analysis complete")


    def _moderation_step(self, state: AgentState) -> AgentState:
        """Perform moderation on the query."""
        start_time = perf_counter()
        try:
            self._ui("moderation", "start", "👮🏻‍♂️ Moderation agent is working…Analyzing query for safety and intent")
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
            logger.info("Step latency - Moderation: %.2f s \n", elapsed_s)
            self._ui("moderation", "end")


    def _decide_after_moderation(self, state: AgentState) -> bool:
        """Decide whether to proceed based on moderation results."""
        decision = state.get("safety_intent_decision", {})
        return bool(decision.get("allow", False))


    def _drug_detection_step(self, state: AgentState) -> AgentState:
        """Perform drug detection on the query."""
        start_time = perf_counter()
        try:
            self._ui("drug_detection", "start", "💊 Drug detection agent is working…Identifying drug names in the query")
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
            logger.info("Step latency - Drug Detection: %.2f s \n", elapsed_s)
            self._ui("drug_detection", "end")


    def _decide_after_drug_detection(self, state: AgentState) -> bool:
        """Decide whether to proceed after drug detection."""
        detected_drug_names = state.get("detected_drug_names", [])
        return bool(detected_drug_names)


    def _retrieval_step(self, state: AgentState) -> AgentState:
        """Retrieve relevant passages based on the query."""
        start_time = perf_counter()
        try:
            self._ui("retrieval", "start", "🧲 Retrieval agent is working…Fetching relevant passages")
            query = state.get("query", "")
            detected_names = state.get("detected_drug_names", [])
            restrict = detected_names if isinstance(detected_names, list) and detected_names else None

            logger.debug(
                "[START] Running retrieval step",
                extra={
                    "query_preview": query[:80],
                    "restrict_names": ", ".join(restrict) if restrict else "",
                },
            )

            passages = self.retriever.retrieve(query, top_k=20, restrict_names=restrict)

            state["passages"] = passages
            # Pre-compute and store formatted passages context for prompts
            passages_context  = format_passages_for_context(passages)
            state["passages_context"] = passages_context

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
            logger.info("Step latency - Retrieval: %.2f s \n", elapsed_s)
            self._ui("retrieval", "end")


    def _summary_writing_step(self, state: AgentState) -> AgentState:
        """Summarize the retrieved passages and synthesize an answer."""
        start_time = perf_counter()
        try:
            self._ui("summary_writing", "start", "👩‍💻 Summary writing agent is working…Generating initial summary")
            query = state.get("query", "")
            passages_context = state.get("passages_context", "")

            logger.debug(
                "[START] Running summary writing step",
                extra={"query_preview": query[:80], "passages_context_length": len(passages_context)},
            )

            summary_text = self.summary_writer.write_summary(query, passages_context) if passages_context else None

            state["summary_initial"] = summary_text
            state["summary_draft"] = summary_text
            state["summary_critique"] = None
            # Propagate any summarization error details for UI display
            state["summary_error"] = self.summary_writer.get_last_error()

            logger.debug(
                "[END] Summary writing completed",
                extra={"draft_summary_available": bool(summary_text), "draft_summary_length": len(summary_text) if summary_text else 0},
            )

            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Summary Writing: %.2f s \n", elapsed_s)
            self._ui("summary_writing", "end")


    def _decide_after_summary_writing(self, state: AgentState) -> str:
        """Decide next step after summary based on configured mode."""
        return "advanced" if self.mode == "advanced" else "light"


    def _reflection_step(self, state: AgentState) -> AgentState:
        """Critique the summary draft and recommend follow-up actions."""
        start_time = perf_counter()
        try:
            self._ui("reflection_writing", "start", "🧑‍🏫 Reflection agent is working...Reviewing summary draft")
            query = state.get("query", "")
            passages_context = state.get("passages_context", "")
            summary_text = state.get("summary_draft")

            logger.debug(
                "[START] Running reflection review step",
                extra={"query_preview": query[:80], "has_summary": bool(summary_text), "passages_context_length": len(passages_context)},
            )

            
            critique = self.reflection_reviewer.review_summary(query, summary_text, passages_context)
            state["summary_critique"] = critique

            logger.debug(
                "[END] Reflection review completed",
                extra={
                    "critique_keys": list(critique.keys()) if isinstance(critique, dict) else [],
                },
            )
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Reflection: %.2f s \n" , elapsed_s)
            logger.info("Critique notes: \n %s \n", critique)
            self._ui("reflection_writing", "end")


    def _summary_revision_step(self, state: AgentState) -> AgentState:
        """Revise the summary using critique notes."""
        start_time = perf_counter()
        try:
            self._ui("summary_revision", "start", "👩‍💻 Summary writing agent is working...Revising summary based on review notes")
            query = state.get("query", "")
            passages_context = state.get("passages_context", "")
            draft = state.get("summary_draft")
            critique = state.get("summary_critique")

            logger.debug(
                "[START] Running summary revision step",
                extra={"query_preview": query[:80], "has_draft": bool(draft), "critique_present": bool(critique), "passages_context_length": len(passages_context)},
            )

            revised = self.summary_writer.rewrite_summary(query, draft, critique, passages_context)

            if revised:
                state["summary_draft"] = revised
                logger.debug("[END] Summary revision completed")
            else:
                logger.debug("Revision failed; retaining original draft summary")
            return state
        finally:
            elapsed_s = perf_counter() - start_time
            logger.info("Step latency - Summary Revision: %.2f s \n", elapsed_s)
            self._ui("summary_revision", "end")


    def _response_building_step(self, state: AgentState) -> AgentState:
        """Build the final answer response."""
        start_time = perf_counter()
        try:
            self._ui("response_building", "start", "Assembling final answer…")
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
            logger.info("Step latency - Response Builder: %.2f s \n", elapsed_s)
            self._ui("response_building", "end")


    def _ui(self, step: str, phase: str, label: str | None = None) -> None:
        """Safely emit UI events if a callback is provided.

        step: machine step name, e.g. 'retrieval'
        phase: 'start' | 'end'
        label: optional human-friendly label
        """
        cb = self.on_event
        if not cb:
            return
        try:
            cb(step, phase, label)
        except Exception:
            # Never let UI wiring break the agent
            logger.debug("UI callback failed for step=%s phase=%s", step, phase)
