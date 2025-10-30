from pathlib import Path
import logging
import streamlit as st
import agent_runner
import summary_writing_agent as swa
import reflection_agent as ra

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
HERO_IMAGE = ASSETS_DIR / "hero-banner.jpg"
DRUG_NAMES_PATH = DATA_DIR / "rag_drug_names.txt"
SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.txt"

logging.basicConfig(level=logging.WARNING)
logging.getLogger(agent_runner.__name__).setLevel(logging.DEBUG)
logging.getLogger(swa.__name__).setLevel(logging.INFO)
logging.getLogger(ra.__name__).setLevel(logging.INFO)

st.set_page_config(
    page_title="Patient Information Assistant (CMI/PI)",
    layout="wide",
    page_icon=":pill:",
)

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #e0f2ff 0%, #f8fbff 100%) !important;
    color: #0f172a;
}
.hero-card {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 18px;
    padding: 24px 28px;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
    border: 1px solid #c7e3ff;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0b5394;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: #1e3a5f;
    margin-bottom: 1.0rem;
}
.pill-button button {
    border-radius: 999px !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 70%) !important;
    color: #ffffff !important;
    padding: 0.35rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    box-shadow: none !important;
    width: 100%;
}
.pill-button button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.75rem;
    font-weight: 600;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #60a5fa 100%);
}
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 1px solid #93c5fd;
}
.highlight-list ul {
    padding-left: 1.2rem;
}
.highlight-list li {
    margin-bottom: 0.4rem;
}

.hero-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 220px;
    border: 2px dashed #93c5fd;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.85);
    color: #1e3a5f;
    font-size: 1rem;
    text-align: center;
    padding: 1.25rem;
}

/* Segmented control: selected tab (simple + robust) */
div[data-testid="stSegmentedControl"] [role="tablist"] [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
    box-shadow: none !important;
    outline: none !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "pill_state" not in st.session_state:
    st.session_state.pill_state = {
        "drug_list": False,
        "sample_questions": False,
    }

@st.cache_data(show_spinner=False)
def load_lines(path: Path) -> list[str]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line.strip() for line in raw_text.splitlines() if line.strip()]

RAG_DRUG_NAMES = load_lines(DRUG_NAMES_PATH)
SAMPLE_QUESTIONS = load_lines(SAMPLE_QUESTIONS_PATH)

hero_container = st.container()
with hero_container:
    left, right = st.columns([3, 1.4])
    with left:
        pill_row = st.columns([1.1, 2, 4], gap="small")
        with pill_row[0]:
            st.markdown('<div class="pill-button">', unsafe_allow_html=True)
            if st.button("Drug list", key="drug_list_button"):
                # Toggle drug_list panel; when enabling, disable sample_questions
                current = bool(st.session_state.pill_state.get("drug_list", False))
                new_val = not current
                st.session_state.pill_state["drug_list"] = new_val
                if new_val:
                    st.session_state.pill_state["sample_questions"] = False
            st.markdown("</div>", unsafe_allow_html=True)
        with pill_row[1]:
            st.markdown('<div class="pill-button">', unsafe_allow_html=True)
            if st.button("Sample questions", key="sample_questions_button"):
                # Toggle sample_questions panel; when enabling, disable drug_list
                current = bool(st.session_state.pill_state.get("sample_questions", False))
                new_val = not current
                st.session_state.pill_state["sample_questions"] = new_val
                if new_val:
                    st.session_state.pill_state["drug_list"] = False
            st.markdown("</div>", unsafe_allow_html=True)
        # Spacer: pill_row[2]
        with pill_row[2]:
            # Align segmented control to the right within this column using subcolumns
            right_align = st.columns([1,1,1,3], gap="small")
            with right_align[3]:
                # About (left) + Mode segmented control (right)
                seg_col, about_col = st.columns([2, 1.1], gap="small", vertical_alignment="bottom")
                with seg_col:
                    st.segmented_control(
                        "Mode",
                        options=["Light", "Advanced"],
                        default="Advanced",
                        key="run_mode_segment",
                        help="Light mode skips review/rewrite for faster results",
                        # label_visibility="collapsed",
                    )                
                with about_col:
                    if hasattr(st, "popover"):
                        with st.popover("About"):
                            st.markdown("Developed by **Oleg Kufirin**")
                            st.caption("Data scienist & AI professional. Navigating the intersection of AI and medicine.")
                            row = st.columns([1, 1], gap="small")
                            with row[0]:
                                icon_col, btn_col = st.columns([0.25, 0.75], gap="small")
                                with icon_col:
                                    st.image(
                                        "https://cdn-icons-png.flaticon.com/512/174/174857.png",
                                        width=35,
                                    )
                                with btn_col:
                                    st.link_button(
                                        "LinkedIn",
                                        "https://www.linkedin.com/in/oleg-kufirin"
                                    )
                            with row[1]:
                                icon_col, btn_col = st.columns([0.25, 0.75], gap="small")
                                with icon_col:
                                    st.image(
                                        "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                                        width=40,
                                    )
                                with btn_col:
                                    st.link_button("GitHub", "https://github.com/oleg-kufirin")

        st.markdown(
            (
                "<div class='hero-card'>"
                "<div class='hero-title'>Patient Information Assistant (CMI/PI) with Agentic AI</div>"
                "<div class='hero-subtitle'>Find authoritative Consumer Medicine Information (CMI) and Product Information (PI) answers in seconds.<br/>Designed as a safety-first assistant &mdash; never a substitute for professional medical advice.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with right:
        if HERO_IMAGE.exists():
            st.image(
                str(HERO_IMAGE),
                width='stretch',
            )
        else:
            st.markdown(
                "<div class='hero-placeholder'>Upload a hero image to app/assets/hero-banner.jpg to customize this space.</div>",
                unsafe_allow_html=True,
            )

pill_state = st.session_state.get("pill_state", {})
if pill_state.get("drug_list"):
    if RAG_DRUG_NAMES:
        drug_list_display = ", ".join(RAG_DRUG_NAMES)
        st.info(f"Available diabetes drug entries (DPP-4 + SGLT-2 + GLP-1): {drug_list_display}")
    else:
        st.info(
            "No drugs configured yet. Update `app/data/rag_drug_names.txt` to keep this list in sync with your RAG."
        )
elif pill_state.get("sample_questions"):
    if SAMPLE_QUESTIONS:
        sample_questions_display = "\n".join(f"- {question}" for question in SAMPLE_QUESTIONS)
        st.info(f"Sample question prompts:\n{sample_questions_display}")
    else:
        st.info(
            "No sample questions configured yet. Update `app/data/sample_questions.txt` to keep this list current."
        )

with st.form(key="query_form"):
    st.write("### Ask about a medicine")
    query = st.text_input(
        "Phrase your question clearly to get accurate information from the CMI/PI documents.",
        key="query",
        placeholder="e.g. What precautions are listed for apixaban in the PI?",
    )
    st.caption("Note: Advanced mode may exceed 1 min due to LLM provider token limit (~6000 tokens/min). For faster results, use Light mode.")
    submitted = st.form_submit_button("Search CMI / PI")

# When user submits, run and persist result so it survives reruns
if submitted and query:
    # Live status + progress while the agent runs
    # Read segmented control selection; default to Advanced
    selected_mode = st.session_state.get("run_mode_segment", "Advanced")
    adv_mode = str(selected_mode).lower() == "advanced"
    steps = ["moderation", "drug_detection", "retrieval", "summary_writing"]
    if adv_mode:
        steps.append("reflection_writing")
        steps.append("summary_revision")
    steps.append("response_building")
    step_index = {name: i for i, name in enumerate(steps)}

    status = st.status("Preparing…", expanded=False)
    pbar = st.progress(0.0, text="Starting…")

    def on_event(step: str, phase: str, label: str | None = None) -> None:
        # Update status panel; only advance progress for known pipeline steps
        if phase == "start":
            status.update(label=label or f"Running {step.replace('_', ' ')}…", state="running")
            return
        # phase == 'end'
        if step not in step_index:
            return  # ignore non-pipeline events like 'workflow'
        idx = step_index[step]
        frac = min(1.0, (idx + 1) / max(1, len(steps)))
        pbar.progress(frac, text=f"Completed {step.replace('_', ' ')}")

    run_mode = "advanced" if adv_mode else "light"
    workflow = agent_runner.AgentWorkflow(on_event=on_event, mode=run_mode)

    # Run the workflow and handle exceptions
    try:
        with status:
            result = workflow.run(query)
        status.update(label="Complete", state="complete")
    except Exception:
        status.update(label="Failed", state="error")
        st.error("An unexpected error occurred while generating the answer. Please try again.")
        st.stop()
    finally:
        # Safety clear
        pbar.empty()

    st.session_state.last_query = query
    st.session_state.last_result = result

# Display from last persisted result (or the just-computed one)
result = st.session_state.get("last_result")
active_query = st.session_state.get("last_query") or ""

if result:
    # st.markdown("---")
    st.subheader("Your query")
    st.write(active_query)

    # Display safety and intent decision
    dec = result.get("safety_intent_decision") or {}
    safety_label = (dec.get("safety_label") or "unknown").replace("_", " ").title()
    intent_label = (dec.get("intent_label") or "unknown").replace("_", " ").title()
    safety_allow = dec.get("safety_allow")
    intent_allow = dec.get("intent_allow")

    # Interpret None as False for safety and intent allow
    safety_ok = False if safety_allow is None else bool(safety_allow)
    intent_ok = False if intent_allow is None else bool(intent_allow)

    # Final allow decision: if explicit, use it; else require both safety and intent to be OK
    allow = dec.get("allow")
    if allow is None:
        allow = safety_ok and intent_ok

    # Display decision summary
    if dec:
        safety_state = "OK" if safety_ok else "BLOCKED"
        intent_state = "OK" if intent_ok else "BLOCKED"
        st.info(
            f"Safety: {safety_state} - {safety_label.upper()} | Intent: {intent_state} - {intent_label.upper()}"
        )
        if not allow and dec.get("message"):
            st.warning(dec.get("message"))
            st.stop()

    # Display detected drug/ingredient names
    detected_names = result.get("detected_drug_names") or []
    st.markdown("### Detected Drugs")
    if detected_names:
        st.info(", ".join(detected_names).upper())
    else:
        st.info("No explicit drug or ingredient names detected.")

    # Show summarization errors (e.g., Groq 413 / token-limit) if any
    sum_err = result.get("summary_error") or {}
    if sum_err:
        if sum_err.get("kind") == "groq_api_error" and sum_err.get("status_code") == 413:
            st.error(
                "The model request was too large (token limit exceeded). This is due to Groq API limitations for free-tier users. \n\n"
                "Consider rephrasing your query to be more specific or narrowing down the scope. Contact support if the issue persists."
            )
            st.stop()
        else:
            st.warning("Summarization failed. The answer may be incomplete.")

    answer = result.get("answer") or {}

    with st.container():
        st.markdown("### Answer")

        summary = answer.get("summary_text")
        initial_summary = result.get("summary_initial")
        if summary or initial_summary:
            if summary and initial_summary and summary.strip() != initial_summary.strip():
                revised_tab, initial_tab = st.tabs(["Revised Summary", "Initial Summary"])
                with revised_tab:
                    st.markdown("**Revised Summary**")
                    st.write(summary)
                with initial_tab:
                    st.markdown("**Initial Draft Summary**")
                    st.write(initial_summary)
            elif summary:
                st.markdown("**Summary**")
                st.write(summary)
            elif initial_summary:
                st.markdown("**Initial Draft Summary**")
                st.write(initial_summary)

        bullets = answer.get("bullets", [])
        if bullets:
            items_html = []
            for b in bullets:
                if isinstance(b, dict):
                    text = b.get("text", "")
                    score = b.get("score")
                    drug_name = b.get("drug_name")
                    ingredients = b.get("active_ingridients")

                    # Build suffix details: similarity + optional drug and ingredients
                    parts = []
                    if isinstance(score, (int, float)):
                        parts.append(f"Similarity: {float(score):.3f}")
                    if drug_name:
                        parts.append(f"Drug: {drug_name}")
                    if ingredients:
                        if isinstance(ingredients, list):
                            ingredients_str = ", ".join(str(i) for i in ingredients if i)
                        else:
                            ingredients_str = str(ingredients)
                        if ingredients_str:
                            parts.append(f"Active ingredients: {ingredients_str}")
                    suffix = f" ({'; '.join(parts)})" if parts else ""

                    # Add inline style as a fallback to ensure grey color takes effect
                    suffix_html = (
                        f" <span class='muted-suffix' style='color:#6b7280'>{suffix}</span>"
                        if suffix
                        else ""
                    )

                    # Compose list item
                    items_html.append(f"<li>{text}<br>{suffix_html}</li>")
                else:
                    items_html.append(f"<li>{b}</li>")

            list_html = (
                "<div class='highlight-list'><strong>Highlights</strong><ol>"
                + "".join(items_html)
                + "</ol></div>"
            )
            st.markdown(list_html, unsafe_allow_html=True)

        citations = answer.get("citations", [])
        if citations:
            st.markdown("**Citations**")
            urls = set()
            for citation in citations:
                url = citation.get("url", "")
                if url and url not in urls:
                    urls.add(url)
                    st.link_button(url, url)

        disclaimer = answer.get("disclaimer") or "General information only, not medical advice."
        if disclaimer:
            st.caption(disclaimer)

