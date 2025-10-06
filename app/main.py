from pathlib import Path

import streamlit as st
import agent_runner

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
HERO_IMAGE = ASSETS_DIR / "hero-banner.jpg"

st.set_page_config(
    page_title="Patient Information Assistant (CMI/PI) - MVP",
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
    padding: 36px 40px;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
    border: 1px solid #c7e3ff;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #0b5394;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #1e3a5f;
    margin-bottom: 1.5rem;
}
.blue-pill {
    display: inline-block;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 70%);
    color: #ffffff;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
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
.result-card {
    background: rgba(255, 255, 255, 0.94);
    border-left: 4px solid #2563eb;
    border-radius: 16px;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.1);
    padding: 28px 32px;
    margin-top: 1.5rem;
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
    min-height: 280px;
    border: 2px dashed #93c5fd;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.85);
    color: #1e3a5f;
    font-size: 1rem;
    text-align: center;
    padding: 2rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

hero_container = st.container()
with hero_container:
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<span class="blue-pill">CMI / PI SEARCH</span>', unsafe_allow_html=True)
        st.markdown(
            (
                "<div class='hero-card'>"
                "<div class='hero-title'>Patient Information Assistant (CMI/PI)</div>"
                "<div class='hero-subtitle'>Find authoritative Consumer Medicine Information (CMI) and Product Information (PI) answers in seconds. Designed as a safety-first assistant &mdash; never a substitute for professional medical advice.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with right:
        if HERO_IMAGE.exists():
            st.image(
                str(HERO_IMAGE),
                caption="Your uploaded hero image",
                use_column_width=True,
            )
        else:
            st.markdown(
                "<div class='hero-placeholder'>Upload a hero image to app/assets/hero-banner.jpg to customize this space.</div>",
                unsafe_allow_html=True,
            )

with st.form(key="query_form"):
    st.write("### Ask about a medicine")
    query = st.text_input(
        "Phrase your question using information found in CMI or PI documents",
        key="query",
        placeholder="e.g. What precautions are listed for apixaban in the PI?",
    )
    submitted = st.form_submit_button("Search CMI / PI")

if submitted and query:
    st.markdown("---")
    st.subheader("Your query")
    st.write(query)

    result = agent_runner.run_graph(query)

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
            f"Safety: {safety_state} - {safety_label} | Intent: {intent_state} - {intent_label}"
        )
        if not allow and dec.get("message"):
            st.warning(dec.get("message"))
            st.stop()

    answer = result.get("answer") or {}

    with st.container():
        st.markdown("### Answer")

        summary = answer.get("summary_text")
        if summary:
            st.markdown("**Summary**")
            st.write(summary)

        bullets = answer.get("bullets", [])
        if bullets:
            st.markdown("<div class='highlight-list'><strong>Highlights</strong><ul>", unsafe_allow_html=True)
            for b in bullets:
                if isinstance(b, dict):
                    text = b.get("text", "")
                    score = b.get("score")
                    score_suffix = ""
                    if isinstance(score, (int, float)):
                        score_suffix = f" (similarity: {score:.3f})"
                    st.markdown(f"<li>{text}<br>{score_suffix}</li>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<li>{b}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

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
