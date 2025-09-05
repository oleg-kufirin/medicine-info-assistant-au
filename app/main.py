import streamlit as st
from app import chains, safety, router, tools
from app.tools import CostInput

st.set_page_config(page_title="Patient Info Assistant (PBS/Medicare) — MVP", layout="wide")

st.title("Patient Information Assistant (PBS/Medicare) — MVP")
st.caption("General information only, not medical advice.")

with st.sidebar:
    st.header("Profile")
    patient_type = st.selectbox("Patient type", ["general","concession"])
    safety_net = st.slider("Safety Net progress", 0, 100, 0, help="Indicative only")
    st.header("Cost settings")
    cfg = tools.load_pricing_config()

query = st.text_input("Ask a question about a PBS-listed medicine")
go = st.button("Search")

if go and query:
    # Safety guard
    decision = safety.guard(query)
    if not decision.allow:
        st.warning(decision.message)

    intent = router.route(query)
    st.write(f"**Intent detected:** {intent}")

    if intent == "cost":
        # naive flag for under-copay (MVP assumes True)
        args = CostInput(patient_type=patient_type, safety_net_progress=safety_net/100.0, under_copay_flag=True)
        breakdown = tools.estimate_cost(args, cfg)
        st.subheader("Indicative Cost")
        st.json(breakdown)

    # Retrieve + synthesize
    passages = chains.retrieve(query)
    if not passages:
        st.error("No index found or no passages retrieved. Run `python scripts/ingest.py` first.")
    else:
        ans = chains.synthesize_answer(query, passages)
        st.subheader("Answer")
        st.write(ans["answer_text"])
        if ans.get("bullets"):
            st.markdown("**Details:**")
            for b in ans["bullets"]:
                st.markdown(f"- {b}")
        if ans.get("citations"):
            st.markdown("**Citations:**")
            for c in ans["citations"]:
                url = c.get("url","")
                sec = c.get("section","")
                if url:
                    st.markdown(f"- [{url}]({url}) {('— '+sec) if sec else ''}")
        st.caption(ans.get("disclaimer",""))
