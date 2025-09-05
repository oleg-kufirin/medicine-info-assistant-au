# Patient Information Assistant (PBS/Medicare) — MVP

A minimal Retrieval-Augmented Generation (RAG) assistant for Australian PBS/Medicare medicine information.
- Answers simple questions like: “Is Drug X on the PBS?” and “Do I need authority?” with citations.
- Optional cost estimation for General vs Concession (indicative only; you must update pricing in `data/pricing_config.yaml`).
- **Not medical advice.**

## Stack
- Python, Streamlit (UI)
- LangChain (RAG plumbing)
- FAISS (vector store)
- Sentence Transformers (embeddings; default: `all-MiniLM-L6-v2`)
- Requests, BeautifulSoup4 (ingestion)
- PyPDF, pdfminer.six (PDF text extraction)

## Quickstart
1) **Install**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) **Seed URLs** (PBS/TGA CMIs/PI pages) — edit `data/seed_urls.yaml` with medicine pages you care about.
3) **Ingest**
```bash
python scripts/ingest.py
```
4) **Run app**
```bash
streamlit run app/main.py
```
5) **Try queries**
- “Is atorvastatin on the PBS?”
- “Do I need authority for semaglutide?”
- “What are common side effects of sertraline?”

## Disclaimer
This assistant provides general PBS/Medicare information **only**. It is **not medical advice**. For personal guidance, consult your pharmacist or GP.
Costs are **indicative**. Confirm current figures on official PBS sites.

## Structure
```
patient-assistant/
  app/
    main.py
    router.py
    chains.py
    tools.py
    safety.py
    prompts/
      system_retrieval.txt
      system_patient.txt
  data/
    seed_urls.yaml
    pricing_config.yaml
    docs_raw/
    index/
  eval/
    qa_seed.csv
    harness.py
  scripts/
    ingest.py
  requirements.txt
  README.md
```
