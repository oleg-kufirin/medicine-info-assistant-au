# Patient Information Assistant (CMI/PI) - MVP

A safety-first retrieval assistant that answers questions using official Australian Consumer Medicine Information (CMI) and Product Information (PI) documents. The Streamlit UI routes every query through moderation, vector search, and LLM summarisation before showing a response with citations and a disclaimer.

## What it does
- Screens user questions with a Groq-hosted safety/intent classifier so emergency, self-harm, or off-topic queries are blocked early.
- Retrieves the most relevant passages from a FAISS index built from PBS pages and locally stored TGA CMI/PI PDFs.
- Optionally summarises the retrieved passages with an LLM prompt for quicker reading.
- Packages highlights, citations, and a standing "not medical advice" disclaimer for the UI.

## Requirements
- Python 3.10+
- A Groq API key (for moderation and summary chains)

Install Python dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration
Create a `.env` file (or update the existing one) with your credentials and model choices:

```
GROQ_API_KEY=...
SAFETY_MODEL=llama-3.1-8b-instant        # optional override
SUMMARY_MODEL=llama-3.1-8b-instant       # optional override
EMBED_MODEL=all-MiniLM-L6-v2             # optional override
```

## Build or refresh the search index
1. Edit `data/seed_urls.yaml` with authoritative PBS URLs you want to crawl.
2. Drop any CMI/PI PDFs you have into `data/downloads/` (optional, but improves coverage).
3. Generate the chunk corpus and FAISS index:
   ```bash
   python scripts/build_search_index.py
   ```
   This writes `data/index/docs.jsonl` and `data/index/faiss.index` if they do not already exist.

## Run the Streamlit app
```bash
streamlit run app/main.py
```
The UI supports optional branding by placing a `hero-banner.jpg` file in `app/assets/`.

## Moderation regression check
The moderation harness exercises a catalogue of safe, blocked, and off-policy prompts:
```bash
python tests/moderation_tests.py
```
Skipped cases usually mean the safety chain could not start (for example, missing `GROQ_API_KEY`).

## Project layout
```
patient-assistant/
  app/
    main.py                  # Streamlit UI
    agent_runner.py          # Orchestrates moderation -> retrieval -> answer
    moderation.py            # Safety + intent classifier via Groq
    retrieval.py             # FAISS-backed passage search
    summarize.py             # Optional LLM summarisation
    response_builder.py      # Formats answers for the UI
    utils.py
    prompts/
      system_safety_classifier.txt
      system_summary.txt
  data/
    downloads/               # Local CMI/PI PDFs
    index/                   # docs.jsonl + faiss.index
    seed_urls.yaml
  scripts/
    build_search_index.py    # Ingestion + indexing pipeline
    clean_pycache.py
  tests/
    moderation_tests.py
  requirements.txt
  README.md
```

## Disclaimer
The assistant supplies general CMI/PI information only and does not replace clinical judgement. Always verify critical details with official product literature or a licensed health professional.
