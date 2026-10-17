# Interactive Campus Info Chatbot (Track A: Essential)

An AI chatbot that answers questions about college rules, clubs, events, locations, facilities and procedures from
handbooks, web pages, event calendars and contact directories. Built with LangChain, FAISS and Streamlit.

> `data/` contains **sample data**. Replace it with your college's real files before demoing.

## Features (mapped to the project PDF)
| Requirement | Where |
|---|---|
| Handbook / policy processing (PDF, DOCX, MD) | `campus/ingest.py` |
| Web scraping (BeautifulSoup + requests, respects robots.txt) | `campus/ingest.py` |
| Categorization (academics, facilities, clubs, events, placement...) | `categorize()` in `ingest.py` |
| FAISS retrieval, TF-IDF fallback with no API key | `campus/retriever.py` |
| Location info + Google Maps links, events, contacts | `campus/services.py` |
| LangChain tools + intent routing + grounded LLM answers | `campus/agent.py` |
| SQLite query history, feedback, analytics | `campus/store.py` |
| Chat, dashboard, filtered search, exports (CSV/.ics/MD), validation, error messages | `app.py` |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY; optional
streamlit run app.py
```
No key? It still runs in **offline mode** (keyword search, extractive answers). Add a key for conversational answers.

## Add your college's content
1. Put handbook PDFs/DOCX/MD in `data/`.
2. Edit `events.csv`, `contacts.csv`, `clubs.csv`, `locations.csv` (keep the column names).
3. Set `COLLEGE_NAME` and `COLLEGE_URLS` (public pages only) in `.env`.
4. Click **Refresh campus data** in the sidebar to re-index.

## Deploy on Streamlit Cloud
Push to GitHub, create an app from `app.py`, and paste your keys under **Settings > Secrets**:
```toml
GOOGLE_API_KEY = "..."
COLLEGE_NAME = "Your College"
COLLEGE_URLS = "https://your-college.edu.in/"
```
Never commit `.env` or `secrets.toml`.

## Architecture
```
PDF/DOCX/MD/CSV/Web -> ingest.py -> Documents (+category, source, updated)
                                      |-> retriever.py (chunk -> FAISS or TF-IDF)
question -> validate -> agent.py: route intent -> tools (location | events | contacts | handbook)
        -> context -> LLM (Gemini/OpenAI) or offline extractive answer -> answer + sources -> store.py
```


## Reference GitHub repos / examples
- https://github.com/toanuitt/College-AI- (Streamlit + LangChain + ChromaDB college chatbot)
- https://github.com/Marium369/AAIDC_Project-1--RAG-based-Chatbot (minimal LangChain + FAISS RAG)
- https://huggingface.co/spaces/UWF-CSDA-LLM/ARGOBot (student-handbook RAG with OpenAI/Gemini + LangGraph)
- https://github.com/topics/langchain-chatbot and https://github.com/topics/faiss-vector-database
