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
| **Option A2: Comprehensive Club & Society Database (39 clubs)** | `data/clubs.csv` & `campus/services.py` |
| **Option A2: Event Discovery & Step-by-Step Registration (28 events)** | `data/events.csv` & `campus/services.py` |
| **Option A2: Multi-Factor Activity Recommender & 30-Day Plan** | `campus/recommender.py` |
| **Option A2: Campus Engagement Hub (4-Year Roadmap, GC, EAA)** | `campus/engagement.py` |
| LangChain tools + intent routing + grounded LLM answers | `campus/agent.py` & `campus/tools.py` |
| SQLite query history, feedback, analytics | `campus/store.py` |
| Interactive Web App (9 tabs, Matchmaker, Calendar Export, Maps) | `app.py` |

## Run locally

The web application is active and accessible at:
- **Local Browser Link**: **[http://localhost:8501](http://localhost:8501)**

To start or restart manually:
```bash
# 1. Activate environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
# 2. Install dependencies
pip install -r requirements.txt
# 3. Launch the Streamlit application
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

## Tips for the 8-week plan
Weeks 1-2: ingestion + chat + deploy. Weeks 3-4: add your real calendars/contacts/maps. Weeks 5-6: pick Option A1
(Student Services) or A2 (Clubs): extend `handbook.md` / `clubs.csv` and the prompt in `agent.py`. Weeks 7-8: polish,
README, demo video. Ideas: regional-language replies, a campus-term glossary, scheduled re-scraping.

## Reference GitHub repos / examples
- https://github.com/toanuitt/College-AI- (Streamlit + LangChain + ChromaDB college chatbot)
- https://github.com/Marium369/AAIDC_Project-1--RAG-based-Chatbot (minimal LangChain + FAISS RAG)
- https://huggingface.co/spaces/UWF-CSDA-LLM/ARGOBot (student-handbook RAG with OpenAI/Gemini + LangGraph)
- https://github.com/topics/langchain-chatbot and https://github.com/topics/faiss-vector-database
