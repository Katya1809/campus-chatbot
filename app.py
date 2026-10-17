"""Interactive Campus Info Chatbot - Track A (Streamlit)."""
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud: copy secrets into env vars so the rest of the code just uses os.getenv
try:
    for _k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "COLLEGE_NAME", "COLLEGE_URLS",
               "GEMINI_MODEL", "OPENAI_MODEL", "EMBEDDINGS"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from campus.agent import CampusAgent  # noqa: E402
from campus.ingest import CATEGORIES, load_all  # noqa: E402
from campus.retriever import CampusRetriever  # noqa: E402
from campus.services import EventService, LocationService  # noqa: E402
from campus.store import Store  # noqa: E402

DATA = Path(__file__).parent / "data"
COLLEGE = os.getenv("COLLEGE_NAME", "Your College")
URLS = [u.strip() for u in os.getenv("COLLEGE_URLS", "").split(",") if u.strip()]
SAMPLES = ["Where is the placement cell?", "How to join the drama club?",
           "What is the attendance requirement?", "Upcoming events this month"]

st.set_page_config(page_title=f"{COLLEGE} Campus Assistant", page_icon="🎓", layout="wide")
st.markdown("""<style>
h1,h2,h3{letter-spacing:-.01em} .block-container{padding-top:2rem;max-width:1100px}
div[data-testid="stChatMessage"]{border-radius:12px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Reading handbooks, calendars and web pages...")
def load_system():
    docs, warnings = load_all(DATA, URLS)
    retr = CampusRetriever(docs)
    if retr.note:
        warnings.append(retr.note)
    return CampusAgent(retr, DATA, COLLEGE), warnings, datetime.now().strftime("%d %b %Y, %H:%M"), len(docs)


@st.cache_resource
def get_store():
    return Store(Path(__file__).parent / "campus.db")


def validate(q: str):
    q = re.sub(r"[\x00-\x1f]+", " ", (q or "")).strip()
    if len(q) < 2:
        return None, "Type a question with at least two characters."
    if len(q) > 500:
        return None, "Keep questions under 500 characters so I can search accurately."
    if time.time() - st.session_state.get("last_q", 0) < 1.0:
        return None, "One moment - you're sending questions too quickly."
    return q, ""


def ask(agent, store, q):
    q, err = validate(q)
    if err:
        st.warning(err)
        return None
    st.session_state.last_q = time.time()
    try:
        res = agent.answer(q, st.session_state.get("messages", []))
    except Exception as e:
        st.error(f"Something went wrong while searching ({type(e).__name__}). Please try again.")
        return None
    res["id"] = store.log(q, res["category"], res["answered"], ",".join(res["sources"]))
    res["query"] = q
    return res


def feedback_cb(mid):
    v = st.session_state.get(f"fb_{mid}")
    get_store().feedback(mid, v)


agent, load_warnings, loaded_at, n_docs = load_system()
store = get_store()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header(f"🎓 {COLLEGE}")
    st.caption(f"Mode: **{agent.mode}** · Search: **{agent.retriever.mode}**")
    st.caption(f"{n_docs} sources · last updated {loaded_at}")
    if st.button("Refresh campus data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    for w in load_warnings:
        st.warning(w)
    st.info("Answers come from published campus information and may be out of date. "
            "Please confirm important details (fees, deadlines) with the office. "
            "Don't share personal data here.")

st.title("Campus Assistant")
tab_chat, tab_dash, tab_search, tab_export, tab_stats = st.tabs(
    ["Chat", "Campus dashboard", "Search", "Export", "Analytics"])

# ---------------- Chat ----------------
with tab_chat:
    cols = st.columns(len(SAMPLES))
    for c, s in zip(cols, SAMPLES):
        if c.button(s, use_container_width=True):
            st.session_state.pending = s
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("sources"):
                    st.caption("Sources: " + ", ".join(m["sources"]))
                st.feedback("thumbs", key=f"fb_{m['id']}", on_change=feedback_cb, args=(m["id"],))
    prompt = st.chat_input("Ask about rules, clubs, events, locations, placements...")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        res = ask(agent, store, prompt)
        if res:
            st.session_state.messages.append({"role": "user", "content": res["query"]})
            st.session_state.messages.append({"role": "assistant", "content": res["answer"],
                                              "sources": res["sources"], "id": res["id"]})
            st.rerun()

# ---------------- Dashboard ----------------
with tab_dash:
    st.subheader("Quick access")
    cats = [c for c in CATEGORIES if c != "contacts"]
    cols = st.columns(5)
    for i, c in enumerate(cats):
        if cols[i % 5].button(c.title(), key=f"cat_{c}", use_container_width=True):
            st.session_state.dash = ask(agent, store, f"Tell me about {c} at the college")
    if st.session_state.get("dash"):
        st.markdown(st.session_state.dash["answer"])
    left, right = st.columns(2)
    with left:
        st.subheader("Upcoming events")
        up = agent.events.upcoming(limit=6)
        if up.empty:
            st.caption("No upcoming events in data/events.csv.")
        for _, r in up.iterrows():
            st.markdown(f"**{r['title']}** — {r['date']} {r['time']}, {r['venue']}  \n{r['description']}")
    with right:
        st.subheader("Contacts and locations")
        for _, r in agent.contacts.df.iterrows():
            with st.expander(f"{r['name']} — {r['role']}"):
                st.markdown(f"{r['department']} · {r['office']}  \n"
                            f"[{r['email']}](mailto:{r['email']}) · [{r['phone']}](tel:{str(r['phone']).replace(' ', '')})")
        for _, r in agent.locations.df.iterrows():
            st.markdown(f"[{r['name']}]({LocationService.maps_url(r, COLLEGE)}) — {r['building']}, {r['floor']}")

# ---------------- Search ----------------
with tab_search:
    q = st.text_input("Search campus information", max_chars=200)
    f1, f2 = st.columns(2)
    stype = f1.selectbox("Source type", ["All", "handbook", "web", "events", "contacts", "clubs", "facilities"])
    cat = f2.selectbox("Category", ["All"] + sorted(CATEGORIES) + ["general"])
    if q.strip():
        hits = agent.retriever.search(q, k=8, category=None if cat == "All" else cat,
                                      source_type=None if stype == "All" else stype)
        if not hits:
            st.info("No matches. Try different words or remove a filter.")
        for h in hits:
            m = h.metadata
            with st.expander(f"{m['source']} · {m['category']}"):
                st.write(h.page_content)
                st.caption(f"Type: {m['type']} · updated {m['updated']}")

# ---------------- Export ----------------
with tab_export:
    st.subheader("Download")
    st.download_button("Contact list (CSV)", agent.contacts.df.to_csv(index=False), "contacts.csv", "text/csv")
    ev = agent.events.upcoming(limit=50)
    st.download_button("Event calendar (.ics)", EventService.to_ics(ev), "campus_events.ics", "text/calendar")
    summary = "\n".join(f"- {c.title()}: {', '.join(CATEGORIES[c][:5])}" for c in CATEGORIES)
    st.download_button("Information summary (Markdown)",
                       f"# {COLLEGE} campus information\n\nTopics covered:\n{summary}\n\nLast updated: {loaded_at}\n",
                       "campus_summary.md", "text/markdown")

# ---------------- Analytics ----------------
with tab_stats:
    df = store.frame()
    if df.empty:
        st.info("No questions yet. Ask something in the Chat tab.")
    else:
        a, b, c = st.columns(3)
        a.metric("Questions", len(df))
        b.metric("Answered", f"{df['answered'].mean():.0%}")
        fb = df["feedback"].dropna()
        c.metric("Helpful votes", f"{fb.mean():.0%}" if len(fb) else "—")
        st.subheader("Popular topics")
        st.bar_chart(df["category"].value_counts())
        st.subheader("Information gaps (unanswered or downvoted)")
        gaps = df[(df["answered"] == 0) | (df["feedback"] == 0)]
        st.dataframe(gaps[["ts", "query", "category"]].tail(20), use_container_width=True, hide_index=True)
