"""Interactive Campus Info Chatbot - Streamlit UI.

Tabs:
  1. Chat              — conversational Q&A with history and source citations
  2. Campus Dashboard  — upcoming events, contacts, club directory
  3. Campus Map        — interactive location map with category filters
  4. Search            — filtered full-text search over all sources
  5. Export            — CSV / ICS / Markdown downloads
  6. Analytics         — query stats, topic breakdown, info gaps
"""
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud: copy secrets into env vars
try:
    for _k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "COLLEGE_NAME",
               "COLLEGE_URLS", "GEMINI_MODEL", "OPENAI_MODEL", "EMBEDDINGS"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from campus.agent import CampusAgent  # noqa: E402
from campus.ingest import CATEGORIES, load_all  # noqa: E402
from campus.retriever import CampusRetriever  # noqa: E402
from campus.services import ClubService, EventService, LocationService  # noqa: E402
from campus.store import Store  # noqa: E402

DATA = Path(__file__).parent / "data"
COLLEGE = os.getenv("COLLEGE_NAME", "Your College")
URLS = [u.strip() for u in os.getenv("COLLEGE_URLS", "").split(",") if u.strip()]

SAMPLES = [
    "Where is the placement cell?",
    "How to join the robotics club?",
    "What is the attendance policy?",
    "Upcoming events this month",
    "Emergency contacts",
    "What clubs can I join?",
]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{COLLEGE} Campus Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { letter-spacing: -.01em; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
div[data-testid="stChatMessage"] { border-radius: 12px; }

/* Quick-action buttons */
div.stButton > button {
    border-radius: 8px;
    font-size: 0.82rem;
    padding: 0.35rem 0.6rem;
    transition: all .15s ease;
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,.12); }

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 10px;
    padding: .75rem 1rem;
    border: 1px solid #e2e8f0;
}

/* Event cards */
.event-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: .75rem 1rem;
    margin-bottom: .5rem;
}
.event-card.cultural { background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%); border-left-color: #a855f7; }
.event-card.sports    { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-left-color: #22c55e; }
.event-card.technical { background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border-left-color: #f97316; }
.event-card.social    { background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%); border-left-color: #eab308; }

/* Location pills */
.location-pill {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 20px;
    padding: .2rem .6rem;
    font-size: .78rem;
    margin: .15rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Cache ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Reading handbooks, calendars and web pages…")
def load_system():
    docs, warnings = load_all(DATA, URLS)
    retr = CampusRetriever(docs)
    if retr.note:
        warnings.append(retr.note)
    return (
        CampusAgent(retr, DATA, COLLEGE),
        warnings,
        datetime.now().strftime("%d %b %Y, %H:%M"),
        len(docs),
    )


@st.cache_resource
def get_store():
    return Store(Path(__file__).parent / "campus.db")


# ── Validation & ask ─────────────────────────────────────────────────────────

def validate(q: str):
    q = re.sub(r"[\x00-\x1f]+", " ", (q or "")).strip()
    if len(q) < 2:
        return None, "Type a question with at least two characters."
    if len(q) > 500:
        return None, "Keep questions under 500 characters."
    if time.time() - st.session_state.get("last_q", 0) < 1.0:
        return None, "One moment — you're sending questions too quickly."
    return q, ""


def ask(agent, store, q: str):
    q, err = validate(q)
    if err:
        st.warning(err)
        return None
    st.session_state.last_q = time.time()
    try:
        res = agent.answer(q, st.session_state.get("messages", []))
    except Exception as e:
        st.error(f"Something went wrong ({type(e).__name__}). Please try again.")
        return None
    res["id"] = store.log(q, res["category"], res["answered"], ",".join(res["sources"]))
    res["query"] = q
    return res


def feedback_cb(mid):
    v = st.session_state.get(f"fb_{mid}")
    get_store().feedback(mid, v)


# ── Event colour helper ───────────────────────────────────────────────────────

_CAT_CLASSES = {
    "cultural": "cultural", "sports": "sports", "technical": "technical", "social": "social",
}


def _event_card(r: pd.Series):
    cat = str(r.get("category", "")).lower()
    css = _CAT_CLASSES.get(cat, "")
    d = pd.to_datetime(r["date"], errors="coerce")
    from datetime import date as _date
    days = (d.date() - _date.today()).days if not pd.isna(d) else None
    badge = f" <span style='font-size:.75rem;color:#666'>(in {days} days)</span>" if days and days >= 0 else ""
    reg = str(r.get("registration", "")) or "Walk-in"
    st.markdown(
        f"<div class='event-card {css}'>"
        f"<strong>{r['title']}</strong>{badge}<br>"
        f"📅 {r['date']} &nbsp; 🕐 {r['time']} &nbsp; 📍 {r['venue']}<br>"
        f"<em>{r.get('club','')}</em> — {r.get('description','')}<br>"
        f"<small>Registration: {reg}</small>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Load system ───────────────────────────────────────────────────────────────

agent, load_warnings, loaded_at, n_docs = load_system()
store = get_store()
st.session_state.setdefault("messages", [])

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(f"🎓 {COLLEGE}")
    st.caption(f"Mode: **{agent.mode}** · Search: **{agent.retriever.mode}**")
    st.caption(f"{n_docs} document chunks · last updated {loaded_at}")

    if st.button("🔄 Refresh campus data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Quick stats")
    col1, col2 = st.columns(2)
    col1.metric("Locations", len(agent.locations.df))
    col2.metric("Events", len(agent.events.df))
    col1.metric("Contacts", len(agent.contacts.df))
    col2.metric("Clubs", len(agent.clubs.df))

    st.divider()
    for w in load_warnings:
        st.warning(w)
    st.info(
        "Answers come from published campus information and may be out of date. "
        "Confirm important details (fees, deadlines) with the relevant office. "
        "Don't share personal data here."
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_dash, tab_map, tab_search, tab_export, tab_stats = st.tabs(
    ["💬 Chat", "📋 Dashboard", "🗺️ Campus Map", "🔍 Search", "📥 Export", "📊 Analytics"]
)

# ════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask me anything about campus")

    # Quick-action sample buttons
    cols = st.columns(len(SAMPLES))
    for c, s in zip(cols, SAMPLES):
        if c.button(s, use_container_width=True, key=f"sample_{s[:20]}"):
            st.session_state.pending = s

    st.divider()

    # Message history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("sources"):
                    st.caption("📚 Sources: " + ", ".join(m["sources"]))
                if m.get("tools"):
                    st.caption("🔧 Tools used: " + ", ".join(m["tools"]))
                st.feedback("thumbs", key=f"fb_{m['id']}", on_change=feedback_cb, args=(m["id"],))

    prompt = st.chat_input("Ask about rules, clubs, events, locations, placements…")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching campus information…"):
                res = ask(agent, store, prompt)
        if res:
            st.session_state.messages.append({"role": "user", "content": res["query"]})
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res["sources"],
                "tools": res["tools"],
                "id": res["id"],
            })
            st.rerun()

# ════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ════════════════════════════════════════════════
with tab_dash:
    # Top category buttons
    st.subheader("Browse by topic")
    cats = [c for c in CATEGORIES if c != "contacts"]
    btn_cols = st.columns(min(len(cats), 5))
    for i, c in enumerate(cats):
        if btn_cols[i % 5].button(c.title(), key=f"cat_{c}", use_container_width=True):
            st.session_state.dash = ask(agent, store, f"Tell me about {c} at the college")
    if st.session_state.get("dash"):
        with st.expander("📖 Topic answer", expanded=True):
            st.markdown(st.session_state.dash["answer"])

    st.divider()

    left, right = st.columns(2)

    # Events column
    with left:
        st.subheader("📅 Upcoming Events")
        up = agent.events.upcoming(limit=8)
        if up.empty:
            st.caption("No upcoming events in data/events.csv.")
        else:
            # Category filter
            cats_ev = ["All"] + sorted(up["category"].dropna().unique().tolist())
            sel = st.selectbox("Filter by category", cats_ev, key="dash_ev_cat")
            filtered = up if sel == "All" else up[up["category"] == sel]
            for _, r in filtered.iterrows():
                _event_card(r)

    # Contacts + Clubs column
    with right:
        st.subheader("👥 Contact Directory")
        dept_filter = st.text_input("Search contacts", placeholder="e.g. hostel, placement, health", key="dash_contact_search")
        if dept_filter:
            contact_rows = agent.contacts.find(dept_filter, n=8)
        else:
            contact_rows = agent.contacts.df.head(10)
        for _, r in contact_rows.iterrows():
            with st.expander(f"**{r['name']}** — {r['role']}"):
                st.markdown(
                    f"🏢 {r['department']} · {r['office']}  \n"
                    f"📧 [{r['email']}](mailto:{r['email']})  \n"
                    f"📞 [{r['phone']}](tel:{str(r['phone']).replace(' ','')})"
                )

        st.divider()
        st.subheader("🎭 Clubs & Societies")
        club_type_filter = st.selectbox(
            "Filter by type",
            ["All", "technical", "cultural", "social", "sports", "residents"],
            key="dash_club_type",
        )
        if club_type_filter == "All":
            club_rows = agent.clubs.df
        else:
            club_rows = agent.clubs.by_type(club_type_filter)
        for _, r in club_rows.iterrows():
            with st.expander(f"**{r['name']}** [{r.get('type','').upper()}]"):
                st.markdown(
                    f"{r.get('description','')}  \n"
                    f"**How to join**: {r.get('how_to_join','')}  \n"
                    f"**Meets**: {r.get('meets','')}  \n"
                    f"📧 {r.get('contact','')}"
                )

# ════════════════════════════════════════════════
# TAB 3: CAMPUS MAP
# ════════════════════════════════════════════════
with tab_map:
    st.subheader("🗺️ Campus Map & Locations")

    loc_df = agent.locations.all_with_coords()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        if not loc_df.empty:
            # Category colour filter
            all_cats = sorted(loc_df["category"].dropna().unique().tolist())
            selected_cats = st.multiselect(
                "Filter by category",
                all_cats,
                default=all_cats,
                key="map_cat_filter",
            )
            map_df = loc_df[loc_df["category"].isin(selected_cats)] if selected_cats else loc_df

            # st.map for interactive Pydeck map
            if not map_df.empty:
                st.map(
                    map_df[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"}),
                    zoom=15,
                )
            else:
                st.info("No locations match the selected categories.")
        else:
            st.info("No coordinate data available for map display.")

    with col_right:
        st.subheader("📍 Location Directory")
        search_loc = st.text_input("Search location", placeholder="e.g. library, hostel, canteen", key="map_search")
        if search_loc:
            show_df = agent.locations.find(search_loc, n=10)
        else:
            show_df = agent.locations.df

        for _, r in show_df.iterrows():
            maps_link = LocationService.maps_url(r, COLLEGE)
            cat = str(r.get("category", ""))
            timings = str(r.get("timings", "")) or "n/a"
            with st.expander(f"**{r['name']}**"):
                st.markdown(
                    f"🏢 {r.get('building','')} · {r.get('floor','')}  \n"
                    f"📂 Category: `{cat}`  \n"
                    f"⏰ {timings}  \n"
                    f"ℹ️ {r.get('description','')}  \n"
                    f"[📍 Open in Google Maps]({maps_link})"
                )

    # Category summary pills
    st.divider()
    st.subheader("Locations by Category")
    cat_summary = agent.locations.category_summary()
    for cat, names in sorted(cat_summary.items()):
        pills = "".join(f"<span class='location-pill'>{n}</span>" for n in names)
        st.markdown(f"**{cat.title()}**: {pills}", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 4: SEARCH
# ════════════════════════════════════════════════
with tab_search:
    st.subheader("🔍 Search Campus Information")
    q = st.text_input("Search query", max_chars=200, placeholder="e.g. fee structure, yoga exam, placement drive")
    f1, f2 = st.columns(2)
    stype = f1.selectbox(
        "Source type",
        ["All", "handbook", "web", "events", "contacts", "clubs", "facilities", "ics"],
        key="search_stype",
    )
    cat = f2.selectbox(
        "Category",
        ["All"] + sorted(CATEGORIES) + ["general"],
        key="search_cat",
    )
    if q.strip():
        hits = agent.retriever.search(
            q, k=10,
            category=None if cat == "All" else cat,
            source_type=None if stype == "All" else stype,
        )
        if not hits:
            st.info("No matches found. Try different keywords or remove a filter.")
        else:
            st.success(f"Found {len(hits)} relevant passages")
        for h in hits:
            m = h.metadata
            with st.expander(f"**{m['source']}** · {m['category']} · {m['type']}"):
                st.write(h.page_content)
                st.caption(f"Type: {m['type']} · Updated: {m['updated']}")

# ════════════════════════════════════════════════
# TAB 5: EXPORT
# ════════════════════════════════════════════════
with tab_export:
    st.subheader("📥 Download Campus Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Contacts**")
        st.download_button(
            "📋 Contact list (CSV)",
            agent.contacts.df.to_csv(index=False),
            "contacts.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("**Events Calendar**")
        ev_all = agent.events.upcoming(limit=50)
        st.download_button(
            "📅 Upcoming events (ICS)",
            EventService.to_ics(ev_all),
            "campus_events.ics",
            "text/calendar",
            use_container_width=True,
        )
        st.download_button(
            "📅 Upcoming events (CSV)",
            ev_all.drop(columns=["_d"], errors="ignore").to_csv(index=False),
            "campus_events.csv",
            "text/csv",
            use_container_width=True,
        )

    with col3:
        st.markdown("**Campus Summary**")
        summary_lines = [f"# {COLLEGE} Campus Information\n", f"Last updated: {loaded_at}\n"]
        summary_lines.append(f"\n## Locations ({len(agent.locations.df)} entries)\n")
        for _, r in agent.locations.df.iterrows():
            summary_lines.append(f"- **{r['name']}**: {r.get('building','')} — {r.get('description','')[:80]}\n")
        summary_lines.append(f"\n## Clubs ({len(agent.clubs.df)} entries)\n")
        for _, r in agent.clubs.df.iterrows():
            summary_lines.append(f"- **{r['name']}** ({r.get('type','')}) — {r.get('description','')[:80]}\n")
        summary_lines.append(f"\n## Topics Covered\n")
        for c, kws in CATEGORIES.items():
            summary_lines.append(f"- **{c.title()}**: {', '.join(kws[:5])}\n")
        st.download_button(
            "📝 Full summary (Markdown)",
            "".join(summary_lines),
            "campus_summary.md",
            "text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Clubs Directory")
    st.download_button(
        "🎭 Clubs & Societies (CSV)",
        agent.clubs.df.to_csv(index=False),
        "campus_clubs.csv",
        "text/csv",
    )
    st.download_button(
        "📍 Locations & Map Data (CSV)",
        agent.locations.df.to_csv(index=False),
        "campus_locations.csv",
        "text/csv",
    )

# ════════════════════════════════════════════════
# TAB 6: ANALYTICS
# ════════════════════════════════════════════════
with tab_stats:
    st.subheader("📊 Usage Analytics")
    df = store.frame()
    if df.empty:
        st.info("No questions yet. Ask something in the Chat tab to see analytics here.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Total Questions", len(df))
        b.metric("Answer Rate", f"{df['answered'].mean():.0%}")
        fb = df["feedback"].dropna()
        c.metric("Helpful votes", f"{fb.mean():.0%}" if len(fb) else "—")
        d.metric("Unique Topics", df["category"].nunique())

        st.divider()
        left2, right2 = st.columns(2)
        with left2:
            st.subheader("Popular Topics")
            st.bar_chart(df["category"].value_counts())
        with right2:
            st.subheader("Questions Over Time")
            df["ts"] = pd.to_datetime(df["ts"])
            daily = df.set_index("ts").resample("D").size().rename("count")
            if not daily.empty:
                st.line_chart(daily)

        st.subheader("Information Gaps (unanswered or downvoted)")
        gaps = df[(df["answered"] == 0) | (df["feedback"] == 0)]
        if gaps.empty:
            st.success("No gaps detected — great coverage!")
        else:
            st.dataframe(
                gaps[["ts", "query", "category"]].tail(20),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Recent Questions")
        st.dataframe(
            df[["ts", "query", "category", "answered", "feedback"]].tail(10),
            use_container_width=True,
            hide_index=True,
        )
