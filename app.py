"""Interactive Campus Info Chatbot & Activity Guide - Streamlit UI.

Tabs:
  1. 💬 Chat                  — Conversational Q&A with grounded tools and history
  2. 🎯 Clubs & Recommender   — Comprehensive club database + interactive activity matchmaker
  3. 📅 Events & Registration — Event discovery, participation guidance & registration workflows
  4. 🌟 Campus Engagement     — 4-year journey, General Championship (GC), EAA & fest leadership
  5. 📋 Campus Directory      — Department contacts & broad campus overview
  6. 🗺️ Campus Map            — Interactive map with coordinate plotting and filters
  7. 🔍 Search                — Filtered semantic/TF-IDF search over all documents
  8. 📥 Export                — CSV, ICS Calendar, and Markdown downloads
  9. 📊 Analytics             — Usage metrics, topic breakdowns, and coverage gap detection
"""
import os
import re
import time
from datetime import date, datetime
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
from campus.engagement import EngagementService  # noqa: E402
from campus.ingest import CATEGORIES, load_all  # noqa: E402
from campus.recommender import ActivityRecommender  # noqa: E402
from campus.retriever import CampusRetriever  # noqa: E402
from campus.services import ClubService, EventService, LocationService  # noqa: E402
from campus.store import Store  # noqa: E402

DATA = Path(__file__).parent / "data"
COLLEGE = os.getenv("COLLEGE_NAME", "IIT Bhubaneswar")
URLS = [u.strip() for u in os.getenv("COLLEGE_URLS", "").split(",") if u.strip()]

SAMPLES = [
    "🎯 Recommend clubs for coding & robotics",
    "📋 How to register for Wissenaire RoboWars?",
    "🎸 Alma Fiesta Battle of the Bands registration",
    "🏆 How do I earn GC points for my department?",
    "🤝 How to join D-Gang dance society?",
    "🏃 Compulsory EAA & Yoga requirements",
]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{COLLEGE} Campus Assistant & Club Guide",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, h4 {
    letter-spacing: -0.02em;
    font-weight: 700;
}
header[data-testid="stHeader"] {
    height: 3rem !important;
    background: transparent !important;
}
.block-container {
    padding-top: 3.8rem !important;
    max-width: 1320px;
}
div[data-testid="stChatMessage"] {
    border-radius: 12px;
}

/* Enhanced tabs bar styling */
div[data-testid="stTabs"] {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}
div[data-testid="stTabs"] [role="tablist"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.35rem;
}
button[data-testid="stTab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.85rem !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    background: #f8fafc !important;
    color: #475569 !important;
    white-space: nowrap;
    transition: all 0.18s ease !important;
}
button[data-testid="stTab"]:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
}
button[data-testid="stTab"][aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}

/* Quick-action buttons */
div.stButton > button {
    border-radius: 8px;
    font-size: 0.84rem;
    font-weight: 500;
    padding: 0.4rem 0.75rem;
    transition: all 0.18s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}

/* Match & Recommendation Card */
.match-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-top: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}
.match-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.match-pct {
    display: inline-block;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.35rem;
}

/* Club Cards */
.club-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.club-pill {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.18rem 0.55rem;
    border-radius: 12px;
    margin-right: 0.35rem;
}

/* Event cards with vivid council gradients */
.event-card {
    background: #ffffff;
    border-left: 5px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    border-top: 1px solid #f1f5f9;
    border-right: 1px solid #f1f5f9;
    border-bottom: 1px solid #f1f5f9;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}
.event-card.cultural {
    background: linear-gradient(135deg, #fdf4ff 0%, #ffffff 100%);
    border-left-color: #a855f7;
}
.event-card.technical {
    background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
    border-left-color: #ea580c;
}
.event-card.sports {
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    border-left-color: #16a34a;
}
.event-card.social {
    background: linear-gradient(135deg, #fefce8 0%, #ffffff 100%);
    border-left-color: #ca8a04;
}
.event-card.orientation {
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
    border-left-color: #2563eb;
}
.event-card.interdepartmental {
    background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    border-left-color: #dc2626;
}
.event-card.residents {
    background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
    border-left-color: #0d9488;
}

.prize-badge {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    margin-top: 0.35rem;
}

/* Roadmap step box */
.roadmap-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}

/* Location pills */
.location-pill {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    margin: 0.15rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Cache ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Reading handbooks, clubs, events, and campus documents…")
def load_system():
    docs, warnings = load_all(DATA, URLS)
    retr = CampusRetriever(docs)
    if retr.note:
        warnings.append(retr.note)
    agent = CampusAgent(retr, DATA, COLLEGE)
    return (
        agent,
        warnings,
        datetime.now().strftime("%d %b %Y, %H:%M"),
        len(docs),
    )


@st.cache_resource
def get_store():
    return Store(Path(__file__).parent / "campus.db")


# ── Validation & Ask ─────────────────────────────────────────────────────────

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


# ── Load System ──────────────────────────────────────────────────────────────

agent, load_warnings, loaded_at, n_docs = load_system()
store = get_store()
st.session_state.setdefault("messages", [])

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(f"🎓 {COLLEGE}")
    st.caption("Campus Activity & Club Guide Assistant")
    st.caption(f"Mode: **{agent.mode}** · Search: **{agent.retriever.mode}**")
    st.caption(f"{n_docs} knowledge chunks · loaded {loaded_at}")

    if st.button("🔄 Reload Campus Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Directory Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Clubs & Societies", len(agent.clubs.df))
    col2.metric("Scheduled Events", len(agent.events.df))
    col1.metric("Campuses & Venues", len(agent.locations.df))
    col2.metric("Key Contacts", len(agent.contacts.df))

    st.divider()
    st.subheader("✨ Quick Actions")
    if st.button("🎯 Open Club Matchmaker", use_container_width=True):
        st.session_state.target_tab = "matchmaker"
    if st.button("🏆 General Championship Guide", use_container_width=True):
        st.session_state.target_tab = "gc_guide"
    if st.button("📋 Fest Registration Workflows", use_container_width=True):
        st.session_state.target_tab = "fest_reg"

    st.divider()
    for w in load_warnings:
        st.warning(w)
    st.info(
        "Campus Activity Guide is synchronized with official IIT Bhubaneswar Gymkhana regulations, "
        "fest portals (Alma Fiesta, Wissenaire, E-Summit, Pravaah), and EAA curriculum policies."
    )


# ── Tabs Navigation ──────────────────────────────────────────────────────────

tab_chat, tab_clubs, tab_events, tab_engagement, tab_dash, tab_map, tab_search, tab_export, tab_stats = st.tabs(
    [
        "💬 Chat Assistant",
        "🎯 Clubs & Recommender",
        "📅 Events & Registration",
        "🌟 Campus Engagement",
        "📋 Directory & Overview",
        "🗺️ Campus Map",
        "🔍 Full Search",
        "📥 Data Export",
        "📊 Usage Analytics",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("🤖 IIT Bhubaneswar Campus & Activity Assistant")
    st.caption("Ask questions about clubs, event registrations, auditions, General Championship (GC) points, locations, or regulations.")

    # Quick sample buttons
    cols = st.columns(len(SAMPLES))
    for c, s in zip(cols, SAMPLES):
        if c.button(s, use_container_width=True, key=f"sample_{s[:20]}"):
            st.session_state.pending = s

    st.divider()

    # Chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("sources"):
                    st.caption("📚 Grounded Sources: " + ", ".join(m["sources"]))
                if m.get("tools"):
                    st.caption("🔧 Tools Executed: " + ", ".join(m["tools"]))
                st.feedback("thumbs", key=f"fb_{m['id']}", on_change=feedback_cb, args=(m["id"],))

    prompt = st.chat_input("Ask about robotics inductions, Alma Fiesta band registrations, EAA sports credits…")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing campus directories and databases…"):
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CLUBS & RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_clubs:
    st.subheader("🎯 Campus Activity & Club Guide")
    st.markdown("Discover the 38+ active student societies across Technical, Cultural, Sports, Social, and Residents councils, or get personalized recommendations based on your hobbies.")

    subtab_rec, subtab_explore = st.tabs(["✨ Activity Matchmaker & Recommender", "📚 Comprehensive Clubs Database"])

    # SUBTAB A: Activity Matchmaker
    with subtab_rec:
        st.markdown("#### 🧭 Find Your Perfect Campus Society Match")
        st.caption("Select your passions, preferred commitment level, and primary goals to receive tailor-made club matches and an action plan.")

        with st.form("recommender_form"):
            c1, c2 = st.columns([3, 2])
            with c1:
                all_interest_options = [
                    "AI & Robotics",
                    "Competitive Coding & Algorithms",
                    "Web Development & UI/UX",
                    "Vocal & Instrumental Music",
                    "Western & Classical Dance",
                    "Stage Plays & Street Theatre (Nukkad)",
                    "Fine Arts & Digital Painting",
                    "Filmmaking & Video Editing",
                    "Photography & Photojournalism",
                    "Debating & Model UN",
                    "Hindi Literature & Poetry",
                    "Trivia Quizzing",
                    "Astronomy & Stargazing",
                    "Finance & Stock Market",
                    "Startups & Entrepreneurship",
                    "Aeromodelling & Drones",
                    "Competitive Football & Futsal",
                    "Cricket & Net Practice",
                    "Badminton & Indoor Racquet",
                    "Basketball & Court Games",
                    "Athletics Track & Gym Fitness",
                    "Swimming & Water Sports",
                    "Chess & Mind Tactics",
                    "Social Work & Blood Donation",
                    "Environment & Rural Mentorship",
                    "Traditional Arts & Odishi Dance",
                    "Yoga & Preventive Wellness",
                ]
                selected_interests = st.multiselect(
                    "What are your core interests and hobbies?",
                    all_interest_options,
                    default=["AI & Robotics", "Competitive Coding & Algorithms", "Vocal & Instrumental Music"],
                )

            with c2:
                skill_lvl = st.selectbox(
                    "Your experience level",
                    [
                        "Beginner (Want to learn from scratch with workshops)",
                        "Intermediate (Have practicing experience / hobbyist)",
                        "Advanced (Looking for competitive teams / Inter-IIT)",
                    ],
                )
                time_comm = st.selectbox(
                    "Available weekly time commitment",
                    [
                        "Casual (1-2 hours/week, weekend workshops)",
                        "Moderate (3-5 hours/week, regular meetings)",
                        "Dedicated (6+ hours/week, core team / daily practice)",
                    ],
                )
                goal = st.selectbox(
                    "Primary campus objective",
                    [
                        "Skill building & Projects",
                        "Meet friends & Fun campus life",
                        "Win competitions & GC points",
                        "Social impact & Community service",
                    ],
                )

            submitted = st.form_submit_button("🚀 Find My Matches & Generate Action Plan", use_container_width=True)

        if submitted or "last_rec" not in st.session_state:
            with st.spinner("Calculating multi-factor compatibility scores…"):
                rec_result = agent.recommender.recommend(
                    interests=selected_interests,
                    skill_level=skill_lvl,
                    time_commitment=time_comm,
                    primary_goal=goal,
                    top_k=4,
                )
                st.session_state.last_rec = rec_result

        rec_data = st.session_state.last_rec

        st.divider()
        st.subheader("🌟 Top Recommended Societies for You")

        if rec_data.get("matched_clubs"):
            rec_cols = st.columns(len(rec_data["matched_clubs"]))
            for idx, cl in enumerate(rec_data["matched_clubs"]):
                with rec_cols[idx]:
                    st.markdown(
                        f"<div class='match-card'>"
                        f"<span class='match-pct'>{cl['match_pct']}% Match</span><br>"
                        f"<strong>{cl['name']}</strong><br>"
                        f"<span class='club-pill'>{cl['type'].upper()}</span> "
                        f"<small style='color:#4b5563'>{cl['sub_category']}</small><br><br>"
                        f"<small>💡 <em>{cl['reasons'][0]}</em></small><br><br>"
                        f"📅 <strong>Meets</strong>: {cl['meets']}<br>"
                        f"📍 <strong>Venue</strong>: {cl['venue']}<br>"
                        f"🚀 <strong>How to Join</strong>: {cl['how_to_join']}<br><br>"
                        f"👤 <strong>Lead</strong>: {cl['contact']}<br>"
                        f"<a href='mailto:{cl['email']}?subject=Interest%20in%20{cl['name']}' style='font-size:0.8rem;text-decoration:none;color:#2563eb;font-weight:600;'>✉️ Email Secretary</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No specific societies matched your query. Try selecting broader interests or choosing 'All'.")

        st.divider()
        rec_left, rec_right = st.columns([1, 1])

        with rec_left:
            st.subheader("📅 Aligned Upcoming Events & Competitions")
            if rec_data["matched_events"]:
                for ev in rec_data["matched_events"]:
                    st.markdown(
                        f"<div class='event-card {ev['category']}'>"
                        f"<strong>{ev['title']}</strong> "
                        f"<span style='font-size:0.75rem;color:#6b7280'>({ev['category'].title()})</span><br>"
                        f"📅 {ev['date']} at {ev['time']} | 📍 {ev['venue']}<br>"
                        f"Format: <strong>{ev['participation_type']}</strong><br>"
                        f"<small>{ev['description']}</small><br>"
                        f"Registration: <strong>{ev['registration']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No upcoming events matched this specific filter.")

        with rec_right:
            st.subheader("🗺️ Your 30-Day Campus Engagement Plan")
            for step in rec_data["action_plan"]:
                st.markdown(
                    f"<div class='roadmap-box'>"
                    f"<strong>{step['week']}</strong>: {step['milestone']}<br>"
                    f"<small style='color:#374151'>{step['guidance']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # SUBTAB B: Comprehensive Club Database Explorer
    with subtab_explore:
        st.markdown("#### 📚 Comprehensive Clubs & Societies Directory")
        st.caption(f"Browse all {len(agent.clubs.df)} registered student societies and welfare bodies at IIT Bhubaneswar.")

        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            council_types = ["All", "technical", "cultural", "sports", "social", "residents"]
            sel_council = st.selectbox("Filter by Council / Group", council_types, format_func=lambda x: x.title(), key="club_dir_council")
        with fcol2:
            search_club = st.text_input("Search societies by name, domain, or skills", placeholder="e.g. robotics, dance, python, cad, stage", key="club_dir_search")

        filtered_clubs = agent.clubs.df.copy()
        if sel_council != "All":
            filtered_clubs = filtered_clubs[filtered_clubs["type"].str.lower() == sel_council.lower()]
        if search_club.strip():
            filtered_clubs = agent.clubs.find(search_club, n=len(filtered_clubs))

        st.caption(f"Showing {len(filtered_clubs)} matching clubs")

        for _, r in filtered_clubs.iterrows():
            with st.expander(f"**{r['name']}** — *{r.get('sub_category','')}* [{r.get('type','').upper()}]"):
                c_a, c_b = st.columns([3, 2])
                with c_a:
                    st.markdown(f"**Description & Mission**:\n{r.get('description','')}")
                    st.markdown(f"**Weekly Activities & Projects**:\n{r.get('key_activities','')}")
                    st.markdown(f"**Key Skills Gained**:\n`{r.get('skills_gained','')}`")
                    if r.get("flagship_events"):
                        st.markdown(f"**Flagship Competitions / Festivals**:\n🏆 *{r.get('flagship_events','')}*")
                with c_b:
                    st.markdown(
                        f"**🤝 How to Join & Induction**:\n{r.get('how_to_join','')}  \n\n"
                        f"**Induction Timeline**: {r.get('induction_timeline','')}  \n"
                        f"**Eligibility**: {r.get('eligibility','')}  \n"
                        f"**Prerequisites**: {r.get('prerequisites','')}  \n"
                        f"**Meets**: ⏰ {r.get('meets','')}  \n"
                        f"**Venue**: 📍 {r.get('venue','')}  \n"
                        f"**Lead Coordinator**: {r.get('contact','')}  \n"
                        f"📧 [{r.get('email','')}](mailto:{r.get('email','')})"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EVENTS & REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_events:
    st.subheader("📅 Event Discovery & Registration Master Guide")
    st.markdown("Discover technical hackathons, cultural stage battles, sports tournaments, and orientation sessions with complete step-by-step registration workflows.")

    # Top metrics
    em1, em2, em3, em4 = st.columns(4)
    em1.metric("Total Events", len(agent.events.df))
    em2.metric("Cultural Shows", len(agent.events.df[agent.events.df["category"] == "cultural"]))
    em3.metric("Technical & Fests", len(agent.events.df[agent.events.df["category"] == "technical"]))
    em4.metric("Sports & Leagues", len(agent.events.df[agent.events.df["category"] == "sports"]))

    st.divider()

    # Event Discovery Controls
    flt_col1, flt_col2, flt_col3 = st.columns([1, 1, 2])
    with flt_col1:
        cat_choices = ["All", "cultural", "technical", "sports", "social", "orientation", "interdepartmental", "residents"]
        ev_cat_filter = st.selectbox("Category", cat_choices, format_func=lambda x: x.title(), key="ev_tab_cat")
    with flt_col2:
        part_choices = ["All", "Solo", "Team", "Band Team", "Open Attendance", "Department Contingent"]
        ev_part_filter = st.selectbox("Participation Type", part_choices, key="ev_tab_part")
    with flt_col3:
        ev_search = st.text_input("Search events", placeholder="e.g. RoboWars, HackOdisha, Battle of the Bands, Nukkad", key="ev_tab_search")

    events_to_show = agent.events.upcoming(
        query=ev_search,
        limit=50,
        category="" if ev_cat_filter == "All" else ev_cat_filter,
        participation_type="" if part_choices[0] == ev_part_filter else ev_part_filter,
    )

    if events_to_show.empty:
        st.info("No events match your selected filters. Try choosing 'All' or a different search term.")
    else:
        today = date.today()
        for _, r in events_to_show.iterrows():
            cat = str(r.get("category", "")).lower()
            d = pd.to_datetime(r["date"], errors="coerce")
            days_away = (d.date() - today).days if not pd.isna(d) else "?"
            countdown = f"(in {days_away} days)" if isinstance(days_away, int) and days_away > 0 else "(today!)" if days_away == 0 else ""
            prize_str = f"<span class='prize-badge'>🏆 {r['prizes']}</span>" if r.get("prizes") else ""

            st.markdown(
                f"<div class='event-card {cat}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<h4 style='margin:0;color:#1e293b;'>{r['title']} <span style='font-size:0.8rem;color:#64748b;font-weight:400;'>{countdown}</span></h4>"
                f"<span class='club-pill'>{cat.upper()}</span>"
                f"</div>"
                f"📅 <strong>{r['date']}</strong> at 🕐 <strong>{r['time']}</strong> | 📍 <strong>{r['venue']}</strong><br>"
                f"Organised by: <em>{r.get('club','')}</em> &nbsp;|&nbsp; Format: <strong>{r.get('participation_type','Individual')}</strong> (Team: {r.get('team_size','1')})<br>"
                f"{r.get('description','')}<br>"
                f"{prize_str}<br>"
                f"<small style='color:#059669;font-weight:600;'>Registration Status: {r.get('registration','')}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Expandable Step-by-Step Registration & Participation Master Guide
            with st.expander(f"📋 Step-by-Step Registration & Participation Master Guide: {r['title']}"):
                rg_col1, rg_col2 = st.columns([3, 2])
                with rg_col1:
                    st.markdown("#### 📝 Step-by-Step Registration Workflow")
                    st.markdown(r.get("registration_process", "Contact the organizing committee at SAC."))

                    st.markdown("#### 💡 Participation Advice & Preparation")
                    st.markdown(r.get("participation_guidance", "Review competition rules and report on time."))

                    st.markdown("#### 📝 Attendance On-Duty (OD) Leave & General Championship Points")
                    st.info(
                        "• **Attendance OD Leave**: Students officially representing their department or institute can claim "
                        "On-Duty (OD) attendance credit by getting the event OD form signed by the Faculty PIC.  \n"
                        "• **General Championship (GC) Contribution**: Participating awards direct points to your department "
                        "in the annual GC shield standings!"
                    )

                with rg_col2:
                    st.markdown("#### 👥 Team & Eligibility")
                    st.markdown(f"• **Eligibility**: {r.get('eligibility','Open to all students')}")
                    st.markdown(f"• **Team Size**: {r.get('team_size','Individual')}")
                    if r.get("prizes"):
                        st.markdown(f"• **Prizes**: {r.get('prizes')}")

                    st.markdown("#### 🔗 Official Portal & Point of Contact")
                    portal = r.get("portal_url", "")
                    if portal.startswith("http"):
                        st.markdown(f"[🌐 Open Registration Portal / Link]({portal})")
                    elif portal.startswith("mailto"):
                        st.markdown(f"[✉️ Email Coordinator]({portal})")
                    else:
                        st.markdown(f"Portal: `{portal}`")

                    st.markdown(f"**Point of Contact**:  \n👤 {r.get('poc_contact','Event Student Coordinator')}")

                    # Single event ICS download button
                    single_ics = EventService.to_ics(pd.DataFrame([r]))
                    st.download_button(
                        f"📅 Add {r['title'][:15]}… to Calendar (.ics)",
                        single_ics,
                        f"{r['title'][:20].replace(' ','_')}.ics",
                        "text/calendar",
                        key=f"ics_btn_{r['title'][:20]}",
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CAMPUS ENGAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_engagement:
    st.subheader("🌟 Campus Engagement & Leadership Opportunities")
    st.markdown("Maximize your college experience: four-year roadmap, General Championship (GC) trophy strategies, EAA sports/yoga curriculum, and fest leadership portfolios.")

    eng_tab1, eng_tab2, eng_tab3, eng_tab4 = st.tabs(
        [
            "🗺️ 4-Year Student Roadmap",
            "🏆 General Championship (GC) Guide",
            "🏃 Compulsory EAA & Yoga",
            "🎪 Fest Organizing & Leadership",
        ]
    )

    # 1. Four-Year Roadmap
    with eng_tab1:
        st.markdown("#### 🎓 The 4-Year Undergraduate Leadership Journey")
        roadmap = agent.engagement.four_year_roadmap()
        for y_key, y_info in roadmap.items():
            with st.expander(f"**{y_info['title']}** — *{y_info['focus']}*", expanded=(y_key == "year1")):
                st.markdown(f"**Recommended Time Commitment**: `{y_info['recommended_time']}`")
                st.markdown("**Key Milestones & Goals**:")
                for m in y_info["key_milestones"]:
                    st.markdown(f"- {m}")
                st.caption(f"⚠️ **Pitfall to Avoid**: {y_info['avoid']}")

    # 2. General Championship (GC) Guide
    with eng_tab2:
        st.markdown("#### 🏆 IIT Bhubaneswar General Championship (GC) Master Strategy")
        gc = agent.engagement.general_championship_guide()
        st.write(gc["overview"])

        st.divider()
        gc_cols = st.columns(3)
        for idx, (p_name, p_data) in enumerate(gc["pillars"].items()):
            with gc_cols[idx]:
                st.markdown(f"### {p_name} ({p_data['weight']})")
                st.caption(p_data["how_to_participate"])
                st.markdown("**Core Events**:")
                for ev in p_data["events"]:
                    st.markdown(f"• {ev}")

        st.divider()
        st.subheader("Points Scoring System")
        sc_cols = st.columns(len(gc["scoring_system"]))
        for idx, s in enumerate(gc["scoring_system"]):
            sc_cols[idx].info(s)

        st.subheader("💡 Strategic Advice for Freshers")
        for tip in gc["tips_for_freshers"]:
            st.markdown(f"- {tip}")

    # 3. Compulsory EAA & Yoga
    with eng_tab3:
        st.markdown("#### 🏃 Extra Academic Activities (EAA) & Compulsory Yoga")
        eaa = agent.engagement.extra_academic_activities_guide()
        st.write(eaa["overview"])

        eaa_cols = st.columns(3)
        for idx, opt in enumerate(eaa["options"]):
            with eaa_cols[idx]:
                st.markdown(f"### {opt['name']}")
                st.markdown(f"**Activities**: {opt['activities']}")
                st.markdown(f"**Timing**: `{opt['timing']}`")
                st.success(f"**Key Benefit**: {opt['key_benefit']}")

        st.divider()
        st.warning(f"**Compulsory Yoga Course & Practical Exam**:\n{eaa['compulsory_yoga']}")

    # 4. Fest Leadership Portfolios
    with eng_tab4:
        st.markdown("#### 🎪 Major Campus Fest Organizing Committees & Leadership Calls")
        fests = agent.engagement.fest_leadership_opportunities()
        for f in fests:
            with st.expander(f"**{f['fest']}** — *{f['scale']}*"):
                st.markdown(f"**Recruitment Timeline**: `{f['recruitment_timeline']}`")
                st.markdown("**Sub-Committees to Join**:")
                for sub in f["sub_committees"]:
                    st.markdown(f"• {sub}")
                st.markdown(f"**Contact**: [{f['contact']}](mailto:{f['contact']})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DIRECTORY & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("📋 Campus Contacts & Topic Directory")

    st.subheader("Browse by Topic")
    cats = [c for c in CATEGORIES if c != "contacts"]
    btn_cols = st.columns(min(len(cats), 5))
    for i, c in enumerate(cats):
        if btn_cols[i % 5].button(c.title(), key=f"cat_{c}", use_container_width=True):
            st.session_state.dash = ask(agent, store, f"Tell me about {c} at the college")
    if st.session_state.get("dash"):
        with st.expander("📖 Topic Answer", expanded=True):
            st.markdown(st.session_state.dash["answer"])

    st.divider()

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("👥 Key Faculty & Department Contacts")
        dept_filter = st.text_input("Search contacts", placeholder="e.g. hostel, placement, health, dean", key="dir_contact_search")
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

    with c_right:
        st.subheader("🏛️ Campus Overview Synthesis")
        overview_text = agent.tools["campus_overview"].invoke({"query": ""})
        st.markdown(overview_text)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: CAMPUS MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("🗺️ Campus Map & Facilities Directory")

    loc_df = agent.locations.all_with_coords()
    col_left, col_right = st.columns([3, 2])

    with col_left:
        if not loc_df.empty:
            all_cats = sorted(loc_df["category"].dropna().unique().tolist())
            selected_cats = st.multiselect(
                "Filter by category",
                all_cats,
                default=all_cats,
                key="map_cat_filter",
            )
            map_df = loc_df[loc_df["category"].isin(selected_cats)] if selected_cats else loc_df

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
        search_loc = st.text_input("Search location", placeholder="e.g. library, hostel, canteen, SAC", key="map_search")
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

    st.divider()
    st.subheader("Locations by Category")
    cat_summary = agent.locations.category_summary()
    for cat, names in sorted(cat_summary.items()):
        pills = "".join(f"<span class='location-pill'>{n}</span>" for n in names)
        st.markdown(f"**{cat.title()}**: {pills}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: FULL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("🔍 Search Full Campus Knowledge Base")
    q = st.text_input("Search query", max_chars=200, placeholder="e.g. attendance policy, Wissenaire robotics, placement drive")
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("📥 Download Campus Databases & Calendars")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎭 Clubs & Societies")
        st.download_button(
            "📋 Comprehensive Clubs Database (CSV)",
            agent.clubs.df.to_csv(index=False),
            "campus_clubs_comprehensive.csv",
            "text/csv",
            use_container_width=True,
        )
        st.download_button(
            "👥 Campus Contacts (CSV)",
            agent.contacts.df.to_csv(index=False),
            "campus_contacts.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 📅 Events & Competitions")
        ev_all = agent.events.upcoming(limit=50)
        st.download_button(
            "📅 Complete Events Calendar (ICS)",
            EventService.to_ics(ev_all),
            "campus_events.ics",
            "text/calendar",
            use_container_width=True,
        )
        st.download_button(
            "📊 Events & Registration Guides (CSV)",
            agent.events.df.drop(columns=["_d"], errors="ignore").to_csv(index=False),
            "campus_events_and_registration.csv",
            "text/csv",
            use_container_width=True,
        )

    with col3:
        st.markdown("#### 🗺️ Engagement Guide & Locations")
        # Build Markdown guide
        engagement_md = [
            f"# {COLLEGE} Comprehensive Campus Activity & Engagement Guide\n",
            f"Generated: {datetime.now().strftime('%d %b %Y')}\n\n",
            "## 1. Student Societies & Clubs\n",
        ]
        for _, r in agent.clubs.df.iterrows():
            engagement_md.append(f"### {r['name']} [{r.get('type','').upper()}]\n")
            engagement_md.append(f"- **Domain**: {r.get('sub_category','')}\n")
            engagement_md.append(f"- **Description**: {r.get('description','')}\n")
            engagement_md.append(f"- **How to Join**: {r.get('how_to_join','')}\n")
            engagement_md.append(f"- **Meets**: {r.get('meets','')} at {r.get('venue','')}\n")
            engagement_md.append(f"- **Contact**: {r.get('contact','')} ({r.get('email','')})\n\n")

        engagement_md.append("## 2. General Championship (GC) Strategy\n")
        engagement_md.append(agent.engagement.describe_engagement("gc"))
        engagement_md.append("\n\n## 3. Extra Academic Activities (EAA)\n")
        engagement_md.append(agent.engagement.describe_engagement("eaa"))

        st.download_button(
            "📝 Campus Engagement Guide (Markdown)",
            "".join(engagement_md),
            "campus_activity_and_engagement_guide.md",
            "text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "📍 Campus Locations & Map Data (CSV)",
            agent.locations.df.to_csv(index=False),
            "campus_locations.csv",
            "text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("📊 Query & Activity Analytics")
    df = store.frame()
    if df.empty:
        st.info("No questions logged yet. Ask questions in the Chat tab to view real-time query analytics.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Total Questions", len(df))
        b.metric("Answer Rate", f"{df['answered'].mean():.0%}")
        fb = df["feedback"].dropna()
        c.metric("Helpful Votes", f"{fb.mean():.0%}" if len(fb) else "—")
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

        st.subheader("Information Coverage Gaps")
        gaps = df[(df["answered"] == 0) | (df["feedback"] == 0)]
        if gaps.empty:
            st.success("No information gaps detected — comprehensive coverage across all categories!")
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
