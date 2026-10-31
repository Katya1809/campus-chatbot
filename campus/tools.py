"""Comprehensive campus information retrieval tools for the CampusAgent.

This module provides structured LangChain tools that integrate:
  - Document processing (PDF, DOCX, MD/TXT, CSV, ICS calendars)
  - Web scraping with structured extraction
  - Location services with campus map data
  - Event calendar processing and filtering
  - Contact directory look-ups
  - Club/society directory searches
  - Campus overview synthesis
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool

from .services import ContactService, EventService, LocationService, ClubService
from .engagement import EngagementService
from .recommender import ActivityRecommender

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_str(v: Any) -> str:
    return "" if pd.isna(v) else str(v).strip()


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def make_location_tool(svc: LocationService, college: str) -> StructuredTool:
    """Tool: find a campus building/room with coordinates and a Maps link."""
    def _fn(query: str) -> str:
        rows = svc.find(query, n=3)
        if rows.empty:
            return "No matching location found. Try a building name, department, or facility keyword."
        parts = []
        for _, r in rows.iterrows():
            maps = LocationService.maps_url(r, college)
            timings = _safe_str(r.get("timings", "")) or "n/a"
            desc = _safe_str(r.get("description", ""))
            parts.append(
                f"**{r['name']}** — {r['building']}, {r['floor']}\n"
                f"  Category: {r.get('category','')}\n"
                f"  {desc}\n"
                f"  Timings: {timings}\n"
                f"  📍 Map: {maps}"
            )
        return "\n\n".join(parts)

    return StructuredTool.from_function(
        _fn, name="find_location",
        description=(
            "Find a campus building, room, lab, hostel, canteen, or facility. "
            "Returns floor, timings, description and a Google Maps link. "
            "Use for questions about where something is located."
        ),
    )


def make_events_tool(svc: EventService) -> StructuredTool:
    """Tool: list upcoming events, filterable by keyword or category."""
    def _fn(query: str = "") -> str:
        up = svc.upcoming(query=query, limit=8)
        if up.empty:
            return "No upcoming events found in the events calendar."
        today = date.today()
        lines = []
        for _, r in up.iterrows():
            d = pd.to_datetime(r["date"], errors="coerce")
            days_away = (d.date() - today).days if not pd.isna(d) else "?"
            tag = f"(today)" if days_away == 0 else f"(in {days_away} days)" if isinstance(days_away, int) and days_away > 0 else ""
            reg = _safe_str(r.get("registration", "")) or "walk-in"
            ptype = _safe_str(r.get("participation_type", "Individual"))
            prizes = f" | 🏆 {_safe_str(r.get('prizes', ''))}" if r.get("prizes") else ""
            lines.append(
                f"• **{r['title']}** {tag} [{_safe_str(r.get('category','')).upper()}]\n"
                f"  📅 {r['date']} at {r['time']} | 📍 {r['venue']}\n"
                f"  Organised by: {r.get('club','')}\n"
                f"  Format: {ptype}{prizes}\n"
                f"  {r.get('description','')}\n"
                f"  Registration: {reg}"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        _fn, name="upcoming_events",
        description=(
            "List upcoming campus events (fests, workshops, sports, cultural shows, hackathons). "
            "Pass a keyword to filter (e.g. 'sports', 'hackathon', 'cultural', 'robotics'). "
            "Returns dates, venues, organisers, format and registration details."
        ),
    )


def make_event_registration_tool(svc: EventService) -> StructuredTool:
    """Tool: look up step-by-step registration process, deadlines, portals, and POC for cultural and technical events."""
    def _fn(query: str = "") -> str:
        return svc.registration_guide(query)

    return StructuredTool.from_function(
        _fn, name="event_registration_process",
        description=(
            "Get detailed step-by-step registration procedures for campus events, competitions, hackathons, "
            "and fests (Alma Fiesta, Wissenaire, E-Summit, RoboWars, Dance, Bands, etc.). "
            "Returns registration portals, eligibility, team size limits, submission deadlines, prizes, and coordinator contacts."
        ),
    )


def make_participation_guidance_tool(svc: EventService) -> StructuredTool:
    """Tool: get practical event participation guidance, OD leave advice, rules, and General Championship (GC) points."""
    def _fn(query: str = "") -> str:
        return svc.participation_guidance(query)

    return StructuredTool.from_function(
        _fn, name="event_participation_guidance",
        description=(
            "Provides practical participation guidance for students: what to prepare, equipment needed, "
            "team guidelines, General Championship (GC) points contribution, and On-Duty (OD) attendance leave application processes."
        ),
    )


def make_contact_tool(svc: ContactService) -> StructuredTool:
    """Tool: look up faculty, department, or emergency contacts."""
    def _fn(query: str) -> str:
        rows = svc.find(query, n=4)
        if rows.empty:
            return "No matching contact found. Try a name, department, role, or keyword like 'placement' or 'emergency'."
        lines = []
        for _, r in rows.iterrows():
            phone = _safe_str(r.get("phone", ""))
            email = _safe_str(r.get("email", ""))
            office = _safe_str(r.get("office", ""))
            lines.append(
                f"• **{r['name']}** — {r.get('role','')}\n"
                f"  Department: {r.get('department','')}\n"
                f"  📧 {email}  📞 {phone}\n"
                f"  Office: {office}"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        _fn, name="find_contact",
        description=(
            "Look up faculty, department heads, emergency contacts, hostel wardens, "
            "T&P cell, health centre, security, and club secretaries. "
            "Returns name, role, email, phone and office location."
        ),
    )


def make_clubs_tool(svc: ClubService) -> StructuredTool:
    """Tool: search clubs and societies."""
    def _fn(query: str = "") -> str:
        rows = svc.find(query, n=5) if query else svc.df
        if rows.empty:
            return "No clubs found."
        lines = []
        for _, r in rows.iterrows():
            contact = _safe_str(r.get("contact", ""))
            meets = _safe_str(r.get("meets", ""))
            venue = _safe_str(r.get("venue", ""))
            how = _safe_str(r.get("how_to_join", ""))
            timeline = _safe_str(r.get("induction_timeline", ""))
            subcat = _safe_str(r.get("sub_category", ""))
            skills = _safe_str(r.get("skills_gained", ""))
            lines.append(
                f"• **{r['name']}** [{r.get('type','').upper()}] — *{subcat}*\n"
                f"  {r.get('description','')}\n"
                f"  How to join: {how} (Timeline: {timeline})\n"
                f"  Meets: {meets} at {venue}\n"
                f"  Skills gained: {skills}\n"
                f"  Contact: {contact} ({_safe_str(r.get('email',''))})"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        _fn, name="find_club",
        description=(
            "Find student clubs, societies, cultural groups, and sports councils. "
            "Filter by keyword (e.g. 'music', 'coding', 'robotics', 'dance', 'photography', 'finance', 'sports'). "
            "Returns descriptions, sub-specialties, induction instructions, schedules, venues, and contact information."
        ),
    )


def make_club_induction_tool(svc: ClubService) -> StructuredTool:
    """Tool: look up detailed induction instructions, tryouts, and audition processes for clubs."""
    def _fn(query: str = "") -> str:
        return svc.join_guide(query)

    return StructuredTool.from_function(
        _fn, name="club_induction_guide",
        description=(
            "Provides detailed induction guides for joining campus clubs and societies. "
            "Explains audition formats, coding tests, portfolio submissions, eligibility, prerequisites, and meeting times."
        ),
    )


def make_recommendation_tool(recommender: ActivityRecommender) -> StructuredTool:
    """Tool: recommend clubs and campus activities based on user interests, skill level, and time commitment."""
    def _fn(query: str = "") -> str:
        rec = recommender.recommend(query)
        return recommender.format_text_recommendation(rec)

    return StructuredTool.from_function(
        _fn, name="recommend_activities",
        description=(
            "Recommend student clubs, activities, and upcoming events tailored to user interests, hobbies, "
            "skill level (beginner/intermediate/advanced), or time availability. "
            "Use when users ask 'What should I join?', 'Recommend clubs for me', 'I like coding and music', etc."
        ),
    )


def make_engagement_tool(svc: EngagementService) -> StructuredTool:
    """Tool: provide campus engagement roadmaps, General Championship (GC) points guides, EAA rules, and fest leadership calls."""
    def _fn(query: str = "") -> str:
        return svc.describe_engagement(query)

    return StructuredTool.from_function(
        _fn, name="campus_engagement_suggestions",
        description=(
            "Provides suggestions on how to get actively involved in campus life: "
            "4-year student progression roadmap, General Championship (GC) scoring strategy, "
            "compulsory Extra Academic Activities (EAA: NSO/NSS/NCC/Yoga), and fest organizing committee opportunities."
        ),
    )


def make_campus_overview_tool(
    contacts: ContactService,
    events: EventService,
    locations: LocationService,
    clubs: ClubService,
    college: str,
) -> StructuredTool:
    """Tool: synthesise a broad campus overview from all data sources."""
    def _fn(query: str = "") -> str:
        today = date.today()
        upcoming = events.upcoming(limit=3)
        ev_lines = []
        for _, r in upcoming.iterrows():
            ev_lines.append(f"• **{r['title']}**\n  📅 {r['date']} at 📍 {r['venue']}")

        emergency = contacts.find("emergency security ambulance", n=3)
        em_lines = []
        for _, r in emergency.iterrows():
            phone = str(r.get("phone", "")).strip()
            clean_phone = phone.replace(" ", "")
            em_lines.append(f"• **{r['name']}**: 📞 [{phone}](tel:{clean_phone})")

        total_locations = len(locations.df)
        total_clubs = len(clubs.df)
        total_contacts = len(contacts.df)

        return (
            f"### 🏛️ {college} Campus Overview\n\n"
            f"📅 **Data as of**: {today.strftime('%d %b %Y')}\n\n"
            f"• 🏢 **Locations indexed**: {total_locations} buildings & laboratories\n\n"
            f"• 🎭 **Clubs & Societies**: {total_clubs} active student societies\n\n"
            f"• 👥 **Faculty & Staff Directory**: {total_contacts} key contact entries\n\n"
            f"#### 📅 Upcoming Events\n\n" + ("\n\n".join(ev_lines) if ev_lines else "• No upcoming events scheduled.") + "\n\n"
            f"#### 🚨 Emergency Helplines\n\n" + ("\n\n".join(em_lines) if em_lines else "• Contact Campus Security at Main Gate.")
        )

    return StructuredTool.from_function(
        _fn, name="campus_overview",
        description=(
            "Get a comprehensive overview of the campus: statistics, upcoming events, "
            "emergency contacts and facility count. Use for general 'what's on campus' questions."
        ),
    )


def make_document_search_tool(retriever) -> StructuredTool:
    """Tool: semantic / TF-IDF search over all ingested documents."""
    def _fn(query: str, source_type: str = "") -> str:
        stype = source_type.strip() or None
        hits = retriever.search(query, k=5, source_type=stype)
        if not hits:
            return "No relevant passages found in campus documents."
        parts = []
        for h in hits:
            meta = h.metadata
            parts.append(
                f"[{meta.get('source','?')} | {meta.get('type','?')} | {meta.get('category','?')}]\n"
                f"{h.page_content.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    return StructuredTool.from_function(
        _fn, name="search_documents",
        description=(
            "Search all campus documents: handbooks (PDF/MD), web pages, CSV data, "
            "and ICS calendars. Returns the most relevant text passages. "
            "Optionally filter by source_type: 'handbook', 'web', 'csv', 'ics'."
        ),
    )
