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
            lines.append(
                f"• **{r['title']}** {tag}\n"
                f"  📅 {r['date']} at {r['time']} | 📍 {r['venue']}\n"
                f"  Organised by: {r.get('club','')}\n"
                f"  {r.get('description','')}\n"
                f"  Registration: {reg}"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        _fn, name="upcoming_events",
        description=(
            "List upcoming campus events (fests, workshops, sports, cultural shows, hackathons). "
            "Pass a keyword to filter (e.g. 'sports', 'hackathon', 'cultural'). "
            "Returns dates, venues, organisers and registration details."
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
            how = _safe_str(r.get("how_to_join", ""))
            lines.append(
                f"• **{r['name']}** [{r.get('type','').upper()}]\n"
                f"  {r.get('description','')}\n"
                f"  How to join: {how}\n"
                f"  Meets: {meets}  |  Contact: {contact}"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        _fn, name="find_club",
        description=(
            "Find student clubs, societies and cultural groups. "
            "Filter by keyword (e.g. 'music', 'coding', 'robotics', 'dance', 'social'). "
            "Returns description, meeting schedule, how to join, and contact email."
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
            ev_lines.append(f"  - {r['title']} on {r['date']} at {r['venue']}")

        emergency = contacts.find("emergency security ambulance", n=3)
        em_lines = []
        for _, r in emergency.iterrows():
            em_lines.append(f"  - {r['name']}: {r.get('phone','')}")

        total_locations = len(locations.df)
        total_clubs = len(clubs.df)
        total_contacts = len(contacts.df)

        return (
            f"## {college} Campus Overview\n\n"
            f"**Data as of**: {today.strftime('%d %b %Y')}\n\n"
            f"**Locations indexed**: {total_locations} buildings/facilities\n"
            f"**Clubs & Societies**: {total_clubs} registered groups\n"
            f"**Contacts**: {total_contacts} faculty/staff/office entries\n\n"
            f"**Upcoming Events (next 3)**:\n" + ("\n".join(ev_lines) if ev_lines else "  No events scheduled.") + "\n\n"
            f"**Emergency Contacts**:\n" + ("\n".join(em_lines) if em_lines else "  See contacts directory.")
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
                f"{h.page_content[:600]}"
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
