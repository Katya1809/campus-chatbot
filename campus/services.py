"""Structured campus services: locations (+Google Maps links), events, contacts, clubs."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

STOP = set("the a an is are of to in on at for how do i can where what who when which me my and or "
           "from with about tell get find".split())


def tokens(s) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower())) - STOP


def _read(path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p).fillna("") if p.exists() else pd.DataFrame()


def _rank(df: pd.DataFrame, query: str, strong: list[str], n: int) -> pd.DataFrame:
    q = tokens(query)
    if df.empty or not q:
        return df.head(n)

    def score(row):
        s = 3 * len(q & tokens(" ".join(str(row.get(c, "")) for c in strong)))
        return s + len(q & tokens(" ".join(map(str, row.values))))

    scores = df.apply(score, axis=1)
    keep = scores > 0
    return df[keep].assign(_s=scores[keep]).sort_values("_s", ascending=False).head(n).drop(columns="_s")


# ---------------------------------------------------------------------------
# Location Service
# ---------------------------------------------------------------------------

class LocationService:
    def __init__(self, path):
        self.df = _read(path)

    @staticmethod
    def maps_url(row, college: str = "") -> str:
        lat = str(row.get("lat", "")).strip()
        lon = str(row.get("lon", "")).strip()
        if lat and lon and lat != "nan" and lon != "nan":
            return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        name = str(row.get("name", ""))
        return f"https://www.google.com/maps/search/?api=1&query={quote_plus(name + ' ' + college)}"

    def find(self, query: str, n: int = 2) -> pd.DataFrame:
        return _rank(self.df, query, ["name", "aliases", "category"], n)

    def describe(self, query: str, college: str = "") -> str:
        rows = self.find(query)
        if rows.empty:
            return "Location not found."
        return "\n".join(
            f"{r['name']} - {r['building']}, {r['floor']}. {r.get('description','')} "
            f"Timings: {r.get('timings','') or 'n/a'}. Map: {self.maps_url(r, college)}"
            for _, r in rows.iterrows())

    def category_summary(self) -> dict[str, list[str]]:
        """Group locations by category for map/dashboard views."""
        if self.df.empty:
            return {}
        out: dict[str, list[str]] = {}
        for _, r in self.df.iterrows():
            cat = str(r.get("category", "other"))
            out.setdefault(cat, []).append(str(r["name"]))
        return out

    def all_with_coords(self) -> pd.DataFrame:
        """Return rows that have valid lat/lon coordinates."""
        if self.df.empty:
            return self.df
        df = self.df.copy()
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        return df.dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------------------
# Event Service
# ---------------------------------------------------------------------------

class EventService:
    def __init__(self, path):
        self.df = _read(path)
        if not self.df.empty:
            self.df["_d"] = pd.to_datetime(self.df["date"], errors="coerce")

    def upcoming(
        self,
        query: str = "",
        limit: int = 8,
        category: str = "",
        participation_type: str = "",
        today: date | None = None,
    ) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        today = pd.Timestamp(today or date.today())
        up = self.df[self.df["_d"] >= today].sort_values("_d")

        if category and category.lower() != "all":
            up = up[up["category"].str.lower().str.contains(category.lower(), na=False)]

        if participation_type and participation_type.lower() != "all":
            up = up[up["participation_type"].str.lower().str.contains(participation_type.lower(), na=False)]

        if query:
            hit = _rank(up, query, ["title", "club", "category", "tags", "description"], limit)
            return hit if not hit.empty else up.head(limit)
        return up.head(limit)

    def past(self, limit: int = 5, today: date | None = None) -> pd.DataFrame:
        """Return recently past events."""
        if self.df.empty:
            return self.df
        today = pd.Timestamp(today or date.today())
        past = self.df[self.df["_d"] < today].sort_values("_d", ascending=False)
        return past.head(limit)

    def cultural_events(self, limit: int = 10) -> pd.DataFrame:
        """Return upcoming cultural and arts events."""
        return self.upcoming(category="cultural", limit=limit)

    def technical_events(self, limit: int = 10) -> pd.DataFrame:
        """Return upcoming technical, robotics, and coding events."""
        return self.upcoming(category="technical", limit=limit)

    def by_month(self, year: int, month: int) -> pd.DataFrame:
        """Return events for a specific month."""
        if self.df.empty:
            return self.df
        return self.df[(self.df["_d"].dt.year == year) & (self.df["_d"].dt.month == month)]

    def by_category(self, category: str, limit: int = 10) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        mask = self.df["category"].str.lower().str.contains(category.lower(), na=False)
        return self.df[mask].head(limit)

    def by_participation_type(self, ptype: str, limit: int = 10) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        mask = self.df["participation_type"].str.lower().str.contains(ptype.lower(), na=False)
        return self.df[mask].head(limit)

    def describe(self, query: str = "") -> str:
        up = self.upcoming(query)
        if up.empty:
            return "No upcoming events found in the events calendar."
        lines = []
        for _, r in up.iterrows():
            reg = r.get("registration", "") or "Walk-in"
            ptype = r.get("participation_type", "Individual")
            prizes = f" | Prizes: {r['prizes']}" if r.get("prizes") else ""
            lines.append(
                f"• {r['title']} [{r.get('category','').upper()}] - {r['date']} at {r['time']} ({r['venue']})\n"
                f"  Organised by: {r.get('club','')}\n"
                f"  Format: {ptype}{prizes}\n"
                f"  Description: {r.get('description','')}\n"
                f"  Registration: {reg}"
            )
        return "\n\n".join(lines)

    def registration_guide(self, query: str = "") -> str:
        """Return comprehensive registration processes for cultural, technical, and sports events."""
        up = self.upcoming(query, limit=4)
        if up.empty:
            # Fallback to search over all events
            up = _rank(self.df, query, ["title", "club", "category", "tags"], 4) if query else self.df.head(4)
        if up.empty:
            return "No matching event registration details found. Try an event name like 'Wissenaire', 'Alma Fiesta', 'Hackathon', or 'RoboWars'."

        guides = []
        for _, r in up.iterrows():
            proc = r.get("registration_process", "") or "Contact the organizing committee at SAC."
            poc = r.get("poc_contact", "") or "Event Student Coordinator"
            portal = r.get("portal_url", "")
            prizes = r.get("prizes", "Certificates & Trophy")
            elig = r.get("eligibility", "Open to all students")
            team = r.get("team_size", "Individual")

            guides.append(
                f"### 📋 Registration Guide: {r['title']}\n"
                f"• **Council / Category**: {r.get('category','').title()} | **Organiser**: {r.get('club','')}\n"
                f"• **Date & Venue**: 📅 {r.get('date','')} at 🕐 {r.get('time','')} | 📍 {r.get('venue','')}\n"
                f"• **Eligibility**: {elig}\n"
                f"• **Team Size**: {team}\n"
                f"• **Prizes / Incentives**: {prizes}\n"
                f"• **Registration Status**: {r.get('registration','')}\n"
                f"• **Step-by-Step Registration Process**:\n"
                f"  {proc}\n"
                f"• **Official Portal / Form**: {portal or 'Visit SAC Helpdesk'}\n"
                f"• **Point of Contact**: {poc}"
            )
        return "\n\n---\n\n".join(guides)

    def participation_guidance(self, query: str = "") -> str:
        """Return event participation advice, rules, OD leave, and GC points mechanism."""
        up = self.upcoming(query, limit=3)
        if up.empty and query:
            up = _rank(self.df, query, ["title", "club", "category"], 3)
        if up.empty:
            up = self.df.head(3)

        parts = []
        for _, r in up.iterrows():
            guidance = r.get("participation_guidance", "Review competition rules and report on time.")
            parts.append(
                f"### 💡 Participation Guidance: {r['title']}\n"
                f"• **Format**: {r.get('participation_type','Individual')} (Team Size: {r.get('team_size','1')})\n"
                f"• **Schedule & Location**: 📅 {r.get('date','')} {r.get('time','')} at 📍 {r.get('venue','')}\n"
                f"• **Preparation & Guidelines**:\n"
                f"  {guidance}\n"
                f"• **Attendance On-Duty (OD) Leave**:\n"
                f"  Students officially representing the institute or department can submit an OD application "
                f"signed by the Faculty PIC to maintain full attendance credits.\n"
                f"• **General Championship (GC) Contribution**:\n"
                f"  Participation contributes direct department points toward the coveted annual GC Rolling Shield!"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def to_ics(df: pd.DataFrame) -> str:
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CampusInfoBot//EN"]
        for i, r in df.reset_index(drop=True).iterrows():
            d = pd.to_datetime(r["date"], errors="coerce")
            if pd.isna(d):
                continue
            t = re.match(r"^(\d{1,2}):(\d{2})", str(r.get("time", "")))
            start = (f"DTSTART:{d:%Y%m%d}T{int(t[1]):02d}{t[2]}00" if t
                     else f"DTSTART;VALUE=DATE:{d:%Y%m%d}")
            desc = str(r.get("description", "")).replace("\n", "\\n")
            lines += ["BEGIN:VEVENT", f"UID:campus-{i}-{d:%Y%m%d}@campusbot", start,
                      f"SUMMARY:{r['title']}", f"LOCATION:{r['venue']}",
                      f"DESCRIPTION:{desc}", f"CATEGORIES:{r.get('category','')}",
                      "END:VEVENT"]
        return "\r\n".join(lines + ["END:VCALENDAR"])


# ---------------------------------------------------------------------------
# Contact Service
# ---------------------------------------------------------------------------

class ContactService:
    def __init__(self, path):
        self.df = _read(path)

    def find(self, query: str, n: int = 3) -> pd.DataFrame:
        return _rank(self.df, query, ["name", "role", "department"], n)

    def by_department(self, department: str) -> pd.DataFrame:
        """Return all contacts for a department (substring match)."""
        if self.df.empty:
            return self.df
        mask = self.df["department"].str.lower().str.contains(department.lower(), na=False)
        return self.df[mask]

    def describe(self, query: str) -> str:
        rows = self.find(query)
        if rows.empty:
            return "Contact not found."
        return "\n".join(f"{r['name']} - {r['role']}, {r['department']}. Email: {r['email']}. "
                         f"Phone: {r['phone']}. Office: {r['office']}" for _, r in rows.iterrows())


# ---------------------------------------------------------------------------
# Club Service
# ---------------------------------------------------------------------------

class ClubService:
    """Comprehensive service for student clubs, societies, and activity groups."""

    def __init__(self, path):
        self.df = _read(path)

    def find(self, query: str, n: int = 5) -> pd.DataFrame:
        return _rank(
            self.df,
            query,
            ["name", "type", "sub_category", "interests_tags", "skills_gained", "description"],
            n,
        )

    def by_type(self, club_type: str) -> pd.DataFrame:
        """Filter by council/type: technical / cultural / social / sports / residents."""
        if self.df.empty:
            return self.df
        mask = self.df["type"].str.lower().str.contains(club_type.lower(), na=False)
        return self.df[mask]

    def by_subcategory(self, sub_category: str) -> pd.DataFrame:
        """Filter by sub-domain e.g. Robotics, Dance, Competitive Coding."""
        if self.df.empty:
            return self.df
        mask = self.df["sub_category"].str.lower().str.contains(sub_category.lower(), na=False)
        return self.df[mask]

    def describe(self, query: str = "") -> str:
        rows = self.find(query, n=6) if query else self.df
        if rows.empty:
            return "No clubs found."
        lines = []
        for _, r in rows.iterrows():
            skills = f" | Skills: {r['skills_gained']}" if r.get("skills_gained") else ""
            flagship = f" | Flagship: {r['flagship_events']}" if r.get("flagship_events") else ""
            lines.append(
                f"• **{r['name']}** [{r.get('type','').upper()}] — *{r.get('sub_category','')}*\n"
                f"  {r.get('description','')}\n"
                f"  How to join: {r.get('how_to_join','')}\n"
                f"  Induction Timeline: {r.get('induction_timeline','')}\n"
                f"  Meets: {r.get('meets','')} at {r.get('venue','')}\n"
                f"  Contact: {r.get('contact','')} ({r.get('email','')}){flagship}{skills}"
            )
        return "\n\n".join(lines)

    def join_guide(self, query: str) -> str:
        """Return structured how-to-join guide for matched clubs."""
        rows = self.find(query, n=3)
        if rows.empty:
            return "No matching club found. Try keywords like 'robotics', 'dance', 'coding', 'music', 'drama'."
        lines = []
        for _, r in rows.iterrows():
            lines.append(
                f"### 🤝 How to Join: **{r['name']}** [{r.get('type','').upper()}]\n"
                f"• **Domain**: {r.get('sub_category','')}\n"
                f"• **Induction Process**: {r.get('how_to_join','')}\n"
                f"• **Induction Timeline**: {r.get('induction_timeline','')}\n"
                f"• **Eligibility**: {r.get('eligibility','Open to all students')}\n"
                f"• **Prerequisites**: {r.get('prerequisites','None required — beginners welcome!')}\n"
                f"• **Weekly Meets**: {r.get('meets','')} at {r.get('venue','')}\n"
                f"• **Key Skills Gained**: {r.get('skills_gained','')}\n"
                f"• **Lead Coordinator**: {r.get('contact','')} (📧 {r.get('email','')})"
            )
        return "\n\n---\n\n".join(lines)

