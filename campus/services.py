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

    def upcoming(self, query: str = "", limit: int = 5, today: date | None = None) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        today = pd.Timestamp(today or date.today())
        up = self.df[self.df["_d"] >= today].sort_values("_d")
        hit = _rank(up, query, ["title", "club", "category"], limit) if query else up
        return (hit if not hit.empty else up).head(limit)

    def past(self, limit: int = 5, today: date | None = None) -> pd.DataFrame:
        """Return recently past events."""
        if self.df.empty:
            return self.df
        today = pd.Timestamp(today or date.today())
        past = self.df[self.df["_d"] < today].sort_values("_d", ascending=False)
        return past.head(limit)

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

    def describe(self, query: str = "") -> str:
        up = self.upcoming(query)
        if up.empty:
            return "No upcoming events found in the events calendar."
        return "\n".join(f"{r['title']} - {r['date']} {r['time']} at {r['venue']} ({r['club']}). "
                         f"{r['description']} Registration: {r.get('registration','') or 'n/a'}"
                         for _, r in up.iterrows())

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
# Club Service  (new)
# ---------------------------------------------------------------------------

class ClubService:
    """Service for student clubs, societies and activity groups."""

    def __init__(self, path):
        self.df = _read(path)

    def find(self, query: str, n: int = 5) -> pd.DataFrame:
        return _rank(self.df, query, ["name", "type", "description"], n)

    def by_type(self, club_type: str) -> pd.DataFrame:
        """Filter by type: technical / cultural / social / sports / residents."""
        if self.df.empty:
            return self.df
        mask = self.df["type"].str.lower().str.contains(club_type.lower(), na=False)
        return self.df[mask]

    def describe(self, query: str = "") -> str:
        rows = self.find(query) if query else self.df
        if rows.empty:
            return "No clubs found."
        return "\n".join(
            f"{r['name']} ({r.get('type','')}) — {r.get('description','')}. "
            f"Join: {r.get('how_to_join','')}. Meets: {r.get('meets','')}. Contact: {r.get('contact','')}"
            for _, r in rows.iterrows()
        )

    def join_guide(self, query: str) -> str:
        """Return structured how-to-join guide for matched clubs."""
        rows = self.find(query, n=3)
        if rows.empty:
            return "No matching club found."
        lines = []
        for _, r in rows.iterrows():
            lines.append(
                f"**{r['name']}** [{r.get('type','').upper()}]\n"
                f"  How to join: {r.get('how_to_join','')}\n"
                f"  Meeting schedule: {r.get('meets','')}\n"
                f"  Contact: {r.get('contact','')}"
            )
        return "\n\n".join(lines)
