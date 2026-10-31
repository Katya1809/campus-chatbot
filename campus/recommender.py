"""Activity & Club Recommendation Engine for IIT Bhubaneswar Campus Assistant.

Matches student interests, skill levels, and time commitments to:
  1. Relevant clubs and societies with tailored induction guidance
  2. Upcoming campus events and flagship competitions
  3. Actionable personalized 4-week engagement action plans
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd

from .services import ClubService, EventService

STOP = set("the a an is are of to in on at for how do i can where what who when which me my and or from with about tell get find".split())


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower())) - STOP


class ActivityRecommender:
    """Intelligent multi-factor recommendation engine for campus clubs and events."""

    def __init__(self, clubs: ClubService, events: EventService):
        self.clubs = clubs
        self.events = events

    def recommend(
        self,
        interests: list[str] | str,
        skill_level: str = "beginner",
        time_commitment: str = "moderate",
        primary_goal: str = "all",
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Compute matching clubs, events, and an action plan based on user profile."""
        if isinstance(interests, str):
            interest_tokens = _tokens(interests)
        else:
            interest_tokens = set()
            for item in interests:
                interest_tokens |= _tokens(item)

        skill_level = skill_level.lower().strip()
        time_commitment = time_commitment.lower().strip()
        primary_goal = primary_goal.lower().strip()

        # Score clubs
        club_scores: list[dict[str, Any]] = []
        for _, r in self.clubs.df.iterrows():
            score = 0.0
            reasons = []

            # 1. Direct interest tag matching (highest weight)
            tags = _tokens(r.get("interests_tags", ""))
            tag_hits = interest_tokens & tags
            if tag_hits:
                score += len(tag_hits) * 35.0
                reasons.append(f"Matches your interest in {', '.join(sorted(tag_hits)[:3])}")

            # 2. Subcategory & description matching
            subcat_tokens = _tokens(r.get("sub_category", ""))
            subcat_hits = interest_tokens & subcat_tokens
            if subcat_hits:
                score += len(subcat_hits) * 20.0

            desc_tokens = _tokens(r.get("description", "")) | _tokens(r.get("skills_gained", ""))
            desc_hits = interest_tokens & desc_tokens
            if desc_hits:
                score += len(desc_hits) * 5.0

            # 3. Skill level factor
            prereq = str(r.get("prerequisites", "")).lower()
            if "beginner" in skill_level or "fresh" in skill_level:
                if "none" in prereq or "beginner" in prereq or "enthusiasm" in prereq:
                    score += 15.0
                    reasons.append("Beginner-friendly with dedicated induction workshops")
            elif "advanced" in skill_level or "competitive" in skill_level:
                if "inter-iit" in str(r.get("flagship_events", "")).lower() or "competitive" in prereq:
                    score += 18.0
                    reasons.append("High competitive caliber & Inter-IIT contingent pathway")

            # 4. Time commitment factor
            meets_str = str(r.get("meets", "")).lower()
            if "casual" in time_commitment or "low" in time_commitment:
                if "sat" in meets_str or "sun" in meets_str or "monthly" in meets_str or "weekly" in meets_str:
                    score += 10.0
            elif "dedicated" in time_commitment or "high" in time_commitment:
                if "daily" in meets_str or "mon and thu" in meets_str or "wed and fri" in meets_str:
                    score += 10.0

            # 5. Primary goal factor
            if "skill" in primary_goal:
                if str(r.get("skills_gained", "")):
                    score += 10.0
            elif "competition" in primary_goal:
                if "yanthrix" in str(r.get("flagship_events", "")).lower() or "contest" in str(r.get("description", "")).lower():
                    score += 12.0
            elif "social" in primary_goal:
                if r.get("type") in ("social", "residents"):
                    score += 15.0

            # Baseline score if empty interests or general match
            if not interest_tokens:
                score = 10.0

            if score > 0:
                match_pct = min(98, int(45 + score * 1.2))
                club_scores.append({
                    "name": r["name"],
                    "type": r.get("type", "general"),
                    "sub_category": r.get("sub_category", ""),
                    "description": r.get("description", ""),
                    "meets": r.get("meets", ""),
                    "venue": r.get("venue", ""),
                    "how_to_join": r.get("how_to_join", ""),
                    "induction_timeline": r.get("induction_timeline", ""),
                    "contact": r.get("contact", ""),
                    "email": r.get("email", ""),
                    "flagship_events": r.get("flagship_events", ""),
                    "skills_gained": r.get("skills_gained", ""),
                    "score": score,
                    "match_pct": match_pct,
                    "reasons": reasons or ["Broad alignment with your campus profile"],
                })

        club_scores.sort(key=lambda x: x["score"], reverse=True)
        matched_clubs = club_scores[:top_k]

        # Score matching upcoming events
        event_matches: list[dict[str, Any]] = []
        today = date.today()
        for _, ev in self.events.upcoming(limit=30).iterrows():
            ev_score = 0.0
            ev_tokens = _tokens(ev.get("tags", "")) | _tokens(ev.get("title", "")) | _tokens(ev.get("description", ""))
            hits = interest_tokens & ev_tokens
            if hits:
                ev_score += len(hits) * 20.0
            # Also check if event club matches top recommended clubs
            matched_club_names = [c["name"].lower() for c in matched_clubs]
            if any(ev.get("club", "").lower() in c or c in ev.get("club", "").lower() for c in matched_club_names):
                ev_score += 25.0

            if ev_score > 0 or not interest_tokens:
                d = pd.to_datetime(ev["date"], errors="coerce")
                days_away = (d.date() - today).days if not pd.isna(d) else "?"
                event_matches.append({
                    "title": ev["title"],
                    "date": ev["date"],
                    "time": ev.get("time", ""),
                    "venue": ev.get("venue", ""),
                    "club": ev.get("club", ""),
                    "category": ev.get("category", ""),
                    "participation_type": ev.get("participation_type", "Individual"),
                    "registration": ev.get("registration", "Open"),
                    "registration_process": ev.get("registration_process", ""),
                    "prizes": ev.get("prizes", ""),
                    "description": ev.get("description", ""),
                    "days_away": days_away,
                    "score": ev_score,
                })

        event_matches.sort(key=lambda x: x["score"], reverse=True)
        matched_events = event_matches[:5]

        # Generate a personalized 30-Day Campus Action Plan
        plan = self._generate_action_plan(matched_clubs, matched_events, skill_level)

        return {
            "matched_clubs": matched_clubs,
            "matched_events": matched_events,
            "action_plan": plan,
            "interests": list(interest_tokens),
            "skill_level": skill_level,
            "time_commitment": time_commitment,
        }

    def _generate_action_plan(
        self, matched_clubs: list[dict[str, Any]], matched_events: list[dict[str, Any]], skill_level: str
    ) -> list[dict[str, str]]:
        c1 = matched_clubs[0]["name"] if matched_clubs else "Campus Clubs"
        c2 = matched_clubs[1]["name"] if len(matched_clubs) > 1 else "Interest Societies"
        ev1 = matched_events[0]["title"] if matched_events else "Upcoming Campus Workshop"

        return [
            {
                "week": "Week 1: Explore & Connect",
                "milestone": f"Attend introductory meetings & follow {c1} & {c2}",
                "guidance": (
                    f"1. Reach out to the student coordinators of {c1} via email to join their discussion groups. "
                    f"2. Visit the club meet venue ({matched_clubs[0].get('venue', 'SAC') if matched_clubs else 'SAC'}) to interact with current members. "
                    f"3. Mark induction dates on your calendar."
                ),
            },
            {
                "week": "Week 2: Hands-on Orientation",
                "milestone": f"Participate in beginner workshops & prepare for inductions",
                "guidance": (
                    f"1. Attend the open fresher tryout or orientation session for {c1}. "
                    f"2. Brush up on foundational tools: {matched_clubs[0].get('skills_gained', 'foundations') if matched_clubs else 'core skills'}. "
                    f"3. Form study or practice circles with fellow freshers in your hostel."
                ),
            },
            {
                "week": "Week 3: Team Formation & Event Registration",
                "milestone": f"Register for {ev1}",
                "guidance": (
                    f"1. Review registration requirements for {ev1}. "
                    f"2. Connect with department teammates or hostel peers for team events. "
                    f"3. Complete portal registration before the deadline."
                ),
            },
            {
                "week": "Week 4: Showcase & Community Contribution",
                "milestone": f"Compete / perform and contribute General Championship points",
                "guidance": (
                    f"1. Take part in the competition or showcase your project work. "
                    f"2. Submit attendance On-Duty (OD) forms to your faculty advisor if representing the institute. "
                    f"3. Celebrate your progress and apply for regular society membership roles!"
                ),
            },
        ]

    def format_text_recommendation(self, rec: dict[str, Any]) -> str:
        """Format recommendation result into concise, beautifully styled Markdown for LLM responses."""
        lines = ["### 🎯 Personalized Campus Activity & Club Recommendations\n"]

        if rec["matched_clubs"]:
            lines.append("#### 🌟 Recommended Clubs & Societies to Join:")
            for c in rec["matched_clubs"][:3]:
                reasons_str = " · ".join(c["reasons"][:2])
                lines.append(
                    f"• **{c['name']}** — **{c['match_pct']}% Match** [{c['type'].upper()}]\n"
                    f"  *{c['sub_category']}*\n"
                    f"  💡 *Why*: {reasons_str}\n"
                    f"  🚀 *How to Join*: {c['how_to_join']}\n"
                    f"  📅 *Meets*: {c['meets']} at {c['venue']}\n"
                    f"  👤 *Contact*: {c['contact']} ({c['email']})"
                )
            lines.append("")

        if rec["matched_events"]:
            lines.append("#### 📅 Aligned Upcoming Events & Competitions:")
            for ev in rec["matched_events"][:3]:
                reg = ev.get("registration", "Open")
                prizes = f" | 🏆 {ev['prizes']}" if ev.get("prizes") else ""
                lines.append(
                    f"• **{ev['title']}** ({ev['category'].title()})\n"
                    f"  📅 {ev['date']} at {ev.get('time','')} | 📍 {ev.get('venue','')}\n"
                    f"  Format: {ev.get('participation_type','Individual')}{prizes}\n"
                    f"  Registration: {reg}"
                )
            lines.append("")

        if rec.get("action_plan"):
            lines.append("#### 🗺️ Suggested 30-Day Campus Engagement Plan:")
            for step in rec["action_plan"]:
                lines.append(f"- **{step['week']}**: {step['milestone']}\n  {step['guidance']}")

        return "\n".join(lines)
