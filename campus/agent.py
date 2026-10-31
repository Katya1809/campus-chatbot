"""Campus agent: intent routing -> LangChain tools -> grounded LLM answer (or offline extractive answer).

Tools integrated:
  1. find_location     — campus map & building info with Google Maps links
  2. upcoming_events   — event calendar processing with ICS support
  3. find_contact      — department and faculty contact directory
  4. find_club         — clubs and societies directory
  5. campus_overview   — comprehensive campus info synthesis
  6. search_documents  — semantic/TF-IDF search over all ingested sources
                          (PDF, DOCX, MD, CSV, ICS, web pages)
"""
from __future__ import annotations

import os
from pathlib import Path

from .engagement import EngagementService
from .ingest import categorize
from .recommender import ActivityRecommender
from .retriever import CampusRetriever
from .services import ClubService, ContactService, EventService, LocationService
from .tools import (
    make_campus_overview_tool,
    make_club_induction_tool,
    make_clubs_tool,
    make_contact_tool,
    make_document_search_tool,
    make_engagement_tool,
    make_event_registration_tool,
    make_events_tool,
    make_location_tool,
    make_participation_guidance_tool,
    make_recommendation_tool,
)

SYSTEM = """You are the Campus Assistant for {college}, helping freshers, students and visitors.
Answer ONLY from the CONTEXT below. If the answer is not there, say you don't have it and suggest the
relevant office. Never invent phone numbers, dates, fees or room numbers. Be concise and friendly, use
Indian college terms naturally (CBCS, internals, HOD, T&P cell), and mention the source name when useful.
When sharing contact info, always include both email AND phone if available.
When sharing event info, always include the date, time, venue, format, and registration details.
When recommending clubs/activities, provide tailored suggestions with rationale and next steps.
Reply in the language the student writes in."""

# Intent -> tool mapping (can activate multiple tools per query)
INTENTS: dict[str, list[str]] = {
    "find_location": [
        "where", "location", "directions", "route to", "how to reach",
        "room", "block", "building", "located", "address", "map", "floor",
    ],
    "upcoming_events": [
        "event", "fest", "workshop", "hackathon", "upcoming", "this month",
        "this week", "schedule", "calendar", "when is",
    ],
    "event_registration_process": [
        "register", "registration", "how to register", "portal", "deadline",
        "team size", "entry fee", "sign up", "prizes", "prize money",
    ],
    "event_participation_guidance": [
        "participate", "how to take part", "guidance", "rules", "od",
        "on duty", "leave", "guidelines", "prepare", "attire",
    ],
    "find_contact": [
        "contact", "phone", "email", "mail", "number", "hod", "dean",
        "principal", "faculty", "professor", "who is", "reach out", "helpline",
    ],
    "find_club": [
        "club", "society", "drama", "music", "dance", "coding",
        "robotics", "astronomy", "photography", "literary", "sports team",
        "sports council", "fine arts", "cinewave", "badminton", "cricket",
        "football", "chess", "volleyball", "quizzing",
    ],
    "club_induction_guide": [
        "how to join", "induction", "auditions", "audition", "tryout", "selection",
        "interview", "prerequisite", "eligibility to join",
    ],
    "recommend_activities": [
        "recommend", "recommendation", "suggest", "which club", "what should i join",
        "interests", "interest", "beginner", "hobby", "hobbies", "what to join",
        "explore activities", "passion",
    ],
    "campus_engagement_suggestions": [
        "engagement", "roadmap", "general championship", "gc", "points",
        "eaa", "nso", "nss", "ncc", "yoga", "leadership", "organizing committee",
        "fest team", "volunteer", "volunteering", "fresher journey",
    ],
    "campus_overview": [
        "overview", "about campus", "what is there", "facilities available",
        "campus life", "general", "everything", "what all",
    ],
}

FALLBACK = (
    "I couldn't find that in the campus information I have. "
    "Try rephrasing, or ask the student help desk / relevant department office."
)


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class CampusAgent:
    def __init__(self, retriever: CampusRetriever, data_dir: Path, college: str):
        self.college = college
        self.retriever = retriever

        # Core services
        self.locations = LocationService(Path(data_dir) / "locations.csv")
        self.events = EventService(Path(data_dir) / "events.csv")
        self.contacts = ContactService(Path(data_dir) / "contacts.csv")
        self.clubs = ClubService(Path(data_dir) / "clubs.csv")
        self.recommender = ActivityRecommender(self.clubs, self.events)
        self.engagement = EngagementService()

        # LangChain tools — each wraps a service + document search
        self.tools: dict[str, object] = {
            "find_location": make_location_tool(self.locations, college),
            "upcoming_events": make_events_tool(self.events),
            "event_registration_process": make_event_registration_tool(self.events),
            "event_participation_guidance": make_participation_guidance_tool(self.events),
            "find_contact": make_contact_tool(self.contacts),
            "find_club": make_clubs_tool(self.clubs),
            "club_induction_guide": make_club_induction_tool(self.clubs),
            "recommend_activities": make_recommendation_tool(self.recommender),
            "campus_engagement_suggestions": make_engagement_tool(self.engagement),
            "campus_overview": make_campus_overview_tool(
                self.contacts, self.events, self.locations, self.clubs, college
            ),
            "search_documents": make_document_search_tool(self.retriever),
        }

        self.llm = self._make_llm()

    @staticmethod
    def _make_llm():
        try:
            provider = os.getenv("LLM_PROVIDER", "gemini").lower()
            if provider == "openai" and os.getenv("OPENAI_API_KEY"):
                # pyrefly: ignore[missing-import]
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
            if os.getenv("GOOGLE_API_KEY"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.2
                )
        except Exception:
            pass
        return None  # offline / extractive mode

    @property
    def mode(self) -> str:
        return "LLM" if self.llm else "offline"

    def route(self, query: str) -> list[str]:
        """Return names of tools relevant to this query (keyword-based)."""
        q = query.lower()
        matched = [tool for tool, kws in INTENTS.items() if any(k in q for k in kws)]
        # Always include document search for grounded context
        if "search_documents" not in matched:
            matched.append("search_documents")
        return matched

    def answer(self, query: str, history: list[dict] | None = None) -> dict:
        used_tools = self.route(query)
        tool_results: list[tuple[str, str]] = []
        doc_snippets: list[str] = []
        context_parts: list[str] = []
        sources: list[str] = []

        # Run structured service tools
        for name in used_tools:
            if name == "search_documents":
                continue  # handled separately below
            tool = self.tools.get(name)
            if tool is None:
                continue
            try:
                out = tool.invoke({"query": query}).strip()
                if out:
                    tool_results.append((name, out))
                    context_parts.append(f"[{name}]\n{out}")
            except Exception:
                pass

        # Run document retrieval (PDF, web, ICS, CSV, MD, DOCX)
        try:
            hits = self.retriever.search(query, k=5)
            for d in hits:
                context_parts.append(
                    f"[{d.metadata['source']} | {d.metadata['type']} | updated {d.metadata['updated']}]\n"
                    f"{d.page_content}"
                )
                doc_snippets.append(d.page_content.strip())
                sources.append(d.metadata["source"])
        except Exception:
            pass

        category = categorize(query)

        if not context_parts:
            return {
                "answer": FALLBACK,
                "category": category,
                "sources": [],
                "tools": used_tools,
                "answered": False,
            }

        context_str = "\n\n".join(context_parts)

        if self.llm:
            try:
                msgs = [("system", SYSTEM.format(college=self.college))]
                for m in (history or [])[-6:]:
                    msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
                msgs.append(("human", f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"))
                text = _text(self.llm.invoke(msgs).content).strip()
                return {
                    "answer": text or FALLBACK,
                    "category": category,
                    "sources": sorted(set(sources)),
                    "tools": used_tools,
                    "answered": bool(text),
                }
            except Exception as e:
                note = f"\n\n_(Note: LLM provider unavailable ({type(e).__name__}). Showing verified campus database results.)_"
        else:
            note = ""

        # High-precision offline / direct response (no character truncation)
        ans_blocks: list[str] = []
        if tool_results:
            for _, res in tool_results:
                ans_blocks.append(res)
            # If only 1 tool ran and we have supplementary handbook documents, append top match
            if len(tool_results) == 1 and doc_snippets:
                top_doc = doc_snippets[0]
                if top_doc and top_doc[:100] not in ans_blocks[0]:
                    ans_blocks.append(f"#### 📖 Relevant Handbook & Web Source Excerpt\n{top_doc}")
        elif doc_snippets:
            for snippet in doc_snippets[:3]:
                ans_blocks.append(snippet)
        else:
            ans_blocks.append(FALLBACK)

        full_answer = "\n\n---\n\n".join(ans_blocks)
        if note:
            full_answer += note

        return {
            "answer": full_answer,
            "category": category,
            "sources": sorted(set(sources)),
            "tools": used_tools,
            "answered": True,
        }
