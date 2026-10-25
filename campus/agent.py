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

from .ingest import categorize
from .retriever import CampusRetriever
from .services import ClubService, ContactService, EventService, LocationService
from .tools import (
    make_campus_overview_tool,
    make_clubs_tool,
    make_contact_tool,
    make_document_search_tool,
    make_events_tool,
    make_location_tool,
)

SYSTEM = """You are the Campus Assistant for {college}, helping freshers, students and visitors.
Answer ONLY from the CONTEXT below. If the answer is not there, say you don't have it and suggest the
relevant office. Never invent phone numbers, dates, fees or room numbers. Be concise and friendly, use
Indian college terms naturally (CBCS, internals, HOD, T&P cell), and mention the source name when useful.
When sharing contact info, always include both email AND phone if available.
When sharing event info, always include the date, time, venue and registration details.
Reply in the language the student writes in."""

# Intent -> tool mapping (can activate multiple tools per query)
INTENTS: dict[str, list[str]] = {
    "find_location": [
        "where", "location", "directions", "route to", "how to reach",
        "room", "block", "building", "located", "address", "map", "floor",
    ],
    "upcoming_events": [
        "event", "fest", "workshop", "hackathon", "upcoming", "this month",
        "this week", "schedule", "calendar", "when is", "registration open",
    ],
    "find_contact": [
        "contact", "phone", "email", "mail", "number", "hod", "dean",
        "principal", "faculty", "professor", "who is", "reach out", "helpline",
    ],
    "find_club": [
        "club", "society", "join", "drama", "music", "dance", "coding",
        "robotics", "astronomy", "photography", "literary", "sports team",
        "how to join", "induction",
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

        # LangChain tools — each wraps a service + document search
        self.tools: dict[str, object] = {
            "find_location": make_location_tool(self.locations, college),
            "upcoming_events": make_events_tool(self.events),
            "find_contact": make_contact_tool(self.contacts),
            "find_club": make_clubs_tool(self.clubs),
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
                note = f"\n\n_(AI service unavailable: {type(e).__name__}. Showing raw matches.)_"
        else:
            note = "\n\n_(Offline mode: add an API key for conversational answers.)_"

        raw = "\n\n".join(c[:500] for c in context_parts[:4])
        return {
            "answer": f"Here's what I found:\n\n{raw}{note}",
            "category": category,
            "sources": sorted(set(sources)),
            "tools": used_tools,
            "answered": True,
        }
