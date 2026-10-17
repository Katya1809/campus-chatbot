"""Campus agent: intent routing -> LangChain tools -> grounded LLM answer (or offline extractive answer)."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import StructuredTool

from .ingest import categorize
from .retriever import CampusRetriever
from .services import ContactService, EventService, LocationService

SYSTEM = """You are the Campus Assistant for {college}, helping freshers, students and visitors.
Answer ONLY from the CONTEXT below. If the answer is not there, say you don't have it and suggest the
relevant office. Never invent phone numbers, dates, fees or room numbers. Be concise and friendly, use
Indian college terms naturally (CBCS, internals, HOD, T&P cell), and mention the source name when useful.
Reply in the language the student writes in."""

INTENTS = {
    "find_location": ["where", "location", "directions", "route to", "how to reach", "room", "block", "building", "located"],
    "upcoming_events": ["event", "fest", "workshop", "hackathon", "upcoming", "this month", "this week", "schedule"],
    "find_contact": ["contact", "phone", "email", "mail", "number", "hod", "dean", "principal", "faculty", "professor", "who is"],
}
FALLBACK = ("I couldn't find that in the campus information I have. Try rephrasing, or ask the "
            "student help desk / relevant department office.")


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class CampusAgent:
    def __init__(self, retriever: CampusRetriever, data_dir: Path, college: str):
        self.college, self.retriever = college, retriever
        self.locations = LocationService(Path(data_dir) / "locations.csv")
        self.events = EventService(Path(data_dir) / "events.csv")
        self.contacts = ContactService(Path(data_dir) / "contacts.csv")
        self.tools = {
            "find_location": StructuredTool.from_function(
                lambda query: self.locations.describe(query, college), name="find_location",
                description="Find a campus building/facility with floor, timings and a Google Maps link."),
            "upcoming_events": StructuredTool.from_function(
                lambda query="": self.events.describe(query), name="upcoming_events",
                description="List upcoming campus events with venue and registration info."),
            "find_contact": StructuredTool.from_function(
                lambda query: self.contacts.describe(query), name="find_contact",
                description="Look up faculty, department or emergency contact details."),
            "search_handbook": StructuredTool.from_function(
                lambda query: "\n---\n".join(d.page_content for d in self.retriever.search(query, k=4)),
                name="search_handbook", description="Semantic search over handbooks, web pages and CSV data."),
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
                return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.2)
        except Exception:
            pass
        return None  # offline / extractive mode

    @property
    def mode(self) -> str:
        return "LLM" if self.llm else "offline"

    def route(self, query: str) -> list[str]:
        q = query.lower()
        return [t for t, kws in INTENTS.items() if any(k in q for k in kws)]

    def answer(self, query: str, history: list[dict] | None = None) -> dict:
        used = self.route(query)
        context, sources = [], []
        for name in used:
            out = self.tools[name].invoke({"query": query}).strip()
            if out:
                context.append(f"[{name}]\n{out}")
        for d in self.retriever.search(query, k=4):
            context.append(f"[{d.metadata['source']} | updated {d.metadata['updated']}]\n{d.page_content}")
            sources.append(d.metadata["source"])
        used += ["search_handbook"] if sources else []
        category = categorize(query)
        if not context:
            return {"answer": FALLBACK, "category": category, "sources": [], "tools": [], "answered": False}

        if self.llm:
            try:
                msgs = [("system", SYSTEM.format(college=self.college))]
                for m in (history or [])[-6:]:
                    msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
                msgs.append(("human", "CONTEXT:\n" + "\n\n".join(context) + f"\n\nQUESTION: {query}"))
                text = _text(self.llm.invoke(msgs).content).strip()
                return {"answer": text or FALLBACK, "category": category, "sources": sorted(set(sources)),
                        "tools": used, "answered": bool(text)}
            except Exception as e:
                note = f"\n\n_(AI service unavailable: {type(e).__name__}. Showing raw matches.)_"
            else:
                note = ""
        else:
            note = "\n\n_(Offline mode: add an API key for conversational answers.)_"
        raw = "\n\n".join(c[:450] for c in context[:3])
        return {"answer": f"Here's what I found:\n\n{raw}{note}", "category": category,
                "sources": sorted(set(sources)), "tools": used, "answered": True}
