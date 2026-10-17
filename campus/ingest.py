"""Multi-source ingestion: PDF, DOCX, MD/TXT, CSV and web pages -> LangChain Documents."""
from __future__ import annotations

import re
import urllib.robotparser as rp
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

CATEGORIES = {
    "admissions": ["admission", "eligibility", "verification", "fee", "scholarship", "tuition"],
    "academics": ["semester", "cbcs", "credit", "exam", "internal", "external", "attendance",
                  "syllabus", "grade", "cgpa", "backlog", "timetable"],
    "facilities": ["library", "hostel", "canteen", "mess", "sports", "gym", "medical", "lab",
                   "auditorium", "parking", "block"],
    "clubs": ["club", "society", "drama", "music", "dance", "nss", "ncc", "committee", "chapter"],
    "events": ["event", "fest", "workshop", "hackathon", "seminar", "competition", "cultural"],
    "placement": ["placement", "internship", "recruiter", "company", "drive", "training"],
    "transport": ["bus", "shuttle", "transport", "route"],
    "emergency": ["emergency", "security", "ambulance", "helpline", "ragging", "grievance"],
    "digital": ["portal", "erp", "lms", "login", "wifi", "online"],
    "contacts": ["contact", "phone", "email", "hod", "dean", "principal", "faculty"],
}
CSV_TYPES = {"events": "events", "contacts": "contacts", "clubs": "clubs", "locations": "facilities"}


def categorize(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    scores = {c: sum(words.count(k) for k in kws) for c, kws in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def _meta(source: str, kind: str, category: str) -> dict:
    return {"source": source, "type": kind, "category": category,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader
    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            m = _meta(f"{path.name} (p.{i})", "handbook", categorize(text))
            docs.append(Document(page_content=text, metadata=m))
    return docs


def load_docx(path: Path) -> list[Document]:
    import docx
    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    return [Document(page_content=text, metadata=_meta(path.name, "handbook", categorize(text)))]


def load_text(path: Path) -> list[Document]:
    # split markdown by headings so each section gets its own category
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n(?=#{1,3} )", raw)
    return [Document(page_content=p.strip(),
                     metadata=_meta(path.name, "handbook", categorize(p)))
            for p in parts if p.strip()]


def load_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path).fillna("")
    kind = CSV_TYPES.get(path.stem)
    docs = []
    for _, row in df.iterrows():
        text = " | ".join(f"{c}: {v}" for c, v in row.items() if str(v).strip())
        cat = kind or categorize(text)
        docs.append(Document(page_content=text, metadata=_meta(path.name, kind or "csv", cat)))
    return docs


def _allowed(url: str) -> bool:
    try:
        u = urlparse(url)
        parser = rp.RobotFileParser(f"{u.scheme}://{u.netloc}/robots.txt")
        parser.read()
        return parser.can_fetch("CampusInfoBot", url)
    except Exception:
        return True  # robots.txt unreachable -> proceed politely


def scrape_url(url: str, timeout: int = 10) -> Document:
    if not _allowed(url):
        raise PermissionError("blocked by robots.txt")
    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "CampusInfoBot/1.0 (student project)"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        t.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = re.sub(r"\n{2,}", "\n", soup.get_text("\n", strip=True))
    m = _meta(url, "web", categorize(text))
    m["title"] = title
    return Document(page_content=text[:20000], metadata=m)


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".md": load_text, ".txt": load_text, ".csv": load_csv}


def load_all(data_dir: Path, urls: list[str]) -> tuple[list[Document], list[str]]:
    """Returns (documents, warnings). One bad source never breaks the rest."""
    docs, warnings = [], []
    for path in sorted(Path(data_dir).glob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        try:
            docs += loader(path)
        except Exception as e:
            warnings.append(f"Could not read {path.name}: {e}")
    for url in urls:
        try:
            docs.append(scrape_url(url))
        except Exception as e:
            warnings.append(f"Could not scrape {url}: {e}")
    return docs, warnings
