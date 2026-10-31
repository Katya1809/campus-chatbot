"""SQLite query history, feedback and analytics (no personal data stored)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class Store:
    def __init__(self, path: str | Path = "campus.db"):
        self.path = str(path)
        with self._c() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS history(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP,
                query TEXT, category TEXT, answered INTEGER, sources TEXT, feedback INTEGER)""")

    def _c(self):
        return sqlite3.connect(self.path)

    def log(self, query: str, category: str, answered: bool, sources: str) -> int:
        with self._c() as c:
            return c.execute("INSERT INTO history(query,category,answered,sources) VALUES(?,?,?,?)",
                             (query, category, int(answered), sources)).lastrowid

    def feedback(self, row_id: int, value: int | None):
        with self._c() as c:
            c.execute("UPDATE history SET feedback=? WHERE id=?", (value, row_id))

    def frame(self) -> pd.DataFrame:
        with self._c() as c:
            return pd.read_sql_query("SELECT * FROM history", c)
