# mowm/qa/factual_lookup.py
"""FactualLookup — search historical facts and events for when/where/who questions.

Loads structured Q&A data from temporal/historical datasets and performs
keyword-based search to answer factual questions about dates, events,
people, and places.

Does NOT use axioms — this is pure factual recall from data, not reasoning.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

# Key historical dates that should always be available
# (the corpus narratives often omit exact years)
_KEY_DATES: dict[str, tuple[int, str]] = {
    "rome fall": (476, "The Western Roman Empire fell in 476 AD when Odoacer deposed Emperor Romulus Augustulus. The earlier Sack of Rome by the Visigoths under Alaric occurred in 410 AD."),
    "rome sack": (410, "Rome was sacked by the Visigoths under Alaric I in 410 AD, the first time in 800 years the city had been taken."),
    "world war 1": (1914, "World War I began on July 28, 1914 and ended November 11, 1918."),
    "world war 2": (1939, "World War II began on September 1, 1939 and ended September 2, 1945."),
    "french revolution": (1789, "The French Revolution began in 1789 with the storming of the Bastille on July 14."),
    "american revolution": (1776, "The American Revolution culminated in the Declaration of Independence on July 4, 1776."),
    "moon landing": (1969, "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong was the first human to walk on the lunar surface."),
    "berlin wall": (1989, "The Berlin Wall fell on November 9, 1989, leading to German reunification in 1990."),
    "renaissance": (1400, "The Renaissance began in Italy around 1400 and spread across Europe through the 16th century."),
    "printing press": (1440, "Johannes Gutenberg invented the movable-type printing press around 1440 in Mainz, Germany."),
    "industrial revolution": (1760, "The Industrial Revolution began in Britain around 1760, transforming manufacturing, agriculture, and transportation."),
    "internet": (1969, "The precursor to the Internet, ARPANET, was created in 1969. The World Wide Web was invented by Tim Berners-Lee in 1989."),
    "electricity": (1879, "Thomas Edison demonstrated the practical incandescent light bulb in 1879. Nikola Tesla developed alternating current systems in the 1880s."),
    "roman empire": (27, "The Roman Empire began in 27 BC when Octavian became Augustus. It fell in 476 AD (West) and 1453 AD (East/Byzantine)."),
    "julius caesar": (44, "Julius Caesar was assassinated on March 15, 44 BC (the Ides of March)."),
    "alexander great": (323, "Alexander the Great died in 323 BC in Babylon at age 32, having conquered an empire from Greece to India."),
}

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "but", "or", "not", "so", "yet", "if", "when", "what",
    "which", "who", "whom", "how", "why", "where", "there",
    "this", "that", "these", "those", "it", "its", "they",
    "he", "she", "we", "you", "me", "him", "her", "us",
    "about", "than", "very", "just", "also",
})


@dataclass
class FactEntry:
    """A single factual entry."""
    question: str
    answer: str
    year: int | None = None
    category: str = ""
    keywords: set[str] = field(default_factory=set)


class FactualLookup:
    """Search factual knowledge from historical datasets.

    Loads Q&A pairs from JSON/JSONL files. Answers factual
    questions (when, where, who) by keyword matching.

    Args:
        data_paths: List of directories or files to load.
    """

    def __init__(self, data_paths: list[str] | None = None) -> None:
        self._facts: list[FactEntry] = []
        if data_paths:
            for path in data_paths:
                self._load_path(path)

    @property
    def count(self) -> int:
        return len(self._facts)

    def _load_path(self, path: str) -> None:
        """Load facts from a file or directory."""
        if not os.path.exists(path):
            return

        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if fpath.endswith(".json"):
                    self._load_json(fpath)
                elif fpath.endswith(".jsonl"):
                    self._load_jsonl(fpath)
        elif path.endswith(".json"):
            self._load_json(path)
        elif path.endswith(".jsonl"):
            self._load_jsonl(path)

    def _load_json(self, path: str) -> None:
        """Load a JSON file (array of records or nested training_data)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Handle nested format: {"training_data": [...]}
        if isinstance(data, dict) and "training_data" in data:
            records = data["training_data"]
        elif isinstance(data, list):
            records = data
        else:
            return

        for rec in records:
            if not isinstance(rec, dict):
                continue
            q = rec.get("question", "")
            a = rec.get("answer", rec.get("text", rec.get("event", "")))
            if not q and not a:
                continue

            year = self._extract_year(q) or self._extract_year(a)
            keywords = self._extract_keywords(f"{q} {a}")

            self._facts.append(FactEntry(
                question=str(q)[:500],
                answer=str(a)[:1000],
                year=year,
                category=rec.get("category", ""),
                keywords=keywords,
            ))

    def _load_jsonl(self, path: str) -> None:
        """Load a JSONL file line by line."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue

                    q = rec.get("question", "")
                    a = rec.get("answer", rec.get("text", ""))
                    if not q and not a:
                        continue

                    year = self._extract_year(q) or self._extract_year(a)
                    keywords = self._extract_keywords(f"{q} {a}")

                    self._facts.append(FactEntry(
                        question=str(q)[:500],
                        answer=str(a)[:1000],
                        year=year,
                        category=rec.get("category", ""),
                        keywords=keywords,
                    ))
        except OSError:
            pass

    def search(self, query: str, top_n: int = 3) -> list[FactEntry]:
        """Search facts by keyword overlap with query.

        Checks key historical dates first, then searches the corpus.

        Args:
            query: Natural language question.
            top_n: Number of results to return.

        Returns:
            List of FactEntry sorted by relevance.
        """
        # Check key dates first
        key_result = self._check_key_dates(query)
        if key_result:
            # Prepend key date, then add corpus results
            corpus = self._search_corpus(query, top_n=top_n - 1)
            return [key_result] + corpus

        return self._search_corpus(query, top_n=top_n)

    def _check_key_dates(self, query: str) -> FactEntry | None:
        """Check if query matches a key historical date."""
        q_lower = query.lower()
        best_match: tuple[int, str, int, str] | None = None
        best_overlap = 0

        for key, (year, answer) in _KEY_DATES.items():
            key_words = set(key.split())
            q_words = set(q_lower.split())
            overlap = len(key_words & q_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (year, answer, overlap, key)

        if best_match and best_overlap >= 1:
            year, answer, _, key = best_match
            return FactEntry(
                question=f"Key date: {key}",
                answer=answer,
                year=year,
                category="key_date",
                keywords=self._extract_keywords(answer),
            )
        return None

    def _search_corpus(
        self, query: str, top_n: int = 3,
    ) -> list[FactEntry]:
        """Search corpus facts by keyword overlap."""
        if not self._facts:
            return []

        q_keywords = self._extract_keywords(query)
        if not q_keywords:
            return []

        scored: list[tuple[float, FactEntry]] = []
        for fact in self._facts:
            overlap = len(q_keywords & fact.keywords)
            if overlap == 0:
                continue
            # Normalize by query length
            score = overlap / len(q_keywords)
            scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_n]]

    def search_by_year(self, year: int) -> list[FactEntry]:
        """Find all facts associated with a specific year."""
        return [f for f in self._facts if f.year == year]

    @staticmethod
    def _extract_year(text: str) -> int | None:
        """Extract a 4-digit year from text."""
        match = re.search(r"\b(\d{3,4})\b", str(text))
        if match:
            year = int(match.group(1))
            if 100 <= year <= 2100:
                return year
        return None

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract non-stopword tokens from text."""
        return {
            t.lower() for t in _TOKEN_RE.findall(str(text))
            if t.lower() not in _STOPWORDS and len(t) > 2
        }
