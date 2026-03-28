# mowm/data_loader.py
"""RealmDataLoader — load domain text corpora from the realms directory.

Maps MoWM domain names to on-disk realm directories and provides
file-format-aware loading for .txt (chunked), .jsonl (line-by-line),
and .json (array or object) files. Gracefully handles a missing realms path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# Maps MoWM domain name -> list of realm sub-directory names to scan.
_DOMAIN_TO_REALMS: dict[str, list[str]] = {
    "physics": ["physics", "science"],
    "logic": ["logic", "mathematics"],
    "economics": ["economics", "finance"],
    "social": ["social", "sociology"],
    "linguistics": ["linguistics", "language"],
    "causal": ["causality", "causal_reasoning"],
    "atomic": ["atomic", "chemistry"],
    "bio": ["biology", "bio"],
    "music": ["music", "audio"],
    "identity": ["my_identity", "self"],
    "colors": ["colors", "visual"],
    "language": ["language", "linguistics"],
    "historical": ["history", "historical"],
    "ensemble": ["ensemble", "mixed"],
    "emotions": ["my_identity"],
    "temporal": ["temporal_timelines_events"],
}

_DEFAULT_REALMS_PATH = Path("E:/RAW_TEXTS/realms")


class RealmDataLoader:
    """Load domain text corpora from a realms directory tree.

    Each domain is mapped to one or more realm sub-directories.  Files
    within those directories are loaded according to their extension:

    - ``.txt``   — split into chunks of *chunk_size* lines each.
    - ``.jsonl``  — one JSON object per line, yielded individually.
    - ``.json``   — decoded as an array (each element) or a single object.

    Args:
        realms_path: Root path that contains realm sub-directories.
                     Defaults to ``E:/RAW_TEXTS/realms``.
        chunk_size: Number of lines per text chunk when loading ``.txt`` files.
    """

    def __init__(
        self,
        realms_path: str | Path | None = None,
        chunk_size: int = 10,
    ) -> None:
        self.realms_path = Path(realms_path) if realms_path is not None else _DEFAULT_REALMS_PATH
        self.chunk_size = chunk_size

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Return True if *realms_path* exists on disk."""
        return self.realms_path.exists()

    def domain_dirs(self, domain: str) -> list[Path]:
        """Return the list of existing realm directories for *domain*.

        Args:
            domain: MoWM domain name (e.g. ``"physics"``).

        Returns:
            List of :class:`pathlib.Path` objects that exist on disk.
        """
        realm_names = _DOMAIN_TO_REALMS.get(domain, [domain])
        dirs: list[Path] = []
        for name in realm_names:
            candidate = self.realms_path / name
            if candidate.exists():
                dirs.append(candidate)
        return dirs

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_domain(
        self,
        domain: str,
        max_samples: int | None = None,
    ) -> list[Any]:
        """Load all records for *domain* from its mapped realm directories.

        Scans every file inside the mapped directories (non-recursively).
        ``.txt`` files are split into chunks; ``.jsonl`` yields one record per
        line; ``.json`` files yield each element of a top-level array (or the
        object itself when the top-level value is not a list).

        Args:
            domain: MoWM domain name (e.g. ``"physics"``, ``"identity"``).
            max_samples: Optional cap on the number of records returned.

        Returns:
            List of records (strings for text chunks, dicts/lists for JSON).
        """
        if not self.available:
            return []

        records: list[Any] = []
        for realm_dir in self.domain_dirs(domain):
            for path in sorted(realm_dir.iterdir()):
                if not path.is_file():
                    continue
                for record in self._load_file(path):
                    records.append(record)
                    if max_samples is not None and len(records) >= max_samples:
                        return records
        return records

    def load_emotions(self) -> list[Any]:
        """Load records from ``my_identity/emotions.jsonl``.

        Returns:
            List of dicts (one per line in the JSONL file), or ``[]`` when
            the path does not exist.
        """
        if not self.available:
            return []
        target = self.realms_path / "my_identity" / "emotions.jsonl"
        if not target.exists():
            return []
        return list(self._load_file(target))

    def load_temporal_events(self) -> list[Any]:
        """Load all ``.json`` files from ``temporal_timelines_events/``.

        Returns:
            Flat list of all records from every JSON file in the directory,
            or ``[]`` when the path does not exist.
        """
        if not self.available:
            return []
        target_dir = self.realms_path / "temporal_timelines_events"
        if not target_dir.exists():
            return []
        records: list[Any] = []
        for path in sorted(target_dir.iterdir()):
            if path.is_file() and path.suffix == ".json":
                records.extend(self._load_file(path))
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> Iterator[Any]:
        """Yield all records from *path* according to its file extension."""
        suffix = path.suffix.lower()
        if suffix == ".txt":
            yield from self._load_txt(path)
        elif suffix == ".jsonl":
            yield from self._load_jsonl(path)
        elif suffix == ".json":
            yield from self._load_json(path)
        # Unknown extensions are silently skipped.

    def _load_txt(self, path: Path) -> Iterator[str]:
        """Yield non-overlapping chunks of *chunk_size* lines from a text file."""
        lines: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                lines.append(raw_line.rstrip("\n"))
                if len(lines) >= self.chunk_size:
                    yield "\n".join(lines)
                    lines = []
        if lines:
            yield "\n".join(lines)

    def _load_jsonl(self, path: Path) -> Iterator[Any]:
        """Yield one decoded JSON value per non-empty line."""
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    pass  # Skip malformed lines

    def _load_json(self, path: Path) -> Iterator[Any]:
        """Yield each element of a top-level JSON array, or the object itself."""
        with path.open(encoding="utf-8", errors="replace") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                return
        if isinstance(data, list):
            yield from data
        else:
            yield data
