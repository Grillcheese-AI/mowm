# tests/test_data_loader.py
"""Tests for RealmDataLoader — domain corpus loading from realm directories."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mowm.data_loader import RealmDataLoader, _DOMAIN_TO_REALMS, _DEFAULT_REALMS_PATH


class TestInit:
    def test_default_path(self):
        loader = RealmDataLoader()
        assert loader.realms_path == _DEFAULT_REALMS_PATH

    def test_custom_path_str(self, tmp_path):
        loader = RealmDataLoader(realms_path=str(tmp_path))
        assert loader.realms_path == tmp_path

    def test_custom_path_pathlib(self, tmp_path):
        loader = RealmDataLoader(realms_path=tmp_path)
        assert loader.realms_path == tmp_path

    def test_nonexistent_path_does_not_raise(self):
        loader = RealmDataLoader(realms_path="/nonexistent/path/to/realms")
        assert not loader.available

    def test_available_true_when_path_exists(self, tmp_path):
        loader = RealmDataLoader(realms_path=tmp_path)
        assert loader.available

    def test_default_chunk_size(self):
        loader = RealmDataLoader()
        assert loader.chunk_size == 10

    def test_custom_chunk_size(self):
        loader = RealmDataLoader(chunk_size=5)
        assert loader.chunk_size == 5


class TestDomainMapping:
    def test_known_domains_present(self):
        for domain in ("physics", "logic", "bio", "music", "identity"):
            assert domain in _DOMAIN_TO_REALMS

    def test_physics_maps_to_realms(self):
        assert "physics" in _DOMAIN_TO_REALMS["physics"]

    def test_identity_maps_to_my_identity(self):
        assert "my_identity" in _DOMAIN_TO_REALMS["identity"]

    def test_temporal_maps_correctly(self):
        assert "temporal_timelines_events" in _DOMAIN_TO_REALMS["temporal"]


class TestLoadWithTempdir:
    def test_load_physics_txt(self, tmp_path):
        physics_dir = tmp_path / "physics"
        physics_dir.mkdir()
        (physics_dir / "laws.txt").write_text("F = ma\nPV = nRT\nE = mc^2\n")
        loader = RealmDataLoader(realms_path=tmp_path, chunk_size=2)
        records = loader.load_domain("physics")
        assert len(records) >= 1
        assert isinstance(records[0], str)

    def test_load_emotions_jsonl(self, tmp_path):
        identity_dir = tmp_path / "my_identity"
        identity_dir.mkdir()
        (identity_dir / "emotions.jsonl").write_text(
            '{"emotion": "joy", "intensity": 0.8}\n'
            '{"emotion": "trust", "intensity": 0.6}\n'
        )
        loader = RealmDataLoader(realms_path=tmp_path)
        records = loader.load_emotions()
        assert len(records) == 2
        assert records[0]["emotion"] == "joy"

    def test_load_temporal_events_json(self, tmp_path):
        events_dir = tmp_path / "temporal_timelines_events"
        events_dir.mkdir()
        events = [{"ts": 1.0, "event": "alpha"}, {"ts": 2.0, "event": "beta"}]
        (events_dir / "timeline.json").write_text(json.dumps(events))
        loader = RealmDataLoader(realms_path=tmp_path)
        records = loader.load_temporal_events()
        assert len(records) == 2
        assert records[0]["event"] == "alpha"

    def test_load_json_object(self, tmp_path):
        bio_dir = tmp_path / "bio"
        bio_dir.mkdir()
        (bio_dir / "data.json").write_text(json.dumps({"key": "value"}))
        loader = RealmDataLoader(realms_path=tmp_path)
        records = loader.load_domain("bio")
        assert len(records) == 1
        assert records[0]["key"] == "value"

    def test_unknown_domain_returns_empty(self, tmp_path):
        loader = RealmDataLoader(realms_path=tmp_path)
        records = loader.load_domain("xyzzy_unknown_domain")
        assert records == []

    def test_max_samples(self, tmp_path):
        physics_dir = tmp_path / "physics"
        physics_dir.mkdir()
        lines = "\n".join(f"line {i}" for i in range(50))
        (physics_dir / "data.txt").write_text(lines)
        loader = RealmDataLoader(realms_path=tmp_path, chunk_size=1)
        records = loader.load_domain("physics", max_samples=3)
        assert len(records) == 3

    def test_chunking(self, tmp_path):
        physics_dir = tmp_path / "physics"
        physics_dir.mkdir()
        (physics_dir / "data.txt").write_text("\n".join(str(i) for i in range(20)))
        loader = RealmDataLoader(realms_path=tmp_path, chunk_size=5)
        records = loader.load_domain("physics")
        # 20 lines / 5 per chunk = 4 chunks
        assert len(records) == 4
        assert "\n" in records[0]  # chunk contains multiple lines

    def test_partial_last_chunk(self, tmp_path):
        physics_dir = tmp_path / "physics"
        physics_dir.mkdir()
        (physics_dir / "data.txt").write_text("\n".join(str(i) for i in range(7)))
        loader = RealmDataLoader(realms_path=tmp_path, chunk_size=5)
        records = loader.load_domain("physics")
        # 5-line chunk + 2-line trailing chunk
        assert len(records) == 2


class TestMissingPaths:
    def test_load_domain_missing_path_returns_empty(self):
        loader = RealmDataLoader(realms_path="/nonexistent/xyz")
        assert loader.load_domain("physics") == []

    def test_load_emotions_missing_path_returns_empty(self):
        loader = RealmDataLoader(realms_path="/nonexistent/xyz")
        assert loader.load_emotions() == []

    def test_load_temporal_missing_path_returns_empty(self):
        loader = RealmDataLoader(realms_path="/nonexistent/xyz")
        assert loader.load_temporal_events() == []

    def test_load_emotions_missing_file_returns_empty(self, tmp_path):
        # realms_path exists but my_identity/emotions.jsonl does not
        loader = RealmDataLoader(realms_path=tmp_path)
        assert loader.load_emotions() == []

    def test_load_temporal_missing_dir_returns_empty(self, tmp_path):
        loader = RealmDataLoader(realms_path=tmp_path)
        assert loader.load_temporal_events() == []
