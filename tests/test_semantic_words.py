"""FastWordEncoder — the distilled word-table encoder (model-free serving).

The teacher is injected as ``embed_fn`` so these tests never import
sentence-transformers or torch; a seeded fake stands in for it.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from mowm.encoding import FastWordEncoder, signal_code, split_words

K, L = 8, 16          # small dims for tests; production uses 80 x 128
DIM = 24              # fake teacher dim (k*l = 128 >= 24 satisfies the QR guard)


def _fake_embed(strings: list[str]) -> np.ndarray:
    """Deterministic stand-in teacher: seeded vector per string, with related
    words made genuinely similar so semantic routing is testable."""
    base = {}
    for s in strings:
        seed = abs(hash_stable(s)) % (2**31)
        base[s] = np.random.default_rng(seed).standard_normal(DIM)
    # make 'force' and 'acceleration' semantic neighbours
    if "force" in base and "acceleration" in base:
        base["acceleration"] = 0.9 * base["force"] + 0.1 * base["acceleration"]
    return np.stack([base[s] for s in strings]).astype(np.float32)


def hash_stable(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def _encoder(words=("force", "acceleration", "energy", "banana"), idf=None):
    words = list(words)
    idf = idf or {w: 1.0 for w in words}
    return FastWordEncoder.build(words, _fake_embed, idf, default_idf=2.0, k=K, l=L)


def test_build_is_deterministic():
    a, b = _encoder(), _encoder()
    for w in a._vec:
        assert np.array_equal(a._vec[w], b._vec[w])


def test_encode_shape_dtype_and_determinism():
    enc = _encoder()
    v = enc.encode("Force and energy!")
    assert v.shape == (K, L) and v.dtype == np.float32
    assert np.array_equal(v, enc.encode("Force and energy!"))
    assert np.isfinite(v).all()


def test_semantic_neighbours_score_higher_than_unrelated():
    # Production form: anisotropy stats fit on EXTERNAL vocabulary. (Fitting
    # PC1 on a 4-word table strips the very direction the related pair shares
    # -- the degenerate-tiny-table case of the in-domain-stats effect the
    # CubbyLLM OOD check measured; external stats are the recommended shape.)
    from mowm.encoding.semantic_words import _unit_rows, fit_anisotropy

    ext = _unit_rows(_fake_embed([f"extword{i}" for i in range(50)]))
    enc = FastWordEncoder.build(
        ["force", "acceleration", "energy", "banana"], _fake_embed,
        {w: 1.0 for w in ("force", "acceleration", "energy", "banana")},
        default_idf=2.0, k=K, l=L, anisotropy=fit_anisotropy(ext),
    )

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    q = enc.encode("force")
    assert cos(q, enc.encode("acceleration")) > cos(q, enc.encode("banana"))


def test_unseen_word_falls_back_to_signal_code_deterministically():
    enc = _encoder()
    v1 = enc.encode("zzgibberish")
    v2 = enc.encode("zzgibberish")
    assert np.array_equal(v1, v2)
    assert np.linalg.norm(v1) > 0
    # and the fallback IS the seeded signal code (weighted by default_idf)
    expected = enc.default_idf * signal_code("zzgibberish", K, L)
    assert np.allclose(v1.ravel(), expected, atol=1e-6)


def test_idf_weighting_steers_the_bundle():
    heavy = _encoder(idf={"force": 10.0, "energy": 0.1, "acceleration": 1.0, "banana": 1.0})

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    v = heavy.encode("force energy")
    assert cos(v, heavy._vec["force"]) > cos(v, heavy._vec["energy"])


def test_npz_roundtrip_preserves_encoding(tmp_path):
    enc = _encoder()
    p = tmp_path / "table.npz"
    enc.save_npz(p)
    loaded = FastWordEncoder.from_npz(p)
    for text in ("force acceleration", "banana zznew"):
        assert np.array_equal(enc.encode(text), loaded.encode(text))
    assert loaded.k == K and loaded.l == L


def test_split_words_matches_the_validated_unit():
    assert split_words("Newton's Second Law: F = m*a!") == {
        "newton", "second", "law"
    }


def test_serving_import_is_model_free():
    # the module under test must not have dragged the teacher stack in
    assert "sentence_transformers" not in sys.modules


def test_build_rejects_impossible_dims():
    with pytest.raises(ValueError):
        FastWordEncoder.build(["a1", "b1"], _fake_embed, {}, 1.0, k=2, l=4)  # k*l < DIM
