"""semantic_words — fast word-table encoder for challenge and axiom text.

Serving is model-free: text -> words -> table lookup -> IDF-weighted bundle of
(k, l) block codes, ~25 us/challenge in the validation screen. The
sentence-transformer TEACHER runs once, offline, inside
``FastWordEncoder.build`` (or any injected ``embed_fn``); it is imported
lazily there and never at serve time — this module imports only numpy.

Validated in CubbyLLM (``validation/logs/exp_f2m2_m2v.json`` and
``exp_f2m2_m2v_ood.json``): routing macro 0.3407 with in-domain statistics /
0.3079 with out-of-domain IDF+PC1 statistics, against 0.3580 for the live
teacher at ~350x the serve-time cost. Recipe: unit teacher vectors per WORD,
minus the anisotropy direction (mean + top principal component), projected
through an orthonormal map into (k, l) block space (cosine-exact), then
0.5/0.5-blended with each word's seeded one-hot SIGNAL code so exact surface
overlap keeps contributing. Unseen words fall back to their signal code, so
out-of-table text still routes on surface identity.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def split_words(text: str) -> set[str]:
    """Lowercased word tokens of length >= 2 (the validated serving unit)."""
    return {w for w in _WORD_SPLIT.split(text.lower()) if len(w) >= 2}


def signal_code(word: str, k: int, l: int) -> np.ndarray:  # noqa: E741
    """Seeded one-hot-per-block code for a word: unit norm, flat (k*l,).

    Seeding matches the CubbyLLM screen (BLAKE2b over the namespaced gram
    ``"w:<word>"``), so tables and screens reproduce each other bit-for-bit.
    """
    digest = hashlib.blake2b(f"w:{word}".encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**63)
    idx = np.random.default_rng(seed).integers(0, l, size=k)
    v = np.zeros(k * l, dtype=np.float32)
    v[np.arange(k) * l + idx] = 1.0 / np.sqrt(k)
    return v


def orthonormal_projection(latent_dim: int, k: int, l: int, seed: int = 0) -> np.ndarray:  # noqa: E741
    """(k*l, latent_dim) with orthonormal columns — cosine-EXACT embedding of
    the teacher space into block space ((Px)·(Py) = x·y). Same construction
    and default seed as the validation screen's ``make_projection``."""
    if k * l < latent_dim:
        raise ValueError(f"need k*l={k * l} >= latent_dim={latent_dim}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((k * l, latent_dim)))
    return Q.astype(np.float32)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_anisotropy(E: np.ndarray, n_pc: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fit the shared-direction statistics (mean + top PCs) of an embedding
    table. Fit on broad external vocabulary when available — the CubbyLLM
    OOD check showed in-domain statistics inflate routing by ~0.03 macro."""
    mu = E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
    return mu, Vt[:n_pc]


def remove_anisotropy(E: np.ndarray, mu: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """All-but-the-top: subtract the mean and the top PC(s), re-unit rows."""
    Ec = E - mu
    for v in pcs:
        Ec = Ec - np.outer(Ec @ v, v)
    return _unit_rows(Ec)


def default_teacher_embed(strings: list[str]) -> np.ndarray:
    """The offline teacher (lazy import — never reached at serve time)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(model.encode(strings, batch_size=256, show_progress_bar=False),
                      dtype=np.float32)


class FastWordEncoder:
    """Serve-time encoder over a distilled word table. numpy-only.

    ``encode(text)`` -> (k, l) float32 block code: IDF-weighted bundle of each
    word's table vector, with seeded signal-code fallback for unseen words.
    """

    def __init__(
        self,
        words: list[str],
        vectors: np.ndarray,            # (n, k*l) float32, unit rows
        idf: dict[str, float],
        default_idf: float,
        k: int,
        l: int,                          # noqa: E741
        meta: dict | None = None,
    ) -> None:
        self.k, self.l = int(k), int(l)
        self._vec = {w: np.asarray(vectors[i], dtype=np.float32)
                     for i, w in enumerate(words)}
        self._idf = dict(idf)
        self.default_idf = float(default_idf)
        self.meta = dict(meta or {})
        self._sig_cache: dict[str, np.ndarray] = {}

    # -- offline construction --------------------------------------------------

    @classmethod
    def build(
        cls,
        words: list[str],
        embed_fn,
        idf: dict[str, float],
        default_idf: float,
        *,
        k: int = 80,
        l: int = 128,                    # noqa: E741
        proj_seed: int = 0,
        anisotropy: tuple[np.ndarray, np.ndarray] | None = None,
        signal_blend: float = 0.5,
        meta: dict | None = None,
    ) -> "FastWordEncoder":
        """Distill ``embed_fn`` (the teacher; runs HERE, once) into a table.

        ``anisotropy`` is (mu, pcs) fit externally via ``fit_anisotropy``;
        None fits it on this table's own vectors (the in-domain form — fine
        for small tables, inflates stats slightly per the OOD check).
        """
        words = sorted(set(words))
        E = _unit_rows(np.asarray(embed_fn(list(words)), dtype=np.float32))
        if anisotropy is None:
            anisotropy = fit_anisotropy(E)
        E_pc = remove_anisotropy(E, *anisotropy)
        P = orthonormal_projection(E.shape[1], k, l, seed=proj_seed)
        proj = _unit_rows(E_pc @ P.T)
        vectors = np.stack([
            (1.0 - signal_blend) * proj[i] + signal_blend * signal_code(w, k, l)
            for i, w in enumerate(words)
        ]).astype(np.float32)
        info = {"teacher_dim": int(E.shape[1]), "proj_seed": proj_seed,
                "signal_blend": signal_blend, "n_words": len(words)}
        info.update(meta or {})
        return cls(words, vectors, idf, default_idf, k, l, meta=info)

    # -- serving ---------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """(k, l) float32 block code for ``text``. Lookup + adds; no model."""
        acc = np.zeros(self.k * self.l, dtype=np.float64)
        for w in split_words(text):
            v = self._vec.get(w)
            if v is None:
                v = self._sig_cache.get(w)
                if v is None:
                    v = signal_code(w, self.k, self.l)
                    self._sig_cache[w] = v
            acc += self._idf.get(w, self.default_idf) * v.astype(np.float64)
        return acc.reshape(self.k, self.l).astype(np.float32)

    # -- persistence -----------------------------------------------------------

    def save_npz(self, path, dtype=np.float32) -> None:
        """``dtype=np.float16`` halves a production-size table on disk;
        ``from_npz`` casts back to float32 (cosine impact is ~1e-3)."""
        words = sorted(self._vec)
        np.savez_compressed(
            path,
            words=np.array(words),
            vectors=np.stack([self._vec[w] for w in words]).astype(dtype),
            idf_words=np.array(sorted(self._idf)),
            idf_values=np.array([self._idf[w] for w in sorted(self._idf)],
                                dtype=np.float32),
            default_idf=np.array(self.default_idf, dtype=np.float32),
            k=np.array(self.k), l=np.array(self.l),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def from_npz(cls, path) -> "FastWordEncoder":
        d = np.load(path, allow_pickle=True)
        idf = {str(w): float(v) for w, v in zip(d["idf_words"], d["idf_values"])}
        return cls(
            [str(w) for w in d["words"]], d["vectors"], idf,
            float(d["default_idf"]), int(d["k"]), int(d["l"]),
            meta=json.loads(str(d["meta"])),
        )
