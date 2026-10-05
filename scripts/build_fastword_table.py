"""Build the production FastWordEncoder table (one-off, offline).

Vocabulary = top corpus words by document frequency (sampled from a token
shard) UNION every word in the axiom corpus. IDF = broad-corpus document
frequencies from the same sample. Anisotropy (mean + top PC) = fit on the
FULL broad-vocabulary teacher embeddings — the external-statistics form
validated by CubbyLLM's `exp_f2m2_m2v_ood` check (in-domain stats inflate
routing by ~0.03 macro). The sentence-transformer teacher runs HERE, once;
serving (`FastWordEncoder.from_npz`) never imports it.

The table is a model asset, not a source file: it is written to
D:\\CUBBY-TRAINED-MODELS by default (git-untracked, per the repo convention
for trained artifacts), with a build log beside it and the routing eval
embedded in the npz meta.

Run from the mowm repo root:  uv run python scripts/build_fastword_table.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from mowm.encoding.semantic_words import (  # noqa: E402
        FastWordEncoder,
        _unit_rows,
        fit_anisotropy,
        split_words,
    )
except Exception:
    # semantic_words is numpy-only by design, but `import mowm` pulls the full
    # package (cubemind and friends). When this builder runs under an
    # interpreter that lacks those — e.g. CubbyLLM's env, whose teacher stack
    # works while mowm's has a tokenizers pin conflict — load the module by
    # file path instead, skipping the package __init__ entirely.
    import importlib.util

    _p = (pathlib.Path(__file__).resolve().parents[1]
          / "mowm" / "encoding" / "semantic_words.py")
    _spec = importlib.util.spec_from_file_location("semantic_words", _p)
    assert _spec is not None and _spec.loader is not None
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    FastWordEncoder = _m.FastWordEncoder
    _unit_rows = _m._unit_rows
    fit_anisotropy = _m.fit_anisotropy
    split_words = _m.split_words

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def corpus_stats(shard: str, tokenizer_json: str, n_docs: int, doc_len: int, seed: int = 0):
    """Sample decoded windows from a token shard -> (document frequencies, n_docs)."""
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(tokenizer_json)
    corpus = np.memmap(shard, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(corpus) - doc_len, size=n_docs)
    df: dict[str, int] = {}
    t0 = time.perf_counter()
    for i, s in enumerate(starts):
        doc = tk.decode([int(t) for t in corpus[s:s + doc_len]]).lower()
        for w in {w for w in _WORD_SPLIT.split(doc) if len(w) >= 2}:
            df[w] = df.get(w, 0) + 1
        if (i + 1) % 10_000 == 0:
            print(f"  decoded {i + 1}/{n_docs} docs ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    return df, n_docs


def loo_routing_eval(encoder: FastWordEncoder, axioms: list[dict]) -> dict:
    """The validated LOO domain-routing metric, compact form (macro null on
    this corpus is 53/837 = 0.0633; micro null 19/155 = 0.1226)."""
    names = [a["name"] for a in axioms]
    formulas = [a["formula_str"] for a in axioms]
    domains = [a["domain"].split(".")[0] for a in axioms]
    N = np.stack([encoder.encode(t).ravel() for t in names]).astype(np.float64)
    F = np.stack([encoder.encode(t).ravel() for t in formulas]).astype(np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    sims = N @ F.T
    # self-retrieval BEFORE the LOO mask (the mask removes the diagonal, which
    # is exactly what self-retrieval measures — computing it after is always 0)
    top1 = float((sims.argmax(axis=1) == np.arange(len(names))).mean())
    np.fill_diagonal(sims, -np.inf)
    doms = np.asarray(domains)
    hit = doms[sims.argmax(axis=1)] == doms
    per_domain = {d: float(hit[doms == d].mean()) for d in sorted(set(domains))}
    return {"macro": float(np.mean(list(per_domain.values()))),
            "micro": float(hit.mean()), "self_retrieval_top1": top1,
            "per_domain": per_domain}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard",
                    default=r"D:\grillcheese_training_data\token_cache\wiki_full.u32")
    ap.add_argument("--tokenizer",
                    default=r"C:\Users\grill\Documents\GitHub\CubbyLLM\data\grillcheese_bbpe128k.json")
    ap.add_argument("--axioms",
                    default=r"C:\Users\grill\Documents\GitHub\CubbyLLM\data\mowm_axioms.json")
    ap.add_argument("--out", default=r"D:\CUBBY-TRAINED-MODELS\fastword_table_v1.npz")
    ap.add_argument("--docs", type=int, default=50_000)
    ap.add_argument("--doc-len", type=int, default=200)
    ap.add_argument("--top-words", type=int, default=30_000)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--teacher",
                    default="st:sentence-transformers/all-MiniLM-L6-v2",
                    help="st:<sentence-transformers id> or m2v:<model2vec "
                         "static id> (e.g. m2v:minishlab/potion-base-8M — "
                         "the teacher swap CubbyLLM's exp_m3_potion_baseline "
                         "measured at +0.05 routing macro over MiniLM)")
    ap.add_argument("--signal-blend", type=float, default=0.5)
    ap.add_argument("--dump-teacher", default="",
                    help="also save the raw teacher vectors (vocab, E) to this .npz")
    args = ap.parse_args()

    t_all = time.perf_counter()
    print(f"corpus stats: {args.docs} docs x {args.doc_len} tokens from "
          f"{pathlib.Path(args.shard).name}")
    df, n_docs = corpus_stats(args.shard, args.tokenizer, args.docs, args.doc_len)
    print(f"  {len(df)} distinct corpus words")

    axioms = json.loads(pathlib.Path(args.axioms).read_text(encoding="utf-8"))["axioms"]
    axiom_words: set[str] = set()
    for a in axioms:
        axiom_words |= split_words(a["name"]) | split_words(a["formula_str"])
        axiom_words |= split_words(a["domain"].replace(".", " "))
    top = sorted(df, key=lambda w: -df[w])[:args.top_words]
    vocab = sorted(set(top) | axiom_words)
    covered = sum(1 for w in axiom_words if w in df)
    print(f"vocabulary: top {args.top_words} corpus words + {len(axiom_words)} axiom words "
          f"-> {len(vocab)} unique ({covered}/{len(axiom_words)} axiom words have corpus DF)")

    idf = {w: float(np.log(n_docs / df[w])) for w in vocab if w in df}
    default_idf = float(np.log(n_docs))

    kind, _, teacher_id = args.teacher.partition(":")
    print(f"embedding vocabulary with the teacher ({args.teacher}; once, offline)...")
    t0 = time.perf_counter()
    if kind == "m2v":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(teacher_id)
        E = np.asarray(model.encode(vocab, show_progress_bar=False),
                       dtype=np.float32)
    elif kind == "st":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(teacher_id)
        E = np.zeros((len(vocab), model.get_sentence_embedding_dimension()), dtype=np.float32)
        for a0 in range(0, len(vocab), 5000):
            b0 = min(a0 + 5000, len(vocab))
            E[a0:b0] = model.encode(vocab[a0:b0], batch_size=args.batch,
                                    show_progress_bar=False)
            print(f"  embedded {b0}/{len(vocab)} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    elif kind == "npz":
        # precomputed teacher vectors for this exact vocabulary (e.g. a specialised
        # copy of an earlier --dump-teacher), so a post-processed teacher builds
        # through the same anisotropy / projection / blend path
        z = np.load(teacher_id, allow_pickle=True)
        if [str(w) for w in z["vocab"]] != vocab:
            raise SystemExit("npz teacher: vocabulary differs from this build's")
        E = np.asarray(z["E"], dtype=np.float32)
    else:
        raise SystemExit(f"unknown teacher kind {kind!r} (use st:, m2v: or npz:)")
    embed_s = time.perf_counter() - t0
    print(f"  teacher dim {E.shape[1]}, {embed_s:.0f}s")
    if args.dump_teacher:
        np.savez(args.dump_teacher, vocab=np.array(vocab, dtype=object), E=E)
        print(f"  teacher vectors -> {args.dump_teacher}")

    anis = fit_anisotropy(_unit_rows(E))
    print(f"anisotropy stats fit on all {len(vocab)} broad-vocabulary embeddings")

    def _embed_fn(ws: list[str]) -> np.ndarray:
        assert ws == vocab, "build() must see the exact prepared vocabulary order"
        return E

    enc = FastWordEncoder.build(
        vocab, _embed_fn, idf, default_idf,
        k=args.k, l=args.l, proj_seed=args.proj_seed, anisotropy=anis,
        signal_blend=args.signal_blend,
        meta={"teacher": args.teacher,
              "shard": pathlib.Path(args.shard).name, "docs": n_docs,
              "doc_len": args.doc_len, "top_words": args.top_words,
              "anisotropy": "mean+PC1 fit on full broad vocabulary",
              "idf": "log(n_docs/df), broad-corpus document frequencies",
              "built": time.strftime("%Y-%m-%d %H:%M:%S")},
    )

    print("routing eval over the axiom corpus (LOO; nulls: macro 0.0633, micro 0.1226)...")
    ev = loo_routing_eval(enc, axioms)
    print(f"  macro {ev['macro']:.4f} ({ev['macro'] / (53 / 837):.2f}x null) | "
          f"micro {ev['micro']:.4f} | self-retrieval {ev['self_retrieval_top1']:.3f}")
    t0 = time.perf_counter()
    for a in axioms:
        enc.encode(a["name"])
    lat_us = (time.perf_counter() - t0) / len(axioms) * 1e6
    print(f"  latency: {lat_us:.0f} us/challenge (table of {len(vocab)} words)")
    enc.meta["eval"] = {k_: v for k_, v in ev.items() if k_ != "per_domain"}
    enc.meta["eval_per_domain"] = ev["per_domain"]
    enc.meta["latency_us_per_challenge"] = lat_us
    enc.meta["embed_seconds"] = embed_s

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    enc.save_npz(out, dtype=np.float16)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB f16, "
          f"{time.perf_counter() - t0:.0f}s)")

    check = FastWordEncoder.from_npz(out)
    a_ = enc.encode("what force accelerates a two kilogram mass").ravel()
    b_ = check.encode("what force accelerates a two kilogram mass").ravel()
    cos = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12))
    print(f"reload check: cosine(pre-save, post-load) = {cos:.6f}")
    print(f"total build time {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
