# mowm/qa/corpus_indexer.py
"""CorpusIndexer — mine discriminative vocabulary from the NV-Embed codebook.

Extracts TF-IDF scored words per domain from the codebook texts. These
replace manual synonym lists with data-driven domain vocabulary.

Usage:
    indexer = CorpusIndexer.from_codebook("mowm_nvembed_codebook.npz")
    physics_words = indexer.top_words("physics", n=100)
    # -> ['velocity', 'acceleration', 'kinetic', 'momentum', ...]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it",
    "its", "how", "why", "where", "there", "here", "up", "about",
    "then", "them", "they", "he", "she", "we", "you", "me", "him",
    "her", "us", "my", "your", "his", "our", "their", "also",
    "like", "well", "back", "even", "still", "way", "take",
    "come", "make", "get", "go", "see", "use", "new", "one", "two",
    "first", "last", "long", "great", "little", "own", "old",
    "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "public", "bad", "good",
    "much", "many", "over", "under", "between", "such", "while",
    "said", "been", "made", "found", "used", "part", "based",
    "called", "known", "using", "example", "including", "however",
    "number", "may", "can", "also", "often", "well", "many",
})


@dataclass
class DomainVocab:
    """Scored vocabulary for a single domain."""
    domain: str
    word_scores: dict[str, float] = field(default_factory=dict)
    total_docs: int = 0


class CorpusIndexer:
    """Mine discriminative vocabulary from corpus texts by domain.

    Uses TF-IDF: words that are frequent in one domain but rare across
    others get the highest scores. These words are the domain's
    "fingerprint" — the vocabulary that distinguishes it.
    """

    def __init__(self) -> None:
        self._domain_vocabs: dict[str, DomainVocab] = {}
        self._total_domains: int = 0

    @classmethod
    def from_codebook(
        cls, codebook_path: str, min_word_len: int = 3,
    ) -> "CorpusIndexer":
        """Build index from NV-Embed codebook texts and domains.

        Args:
            codebook_path: Path to .npz codebook file.
            min_word_len: Minimum word length to index.

        Returns:
            CorpusIndexer with TF-IDF scored vocabulary per domain.
        """
        cb = np.load(codebook_path, allow_pickle=True)
        texts = list(cb["texts"])
        domains = list(cb["domains"])

        indexer = cls()
        indexer._build_from_texts(texts, domains, min_word_len)
        return indexer

    def from_texts(
        self,
        texts: list[str],
        domains: list[str],
        min_word_len: int = 3,
    ) -> None:
        """Build index from raw text + domain label pairs."""
        self._build_from_texts(texts, domains, min_word_len)

    def _build_from_texts(
        self,
        texts: list[str],
        domains: list[str],
        min_word_len: int,
    ) -> None:
        """Compute TF-IDF scores per domain."""
        # Count word frequency per domain
        domain_word_counts: dict[str, Counter] = {}
        domain_doc_counts: dict[str, int] = {}

        # Also track document frequency (how many domains contain this word)
        word_domain_presence: dict[str, set[str]] = {}

        for text, domain in zip(texts, domains):
            domain_doc_counts[domain] = domain_doc_counts.get(domain, 0) + 1
            words = {
                t.lower() for t in _TOKEN_RE.findall(str(text))
                if len(t) >= min_word_len and t.lower() not in _STOPWORDS
            }

            counter = domain_word_counts.setdefault(domain, Counter())
            counter.update(words)

            for w in words:
                word_domain_presence.setdefault(w, set()).add(domain)

        self._total_domains = len(domain_word_counts)

        # Compute TF-IDF per domain
        for domain, counter in domain_word_counts.items():
            vocab = DomainVocab(
                domain=domain,
                total_docs=domain_doc_counts.get(domain, 0),
            )

            total_words = sum(counter.values()) or 1
            for word, count in counter.items():
                # TF: normalized frequency within domain
                tf = count / total_words

                # IDF: inverse domain frequency (how unique to this domain)
                n_domains_with_word = len(
                    word_domain_presence.get(word, set()),
                )
                idf = math.log(
                    (self._total_domains + 1) / (n_domains_with_word + 1),
                )

                vocab.word_scores[word] = tf * idf

            self._domain_vocabs[domain] = vocab

    def top_words(self, domain: str, n: int = 100) -> list[str]:
        """Return the top-N most discriminative words for a domain.

        Args:
            domain: Domain name.
            n: Number of words to return.

        Returns:
            List of words sorted by TF-IDF score (highest first).
        """
        vocab = self._domain_vocabs.get(domain)
        if not vocab:
            return []

        ranked = sorted(
            vocab.word_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [word for word, _ in ranked[:n]]

    def top_words_with_scores(
        self, domain: str, n: int = 100,
    ) -> list[tuple[str, float]]:
        """Return top-N words with their TF-IDF scores."""
        vocab = self._domain_vocabs.get(domain)
        if not vocab:
            return []

        ranked = sorted(
            vocab.word_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]

    def domain_vocabulary(
        self, min_score: float = 0.0,
    ) -> dict[str, set[str]]:
        """Return vocabulary sets for all domains.

        Args:
            min_score: Minimum TF-IDF score to include a word.

        Returns:
            Dict mapping domain name to set of discriminative words.
        """
        result: dict[str, set[str]] = {}
        for domain, vocab in self._domain_vocabs.items():
            words = {
                w for w, s in vocab.word_scores.items() if s >= min_score
            }
            result[domain] = words
        return result

    @property
    def domains(self) -> list[str]:
        """List of indexed domains."""
        return list(self._domain_vocabs.keys())

    @property
    def total_words(self) -> int:
        """Total unique words across all domains."""
        all_words: set[str] = set()
        for vocab in self._domain_vocabs.values():
            all_words.update(vocab.word_scores.keys())
        return len(all_words)

    def save(self, path: str) -> None:
        """Save indexed vocabulary to .npz file."""
        domains = []
        words_list = []
        scores_list = []
        for domain, vocab in self._domain_vocabs.items():
            for word, score in vocab.word_scores.items():
                domains.append(domain)
                words_list.append(word)
                scores_list.append(score)

        np.savez_compressed(
            path,
            domains=np.array(domains, dtype=object),
            words=np.array(words_list, dtype=object),
            scores=np.array(scores_list, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str) -> "CorpusIndexer":
        """Load indexed vocabulary from .npz file."""
        data = np.load(path, allow_pickle=True)
        domains = list(data["domains"])
        words = list(data["words"])
        scores = list(data["scores"])

        indexer = cls()
        for domain, word, score in zip(domains, words, scores):
            vocab = indexer._domain_vocabs.setdefault(
                domain, DomainVocab(domain=domain),
            )
            vocab.word_scores[word] = float(score)

        indexer._total_domains = len(indexer._domain_vocabs)
        return indexer
