"""Frozen text encoders: the real CLM one (Qwen3-8B behind vLLM) and a CPU stand-in.

Both map texts to L2-normalised vectors and never change, which is the property the online loop
relies on: logged embeddings, the outcome map and replay sets stay valid across head updates.

* ``ClmEncoder``   ``clm.Embedder`` against a vLLM pooling server (the encoder the released head
                   was trained with); use with the reference checkpoint from ``clm-download``.
* ``HashEncoder``  bag of words and word bigrams, each hashed to a fixed Gaussian vector and
                   summed.  No GPU, deterministic, and texts that share words get similar vectors,
                   which is all the testbeds need to exercise the learning machinery.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class HashEncoder:
    def __init__(self, dim: int = 256, ngram: int = 2):
        self.dim, self.ngram = dim, ngram
        self._feat: dict[str, np.ndarray] = {}
        self._text: dict[str, np.ndarray] = {}

    def _vec(self, feature: str) -> np.ndarray:
        v = self._feat.get(feature)
        if v is None:
            seed = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            v = self._feat[feature] = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v

    def features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = list(words)
        for n in range(2, self.ngram + 1):
            feats += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        return feats or [""]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = self._text.get(t)
            if v is None:
                v = self._text[t] = _l2(np.sum([self._vec(f) for f in self.features(t)], axis=0))
            out.append(v)
        return np.stack(out)


class ClmEncoder:
    """The encoder CLM serves with; ``clm.Embedder`` keeps its own LRU cache."""

    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str = "qwen3-8b",
                 max_tokens: int = 2048):
        from clm.embedder import Embedder
        self.embedder = Embedder(url=url, model=model, max_tokens=max_tokens)
        self.dim = 4096

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)[0]
