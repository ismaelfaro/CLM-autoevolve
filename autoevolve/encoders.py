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


class TransformersEncoder:
    """Qwen3-8B in-process with Hugging Face ``transformers``: no vLLM server needed (e.g. Colab).

    Mirrors what ``clm-serve``'s embedder gets from ``vllm serve Qwen/Qwen3-8B --runner pooling``:
    the raw text (no chat template), the last ``max_tokens`` tokens kept, the final hidden state
    of the last token, L2-normalised.  Numerics differ slightly from vLLM's kernels; the released
    head was trained on vLLM embeddings, so treat small score differences as expected.
    Needs ~17 GB of GPU memory in bf16 (Colab L4 / A100).
    """

    def __init__(self, model: str = "Qwen/Qwen3-8B", device: str | None = None, max_tokens: int = 2048,
                 batch: int = 16, dtype: str = "bfloat16", hf_model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model)
        self.tok.padding_side, self.tok.truncation_side = "right", "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if hf_model is None:
            import transformers
            # transformers 5 renamed torch_dtype -> dtype; 4.x would silently load fp32 (32 GB) with `dtype`
            key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            hf_model = AutoModel.from_pretrained(model, **{key: getattr(torch, dtype)})
        self.model = hf_model.to(self.device).eval()
        self.max_tokens, self.batch = max_tokens, batch
        self.dim = self.model.config.hidden_size
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_tokens,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state                     # [b, T, H], after the final norm
            last = enc["attention_mask"].sum(1) - 1                         # right padding: last real token
            v = h[self.torch.arange(len(chunk), device=h.device), last].float().cpu().numpy()
            for t, x in zip(chunk, _l2(v)):
                self._cache[t] = x.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])


class ClmEncoder:
    """The encoder CLM serves with; ``clm.Embedder`` keeps its own LRU cache."""

    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str = "qwen3-8b",
                 max_tokens: int = 2048):
        from clm.embedder import Embedder
        self.embedder = Embedder(url=url, model=model, max_tokens=max_tokens)
        self.dim = 4096

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)[0]
