"""The experience log: what CLM was asked, what it chose, and how the episode ended.

A ``Decision`` stores encoder embeddings, not text.  The encoder is frozen and ``clm.Embedder``
already caches every vector it served, so recording costs no extra encoder call, and training
on the exact serving-time vectors avoids train/serve skew in tokenisation or truncation.

Outcomes usually arrive per episode (task passed, game survived), not per step.  ``assign_credit``
turns one episode outcome into per-step labels; how it does that is the main design choice:

* ``all``             every step inherits the outcome.  Simple, and wrong for long failures:
                      most steps of a failed run were fine.
* ``terminal``        successes label every step; failures label only the last ``window`` steps
                      negative and leave earlier steps unlabelled (CLM's best-of-N evaluation
                      scores the same final window).
* ``discounted``      step weight ``gamma ** (T - 1 - t)``: the closer to the end, the more credit.

Steps with a dense signal of their own (a verifier, a safety shield that overrode the action, a
failing unit test) should set ``Decision.step_reward``; that label always wins over the episode's.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Decision:
    episode: str
    step: int
    state: np.ndarray                 # [d] encoder embedding of the state text the head saw
    candidates: np.ndarray            # [K, d] encoder embeddings of the options
    chosen: int                       # index executed
    probs: np.ndarray                 # [K] answer distribution served (the behaviour policy)
    group: int                        # decision-point id: same state + same question
    head_version: int = 0
    step_reward: float | None = None  # dense per-step signal, if the environment has one
    task: int | None = None           # task id (outcome map: same-task bonus)


@dataclass
class Labeled:
    d: Decision
    success: bool
    weight: float


@dataclass
class Episode:
    decisions: list[Decision] = field(default_factory=list)
    outcome: float | None = None


def assign_credit(decisions: list[Decision], outcome: float, strategy: str = "terminal",
                  window: int = 1, gamma: float = 0.9) -> list[Labeled]:
    """Per-step labels from one episode outcome (``outcome > 0.5`` = success)."""
    win = outcome > 0.5
    n = len(decisions)
    out = []
    for t, d in enumerate(decisions):
        if d.step_reward is not None:
            out.append(Labeled(d, d.step_reward > 0.5, 1.0))
            continue
        if strategy == "all":
            out.append(Labeled(d, win, 1.0))
        elif strategy == "terminal":
            if win or t >= n - window:
                out.append(Labeled(d, win, 1.0))
        elif strategy == "discounted":
            out.append(Labeled(d, win, gamma ** (n - 1 - t)))
        else:
            raise ValueError(f"unknown credit strategy {strategy!r}")
    return out


class ExperienceBuffer:
    """Open episodes, then a bounded store of labelled steps (reservoir-sampled when full).

    ``baseline(group)`` is a running mean reward per decision point, the advantage baseline for
    the bandit objective; a global mean stands in for unseen groups.
    """

    def __init__(self, capacity: int = 200_000, strategy: str = "terminal", window: int = 1,
                 gamma: float = 0.9, seed: int = 0):
        self.capacity, self.strategy, self.window, self.gamma = capacity, strategy, window, gamma
        self.open: dict[str, Episode] = defaultdict(Episode)
        self.items: list[Labeled] = []
        self.fresh: deque[Labeled] = deque()
        self.seen = 0
        self.rng = random.Random(seed)
        self._sum: dict[int, float] = defaultdict(float)
        self._cnt: dict[int, int] = defaultdict(int)
        self._all = [0.0, 0]

    def record(self, d: Decision) -> None:
        self.open[d.episode].decisions.append(d)

    def close(self, episode: str, outcome: float) -> list[Labeled]:
        ep = self.open.pop(episode, None)
        if ep is None:
            return []
        labeled = assign_credit(ep.decisions, outcome, self.strategy, self.window, self.gamma)
        for item in labeled:
            r = float(item.success)
            self._sum[item.d.group] += r; self._cnt[item.d.group] += 1
            self._all[0] += r; self._all[1] += 1
            self.fresh.append(item)
            self.seen += 1
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                j = self.rng.randrange(self.seen)
                if j < self.capacity:
                    self.items[j] = item
        return labeled

    def discard(self, episode: str) -> None:
        """Drop an open episode that will never get an outcome."""
        self.open.pop(episode, None)

    def baseline(self, group: int) -> float:
        if self._cnt[group] >= 2:
            return self._sum[group] / self._cnt[group]
        return self._all[0] / self._all[1] if self._all[1] else 0.5

    def take_fresh(self) -> list[Labeled]:
        out = list(self.fresh); self.fresh.clear()
        return out

    def sample(self, n: int) -> list[Labeled]:
        return self.rng.sample(self.items, min(n, len(self.items)))

    def __len__(self) -> int:
        return len(self.items)


def to_tensors(items: list[Labeled], device="cpu") -> dict[str, torch.Tensor]:
    """Stack labelled steps into the tensors ``objectives`` expects (candidates padded to max K)."""
    k = max(len(i.d.candidates) for i in items)
    dim = items[0].d.state.shape[-1]
    cands = np.zeros((len(items), k, dim), dtype=np.float32)
    mask = np.zeros((len(items), k), dtype=bool)
    for r, i in enumerate(items):
        cands[r, :len(i.d.candidates)] = i.d.candidates
        mask[r, :len(i.d.candidates)] = True
    t = lambda x, **kw: torch.as_tensor(np.asarray(x), device=device, **kw)  # noqa: E731
    return {
        "s": t(np.stack([i.d.state for i in items]), dtype=torch.float32),
        "a": t(np.stack([i.d.candidates[i.d.chosen] for i in items]), dtype=torch.float32),
        "cands": t(cands), "mask": t(mask),
        "chosen": t([i.d.chosen for i in items], dtype=torch.long),
        "logged_prob": t([float(i.d.probs[i.d.chosen]) for i in items], dtype=torch.float32),
        "success": t([i.success for i in items], dtype=torch.bool),
        "weight": t([i.weight for i in items], dtype=torch.float32),
        "group": t([i.d.group for i in items], dtype=torch.long),
        "task": t([i.d.task if i.d.task is not None else -1 for i in items], dtype=torch.long),
    }
