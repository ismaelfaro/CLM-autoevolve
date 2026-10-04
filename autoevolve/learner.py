"""The online learner: experience in, gated LoRA updates out, published as a hot-reloadable head.

    learner = OnlineLearner(base_ckpt, publish_path="heads/clm-online.pt", replay=(S, A))
    ... buffer.record(decision) per answer; buffer.close(episode, outcome) per episode ...
    learner.update(buffer)                       # a few optimiser steps on fresh + replayed steps
    learner.maybe_publish(eval_fn)               # promote only if the held-out score improves

``clm-serve --model clm-online=heads/clm-online.pt`` serves the published head next to
``clm-latest``; the server reloads it when the file's mtime changes.  Every promoted adapter is
kept (``versions/``), so a rollback is publishing an older one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from . import objectives as obj
from .experience import ExperienceBuffer, Labeled, to_tensors
from .lora import LoRAHeads, atomic_save


@dataclass
class LearnerConfig:
    rank: int = 16
    alpha: float = 32.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps_per_update: int = 8
    batch: int = 256
    fresh_frac: float = 0.5           # of each batch: newest steps; the rest sampled from the buffer
    replay_batch: int = 256           # original-distribution pairs per step (anti-forgetting)
    replay_weight: float = 0.4        # README: 40% Nemotron replay kept hard-negative top-1 at 68.5 vs 56.2
    kl_weight: float = 0.1
    losses: tuple[str, ...] = ("infonce",)   # any of: infonce, ce, bandit, dpo
    hard_negative_boost: float = 1.0
    bandit_clip: float = 0.2
    dpo_beta: float = 0.1
    use_failures: bool = True         # False: learn from successes only (failures dropped)
    ignore_outcome: bool = False      # True: every executed step is a positive, as train/finetune.py does today
    promote_margin: float = 0.0
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class LearnerStats:
    updates: int = 0
    published: int = 0
    rejected: int = 0
    history: list = field(default_factory=list)


class OnlineLearner:
    def __init__(self, base_ckpt: dict, publish_path: str | None = None, cfg: LearnerConfig | None = None,
                 replay: tuple[torch.Tensor, torch.Tensor] | None = None,
                 anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        self.cfg = cfg or LearnerConfig()
        self.model = LoRAHeads(base_ckpt, rank=self.cfg.rank, alpha=self.cfg.alpha).to(self.cfg.device)
        self.opt = torch.optim.AdamW(self.model.trainable_parameters(), lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)
        self.publish_path = publish_path
        self.replay = replay          # (states [N, d], gold actions [N, d]) from the base training mix
        self.anchor = anchor          # (states [M, d], candidate sets [M, K, d], mask [M, K])
        self.stats = LearnerStats()
        self.version = 0
        self.best_adapter = self.model.adapter_state()
        self.best_score: float | None = None
        self._g = torch.Generator().manual_seed(0)

    # ------------------------------------------------------------------ training
    def _replay_batch(self, n):
        idx = torch.randint(len(self.replay[0]), (n,), generator=self._g)
        return self.replay[0][idx].to(self.cfg.device), self.replay[1][idx].to(self.cfg.device)

    def _anchor_batch(self, n):
        idx = torch.randint(len(self.anchor[0]), (n,), generator=self._g)
        return tuple(x[idx].to(self.cfg.device) for x in self.anchor)

    def loss(self, batch: dict[str, torch.Tensor], buffer: ExperienceBuffer | None) -> tuple[torch.Tensor, dict]:
        c, m = self.cfg, self.model
        parts: dict[str, torch.Tensor] = {}
        if "infonce" in c.losses:
            parts["infonce"] = obj.outcome_infonce(m, batch["s"], batch["a"], batch["success"], batch["group"],
                                                   batch["weight"], c.hard_negative_boost)
        if "ce" in c.losses:
            parts["ce"] = obj.outcome_ce(m, batch["s"], batch["cands"], batch["mask"], batch["chosen"],
                                         batch["success"], batch["weight"])
        if "bandit" in c.losses:
            base = torch.tensor([buffer.baseline(int(g)) for g in batch["group"]] if buffer else
                                [0.5] * len(batch["group"]), device=c.device)
            adv = (batch["success"].float() - base) * batch["weight"]
            parts["bandit"] = obj.bandit_ppo(m, batch["s"], batch["cands"], batch["mask"], batch["chosen"],
                                             batch["logged_prob"], adv, c.bandit_clip)
        if "dpo" in c.losses:
            pairs = _preference_pairs(batch)
            if pairs is not None:
                parts["dpo"] = obj.pairwise_dpo(m, *pairs, beta=c.dpo_beta)
        if self.replay is not None and c.replay_weight > 0:
            parts["replay"] = c.replay_weight * obj.replay_infonce(m, *self._replay_batch(c.replay_batch))
        if self.anchor is not None and c.kl_weight > 0:
            parts["kl"] = c.kl_weight * obj.anchor_kl(m, *self._anchor_batch(c.replay_batch))
        total = sum(parts.values()) if parts else torch.zeros((), device=c.device)
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def update(self, buffer: ExperienceBuffer, steps: int | None = None) -> dict:
        """A few optimiser steps on fresh experience mixed with older buffered steps."""
        fresh = buffer.take_fresh()
        if not fresh and not len(buffer):
            return {}
        self.model.train()
        logs = {}
        for _ in range(steps or self.cfg.steps_per_update):
            n_fresh = min(len(fresh), int(self.cfg.batch * self.cfg.fresh_frac))
            items = (buffer.rng.sample(fresh, n_fresh) if n_fresh else []) + \
                buffer.sample(self.cfg.batch - n_fresh)
            items = self._view(items)
            if not items:
                break
            loss, logs = self.loss(to_tensors(items, self.cfg.device), buffer)
            if not loss.requires_grad:
                continue
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.cfg.grad_clip)
            self.opt.step()
        self.stats.updates += 1
        return logs

    def _view(self, items):
        if self.cfg.ignore_outcome:
            return [Labeled(i.d, True, i.weight) for i in items]
        if not self.cfg.use_failures:
            return [i for i in items if i.success]
        return items

    # ------------------------------------------------------------------ gating / publishing
    def maybe_publish(self, evaluate: Callable[[LoRAHeads], float]) -> bool:
        """Score the candidate on held-out episodes; publish if it beats the last promoted adapter,
        else roll the live weights back to it (so a bad update never compounds)."""
        self.model.eval()
        with torch.no_grad():
            score = float(evaluate(self.model))
        if self.best_score is None:
            current = self.model.adapter_state()
            self.model.load_adapter(self.best_adapter)
            with torch.no_grad():
                self.best_score = float(evaluate(self.model))
            self.model.load_adapter(current)
        ok = score > self.best_score + self.cfg.promote_margin
        self.stats.history.append({"t": time.time(), "score": score, "incumbent": self.best_score, "promoted": ok})
        if ok:
            self.best_score, self.best_adapter = score, self.model.adapter_state()
            self.publish()
        else:
            self.stats.rejected += 1
            self.model.load_adapter(self.best_adapter)
        return ok

    def publish(self) -> str | None:
        self.version += 1
        self.stats.published += 1
        if not self.publish_path:
            return None
        ck = self.model.merged_checkpoint({"version": self.version, "score": self.best_score})
        vdir = os.path.join(os.path.dirname(os.path.abspath(self.publish_path)), "versions")
        os.makedirs(vdir, exist_ok=True)
        torch.save(self.model.adapter_state(), os.path.join(vdir, f"adapter_v{self.version:04d}.pt"))
        atomic_save(ck, self.publish_path)
        with open(os.path.join(vdir, "log.jsonl"), "a") as f:
            f.write(json.dumps({"version": self.version, "score": self.best_score, "t": time.time()}) + "\n")
        return self.publish_path


def _preference_pairs(batch):
    """(state, winner, loser) for every group with both a success and a failure in the batch."""
    s, a, ok, g = batch["s"], batch["a"], batch["success"], batch["group"]
    win, lose = {}, {}
    for i, (gi, oi) in enumerate(zip(g.tolist(), ok.tolist())):
        (win if oi else lose).setdefault(gi, i)
    common = [k for k in win if k in lose]
    if not common:
        return None
    wi = torch.tensor([win[k] for k in common]); li = torch.tensor([lose[k] for k in common])
    return s[wi], a[wi], a[li]
