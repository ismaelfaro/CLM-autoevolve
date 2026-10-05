"""The online learner: experience in, gated LoRA updates out, published as a hot-reloadable head.

    learner = OnlineLearner(base_ckpt, publish_path="heads/clm-online.pt", replay=(S, A))
    ... act with learner.serving; buffer.record(decision); buffer.close(episode, outcome) ...
    learner.update(buffer)                       # a few optimiser steps on the *candidate* adapter
    learner.maybe_publish(evaluate)              # promote the candidate if it beats what is served

Two adapters live side by side.  ``model`` is the candidate and keeps learning across updates, so
small gains can accumulate.  ``serving`` is what acts and what ``clm-serve`` has.  The gate scores
both on the *same* held-out data every time.  The candidate is promoted only when it is better.
It is reset to the served adapter only after ``rollback_patience`` gates in a row where it is
clearly worse (a sustained regression, not one noisy batch).

``clm-serve --model clm-online=heads/clm-online.pt`` serves the published head next to
``clm-latest``; the server reloads it when the file's mtime changes.  Every promoted adapter is
kept (``versions/``), so rolling back the *server* is publishing an older one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

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
    losses: tuple[str, ...] = ("infonce",)   # any of: infonce, ce, bandit, dpo, distill
    hard_negative_boost: float = 1.0
    bandit_clip: float = 0.2
    dpo_beta: float = 0.1
    distill_beta: float = 4.0
    use_failures: bool = True         # False: learn from successes only (failures dropped)
    ignore_outcome: bool = False      # True: every executed step is a positive, as train/finetune.py does today
    gate: bool = True                 # False: every update goes straight to serving
    promote_margin: float = 0.0
    rollback_tolerance: float = 0.02  # "clearly worse" = below the served score by this much
    rollback_patience: int = 3
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class LearnerStats:
    updates: int = 0
    published: int = 0
    rejected: int = 0
    rollbacks: int = 0
    history: list = field(default_factory=list)


class OnlineLearner:
    def __init__(self, base_ckpt: dict, publish_path: str | None = None, cfg: LearnerConfig | None = None,
                 replay: tuple[torch.Tensor, torch.Tensor] | None = None,
                 anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                 outcome_map=None, task_of: Callable | None = None):
        self.cfg = cfg or LearnerConfig()
        mk = lambda: LoRAHeads(base_ckpt, rank=self.cfg.rank, alpha=self.cfg.alpha).to(self.cfg.device)  # noqa: E731
        self.model, self.serving = mk(), mk().eval()
        self.serving.load_adapter(self.model.adapter_state())
        self._new_optimizer()
        self.publish_path = publish_path
        self.replay = replay          # (states [N, d], gold actions [N, d]) from the base training mix
        self.anchor = anchor          # (states [M, d], candidate sets [M, K, d], mask [M, K])
        self.outcome_map = outcome_map
        self.task_of = task_of
        self.stats = LearnerStats()
        self.version = 0
        self._bad = 0
        self._g = torch.Generator().manual_seed(0)

    def _new_optimizer(self):
        self.opt = torch.optim.AdamW(self.model.trainable_parameters(), lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)

    # ------------------------------------------------------------------ training
    def _replay_batch(self, n):
        idx = torch.randint(len(self.replay[0]), (n,), generator=self._g)
        return self.replay[0][idx].to(self.cfg.device), self.replay[1][idx].to(self.cfg.device)

    def _anchor_batch(self, n):
        idx = torch.randint(len(self.anchor[0]), (n,), generator=self._g)
        return tuple(x[idx].to(self.cfg.device) for x in self.anchor)

    def loss(self, batch: dict[str, torch.Tensor], buffer: ExperienceBuffer | None,
             tasks: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
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
        if "distill" in c.losses and self.outcome_map is not None and len(self.outcome_map):
            target, conf = self.outcome_map.soft_targets(batch["s"], batch["cands"], batch["mask"], tasks,
                                                         beta=c.distill_beta)
            parts["distill"] = obj.map_distill(m, batch["s"], batch["cands"], batch["mask"], target, conf)
        if self.replay is not None and c.replay_weight > 0:
            parts["replay"] = c.replay_weight * obj.replay_infonce(m, *self._replay_batch(c.replay_batch))
        if self.anchor is not None and c.kl_weight > 0:
            parts["kl"] = c.kl_weight * obj.anchor_kl(m, *self._anchor_batch(c.replay_batch))
        total = sum(parts.values()) if parts else torch.zeros((), device=c.device)
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def _view(self, items):
        if self.cfg.ignore_outcome:
            return [Labeled(i.d, True, i.weight) for i in items]
        if not self.cfg.use_failures:
            return [i for i in items if i.success]
        return items

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
            tasks = (torch.tensor([self.task_of(i.d) for i in items], device=self.cfg.device)
                     if self.task_of else None)
            loss, logs = self.loss(to_tensors(items, self.cfg.device), buffer, tasks)
            if not loss.requires_grad:
                continue
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.cfg.grad_clip)
            self.opt.step()
        self.stats.updates += 1
        self.model.eval()
        if not self.cfg.gate:
            self._promote(None)
        return logs

    # ------------------------------------------------------------------ gating / publishing
    def maybe_publish(self, evaluate: Callable[[LoRAHeads], float]) -> bool:
        """Score candidate and served adapters on the same held-out data; promote if better."""
        if not self.cfg.gate:
            return False
        with torch.no_grad():
            cand, served = float(evaluate(self.model)), float(evaluate(self.serving))
        ok = cand > served + self.cfg.promote_margin
        self.stats.history.append({"t": time.time(), "candidate": cand, "served": served, "promoted": ok})
        if ok:
            self._promote(cand)
            self._bad = 0
            return True
        self.stats.rejected += 1
        self._bad = self._bad + 1 if cand < served - self.cfg.rollback_tolerance else 0
        if self._bad >= self.cfg.rollback_patience:
            self.model.load_adapter(self.serving.adapter_state())
            self._new_optimizer()
            self.stats.rollbacks += 1
            self._bad = 0
        return False

    def _promote(self, score):
        self.serving.load_adapter(self.model.adapter_state())
        self.publish(score)

    def publish(self, score=None) -> str | None:
        self.version += 1
        self.stats.published += 1
        if not self.publish_path:
            return None
        ck = self.serving.merged_checkpoint({"version": self.version, "score": score})
        vdir = os.path.join(os.path.dirname(os.path.abspath(self.publish_path)), "versions")
        os.makedirs(vdir, exist_ok=True)
        torch.save(self.serving.adapter_state(), os.path.join(vdir, f"adapter_v{self.version:04d}.pt"))
        atomic_save(ck, self.publish_path)
        with open(os.path.join(vdir, "log.jsonl"), "a") as f:
            f.write(json.dumps({"version": self.version, "score": score, "t": time.time()}) + "\n")
        return self.publish_path


@torch.no_grad()
def snips(model: LoRAHeads, items: list[Labeled], bias: Callable | None = None, temperature: float = 1.0) -> float:
    """Self-normalised importance-sampling estimate of the success rate ``model`` would get on the
    logged decisions ``items`` (Swaminathan & Joachims 2015).  The policy is
    ``softmax(head_logits / temperature + bias)``: ``bias(batch) -> [B, K]`` is the same logit
    correction (e.g. the outcome map) the policy adds when acting."""
    if not items:
        return 0.0
    b = to_tensors(items, next(model.parameters()).device)
    logits = model.candidate_logits(b["s"], b["cands"]) / temperature
    if bias is not None:
        logits = logits + bias(b)          # after the temperature, exactly as the policy acts
    pi = F.softmax(logits.masked_fill(~b["mask"], float("-inf")), -1)
    w = pi.gather(1, b["chosen"].unsqueeze(1)).squeeze(1) / b["logged_prob"].clamp(min=1e-6)
    return float((w * b["success"].float()).sum() / w.sum().clamp(min=1e-8))


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
