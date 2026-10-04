#!/usr/bin/env python3
"""Synthetic check of the online loop: does learning from failures beat ignoring outcomes?

A stand-in world (no GPU, no encoder): states and actions are fixed random vectors (the frozen
encoder's embeddings); the right action among K candidates is ``argmax s^T W* a``.  The base head
is pre-trained, like CLM, on gold pairs of a *related but shifted* rule, so it starts imperfect.
Episodes have L steps and end at the first wrong action (a T-Rex-style crash); only the episode
outcome is reported.  Every ``--update-every`` episodes the learner takes a few LoRA steps and the
gate decides, on held-out logged episodes, whether to publish.

Arms
  frozen        the base head, never updated
  all-positive  every executed step is a positive (reward ignored, as train/finetune.py does)
  all-positive-ungated  the same without the gate (what ignoring outcomes does when nothing stops it)
  success-only  successful steps are positives, failures dropped
  infonce       successes positive + failed actions as same-state hard negatives (terminal credit)
  ce            candidate-set CE on successes + unlikelihood on failures (terminal credit)
  ce-credit-all the same, but every step of a failed episode is labelled a failure
  bandit        clipped-ratio policy gradient on logged probabilities
  infonce+ce

Metrics: online success rate (prequential, over the last third of the stream), greedy accuracy
on unseen states under the new rule, and accuracy on the base rule (forgetting).

This only validates the mechanics on a toy; it is not evidence about CLM-8B itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoevolve import Decision, ExperienceBuffer, LearnerConfig, LoRAHeads, OnlineLearner, snips  # noqa: E402
from autoevolve import objectives as obj  # noqa: E402
from autoevolve.lora import random_checkpoint  # noqa: E402

D = 32


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class World:
    def __init__(self, seed, n_states=400, n_actions=300, k=4, length=3, shift=0.7, rank=6):
        r = np.random.default_rng(seed)
        self.states = unit(r.standard_normal((n_states, D))).astype(np.float32)
        self.actions = unit(r.standard_normal((n_actions, D))).astype(np.float32)
        low = lambda: r.standard_normal((D, rank)) @ r.standard_normal((rank, D)) / D  # noqa: E731
        w_new, w_other = low(), low()
        self.w_new = w_new
        self.w_base = math.cos(shift) * w_new + math.sin(shift) * w_other   # related, not equal
        self.k, self.length, self.r = k, length, r

    def draw(self, r=None, fresh=False):
        r = r or self.r
        s = unit(r.standard_normal(D)).astype(np.float32) if fresh else self.states[r.integers(len(self.states))]
        idx = r.choice(len(self.actions), self.k, replace=False)
        return s, self.actions[idx]

    @staticmethod
    def best(s, cands, w):
        return int(np.argmax(cands @ (w.T @ s)))


def pretrain_base(world, seed, steps=4000, n=100000):
    """Train full heads on gold decisions of the base rule (stands in for CLM pre/post-training):
    InfoNCE on (state, gold action) plus cross-entropy over each decision's candidate set."""
    r = np.random.default_rng(seed + 1)
    S, C, G = [], [], []
    for _ in range(n):
        s, c = world.draw(r, fresh=True)
        S.append(s); C.append(c); G.append(world.best(s, c, world.w_base))
    S, C, G = torch.tensor(np.stack(S)), torch.tensor(np.stack(C)), torch.tensor(G)
    A = C[torch.arange(n), G]
    mask = torch.ones(512, world.k, dtype=torch.bool)
    ok = torch.ones(512, dtype=torch.bool)
    ck = random_checkpoint(width=256, depth=3, proj=64, hidden=D, seed=seed)
    m = LoRAHeads(ck, rank=1)
    full = [p for n_, p in m.named_parameters() if "lora_" not in n_]
    for p in full:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(full, lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(S), (512,), generator=g)
        loss = obj.replay_infonce(m, S[idx], A[idx]) + obj.outcome_ce(m, S[idx], C[idx], mask, G[idx], ok)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in full:
        p.requires_grad_(False)
    ck = m.merged_checkpoint()
    ck["cfg"].pop("autoevolve")
    return ck, (S, A)


@torch.no_grad()
def probs(model, s, cands):
    return F.softmax(model.candidate_logits(torch.from_numpy(s)[None], torch.from_numpy(cands)[None]), -1)[0].numpy()


@torch.no_grad()
def greedy_acc(model, world, w, n=1000, seed=99):
    r = np.random.default_rng(seed)
    S, C = zip(*(world.draw(r, fresh=True) for _ in range(n)))
    S, C = np.stack(S), np.stack(C)
    lg = model.candidate_logits(torch.from_numpy(S), torch.from_numpy(C)).numpy()
    return float(np.mean([lg[i].argmax() == world.best(S[i], C[i], w) for i in range(n)]))


def anchor_set(world, seed, n=2000):
    r = np.random.default_rng(seed + 2)
    S, C = zip(*(world.draw(r, fresh=True) for _ in range(n)))
    return torch.tensor(np.stack(S)), torch.tensor(np.stack(C)), torch.ones(n, world.k, dtype=torch.bool)


ARMS = {
    "frozen": None,
    "all-positive": dict(losses=("infonce",), ignore_outcome=True),
    "all-positive-ungated": dict(losses=("infonce",), ignore_outcome=True, gate=False),
    "success-only": dict(losses=("infonce",), use_failures=False),
    "infonce": dict(losses=("infonce",)),
    "ce": dict(losses=("ce",)),
    "ce-credit-all": dict(losses=("ce",), credit="all"),
    "bandit": dict(losses=("bandit",)),
    "infonce+ce": dict(losses=("infonce", "ce")),
}


def run_arm(name, world, base_ck, replay, anchor, args, seed):
    spec = dict(ARMS[name] or {})
    credit = spec.pop("credit", "terminal")
    cfg = LearnerConfig(rank=args.rank, lr=args.lr, steps_per_update=args.steps, batch=args.batch,
                        replay_batch=256, kl_weight=args.kl, replay_weight=args.replay, **spec)
    learner = OnlineLearner(base_ck, None, cfg, replay=replay, anchor=anchor)
    train_buf = ExperienceBuffer(strategy=credit, seed=seed)
    val_buf = ExperienceBuffer(strategy="terminal", seed=seed)
    r = np.random.default_rng(seed + 10)
    group_of: dict[bytes, int] = {}
    outcomes = []

    def gate_score(model):
        """SNIPS success estimate on held-out logged steps: only what the system observed, never W*."""
        return snips(model, val_buf.items[-2000:])

    for ep in range(args.episodes):
        eid = f"{ep}"
        ok = 1.0
        buf = val_buf if ep % 5 == 4 else train_buf
        for t in range(world.length):
            s, cands = world.draw()
            p = probs(learner.serving, s, cands)
            choice = int(r.choice(len(p), p=p / p.sum())) if args.explore else int(p.argmax())
            g = group_of.setdefault(s.tobytes(), len(group_of))      # decision point = state
            buf.record(Decision(eid, t, s, cands, choice, p, g))
            if choice != world.best(s, cands, world.w_new):
                ok = 0.0
                break
        buf.close(eid, ok)
        outcomes.append(ok)
        if ARMS[name] and (ep + 1) % args.update_every == 0 and len(train_buf):
            learner.update(train_buf)
            learner.maybe_publish(gate_score)
    tail = outcomes[-len(outcomes) // 3:]
    return {"arm": name, "seed": seed, "online_success_last_third": float(np.mean(tail)),
            "online_success_first_third": float(np.mean(outcomes[:len(outcomes) // 3])),
            "new_rule_acc": greedy_acc(learner.serving, world, world.w_new),
            "base_rule_acc": greedy_acc(learner.serving, world, world.w_base),
            "published": learner.stats.published, "rejected": learner.stats.rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--update-every", type=int, default=25)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--replay", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--explore", action=argparse.BooleanOptionalAction, default=True,
                    help="sample from the answer distribution (default) instead of argmax")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        world = World(seed)
        base_ck, replay = pretrain_base(world, seed)
        anchor = anchor_set(world, seed)
        for arm in args.arms:
            world.r = np.random.default_rng(seed + 100)      # identical episode stream per arm
            row = run_arm(arm, world, base_ck, replay, anchor, args, seed)
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("\narm               online(1st/3)  online(last/3)  new-rule acc  base-rule acc  published/rejected")
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm]
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        sd = lambda k: np.std([r[k] for r in rs])  # noqa: E731
        print(f"{arm:16s}  {m('online_success_first_third'):.3f}          "
              f"{m('online_success_last_third'):.3f}±{sd('online_success_last_third'):.3f}    "
              f"{m('new_rule_acc'):.3f}±{sd('new_rule_acc'):.3f}   {m('base_rule_acc'):.3f}±{sd('base_rule_acc'):.3f}   "
              f"{m('published'):.0f}/{m('rejected'):.0f}")
    if args.out:
        json.dump({"args": vars(args), "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
