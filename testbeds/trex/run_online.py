#!/usr/bin/env python3
"""T-Rex testbed: does CLM get better at a task from its own successes and failures?

    python testbeds/trex/run_online.py                      # CPU: hash encoder + stand-in base head
    python testbeds/trex/run_online.py --encoder clm --ckpt "$(clm-download)"   # real CLM-8B (GPU)

Protocol, per seed and per arm (every arm sees the same courses):

1. **Base head.**  With ``--encoder hash`` a stand-in for "CLM before this task": heads trained
   on the planner's choices on an *easy* course (single cacti only: no groups, no birds).  It
   knows to jump cacti, not how to time groups or what to do about birds.  With ``--encoder clm``
   the released CLM head is the base, zero-shot.
2. **Online phase** (``--train-frames`` on one training course, many lives): the agent acts by
   sampling from its answer distribution; each obstacle encounter is an episode, cleared =
   success, crash = failure; the learner updates the LoRA every ``--update-every`` episodes and
   the gate promotes it only if it beats the served adapter on held-out logged episodes.
3. **Evaluation** on unseen courses (``--eval-seeds``): greedy, no learning, outcome map frozen.

Arms
  frozen                 base head only
  map                    + outcome map as an instant logit bias (no gradient learning)
  lora                   LoRA trained on outcomes (CE on successes, unlikelihood on failures), gated
  lora+map               both: the map acts at once, the LoRA learns
  lora+map+distill       + the LoRA is also distilled towards the map's success distribution
  lora-ungated           LoRA without the gate (every update served)
  all-positive-ungated   every executed step is a positive, no gate (reward ignored, as
                         CLM's train/finetune.py does today; the self-confirmation control)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from autoevolve import (ClmEncoder, Decision, ExperienceBuffer, HashEncoder, LearnerConfig, LoRAHeads,  # noqa: E402
                        OnlineLearner, OutcomeMap, snips)
from autoevolve import objectives as obj  # noqa: E402
from autoevolve.lora import random_checkpoint  # noqa: E402
from testbeds.trex.env import ACTIONS, OPTIONS, EasyCourse, TRexEnv  # noqa: E402
from testbeds.trex.planner import snapshot  # noqa: E402

ARMS = {
    "frozen": dict(learn=False),
    "map": dict(map=True),
    "lora": dict(losses=("ce",)),
    "lora+map": dict(losses=("ce",), map=True),
    "lora+map+distill": dict(losses=("ce", "distill"), map=True),
    "lora-ungated": dict(losses=("ce",), gate=False),
    "all-positive-ungated": dict(losses=("ce",), gate=False, ignore_outcome=True),
}


# --------------------------------------------------------------------------- base head
class _PlannerAgent:
    """Plays with the planner's best action (plus some noise for coverage), recording labels."""

    def __init__(self, env, noise, rng):
        self.env, self.noise, self.rng, self.data = env, noise, rng, []

    def decide(self, text, label, ep):
        best = self.env.planner.plan(snapshot(self.env.game, self.env.held), (0, 0), (self.env.k, self.env.k)).best
        self.data.append((text, ACTIONS.index(best)))
        return self.rng.choice(ACTIONS) if self.rng.random() < self.noise else best

    def outcome(self, ep, ok):
        pass

    def discard(self, ep):
        pass


def build_base(enc, seed, frames=40_000, steps=1500):
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(4):
        env = TRexEnv(1000 + 10 * seed + i, course=EasyCourse(seed * 10 + i), oracle=True)
        agent = _PlannerAgent(env, 0.15, rng)
        env.play(agent, frames // 4)
        texts += [t for t, _ in agent.data]; labels += [y for _, y in agent.data]
    S = torch.tensor(enc.embed(texts))
    opts = torch.tensor(enc.embed([OPTIONS[a] for a in ACTIONS]))
    C = opts.unsqueeze(0).expand(len(S), -1, -1).contiguous()
    Y = torch.tensor(labels)
    ck = random_checkpoint(width=256, depth=3, proj=64, hidden=S.shape[1], seed=seed)
    m = LoRAHeads(ck, rank=1)
    full = [p for n, p in m.named_parameters() if "lora_" not in n]
    for p in full:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(full, lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(512, 3, dtype=torch.bool)
    ok = torch.ones(512, dtype=torch.bool)
    for _ in range(steps):
        idx = torch.randint(len(S), (512,), generator=g)
        loss = obj.outcome_ce(m, S[idx], C[idx], mask, Y[idx], ok)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in full:
        p.requires_grad_(False)
    with torch.no_grad():
        acc = float((m.candidate_logits(S, C).argmax(-1) == Y).float().mean())
    ck = m.merged_checkpoint()
    ck["cfg"].pop("autoevolve")
    keep = torch.randperm(len(S), generator=g)[:4000]
    anchor = (S[keep], C[keep], torch.ones(len(keep), 3, dtype=torch.bool))
    return ck, anchor, {"base_examples": len(S), "base_train_acc": acc,
                        "label_mix": {a: labels.count(i) / len(labels) for i, a in enumerate(ACTIONS)}}


# --------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, enc, head, learner=None, omap=None, *, learn=True, greedy=False, eps=0.05,
                 temperature=1.0, map_strength=2.0, update_every=40, window=8, gate_window=3000, seed=0):
        self.enc, self.head, self.learner, self.omap = enc, head, learner, omap
        self.learn, self.greedy, self.eps, self.t, self.map_strength = learn, greedy, eps, temperature, map_strength
        self.update_every, self.gate_window = update_every, gate_window
        self.opts_np = enc.embed([OPTIONS[a] for a in ACTIONS]).astype(np.float32)
        self.opts = torch.from_numpy(self.opts_np)[None]
        self.train = ExperienceBuffer(strategy="terminal", window=window, seed=seed)
        self.val = ExperienceBuffer(strategy="terminal", window=window, seed=seed + 1)
        self.rng = np.random.default_rng(seed)
        self.groups: dict[str, int] = {}
        self.tasks: dict[str, int] = {}
        self.closed = 0
        self.step = 0

    def actor(self):
        return self.learner.serving if self.learner else self.head

    def _bias(self, b):
        return self.omap.bias(b["s"], b["cands"], b["task"], self.map_strength)

    @torch.no_grad()
    def decide(self, text, label, ep):
        s = self.enc.embed([text])[0].astype(np.float32)
        st = torch.from_numpy(s)[None]
        tid = self.tasks.setdefault(label, len(self.tasks))
        lg = self.actor().candidate_logits(st, self.opts)[0]
        if self.omap is not None:
            lg = lg + self.omap.bias(st, self.opts, torch.tensor([tid]), self.map_strength)[0]
        p = F.softmax(lg / self.t, -1).numpy()
        p = (1 - self.eps) * p + self.eps / len(p)
        a = int(p.argmax()) if self.greedy else int(self.rng.choice(len(p), p=p / p.sum()))
        if self.learn:
            buf = self.val if zlib.crc32(ep.encode()) % 5 == 0 else self.train
            self.step += 1
            buf.record(Decision(ep, self.step, s, self.opts_np, a, p, self.groups.setdefault(text, len(self.groups)),
                                task=tid))
        return ACTIONS[a]

    def outcome(self, ep, ok):
        if not self.learn:
            return
        if zlib.crc32(ep.encode()) % 5 == 0:
            self.val.close(ep, float(ok))
            return
        labeled = self.train.close(ep, float(ok))
        if self.omap is not None:
            self.omap.add_labeled(labeled)
        self.closed += 1
        if self.learner and self.closed % self.update_every == 0:
            self.learner.update(self.train)
            bias = self._bias if self.omap is not None else None
            self.learner.maybe_publish(lambda m: snips(m, self.val.items[-self.gate_window:], bias, self.t))

    def discard(self, ep):
        self.train.discard(ep); self.val.discard(ep)


# --------------------------------------------------------------------------- one arm
def summarize(t):
    enc = t.cleared + t.deaths
    return {"frames": t.frames, "deaths": t.deaths, "cleared": t.cleared,
            "deaths_per_min": round(t.deaths / (t.frames / 3600), 3),
            "clear_rate": round(t.cleared / enc, 4) if enc else None,
            "oracle_agreement": round(t.oracle_agree / t.oracle_seen, 4) if t.oracle_seen else None,
            "by_type": {k: [v[0], v[1]] for k, v in sorted(t.by_type.items())}}


def run_arm(name, enc, base_ck, anchor, args, seed):
    spec = dict(ARMS[name])
    learn, use_map = spec.pop("learn", True), spec.pop("map", False)
    omap = OutcomeMap(dim=anchor[0].shape[1], k=args.map_k, tau_state=args.map_tau) if use_map else None
    learner = None
    if learn and "losses" in spec:
        cfg = LearnerConfig(rank=args.rank, lr=args.lr, steps_per_update=args.steps, batch=args.batch,
                            replay_batch=256, kl_weight=args.kl, **spec)
        learner = OnlineLearner(base_ck, None, cfg, anchor=anchor, outcome_map=omap, task_of=lambda d: d.task)
    head = LoRAHeads(base_ck, rank=1).eval()
    agent = Agent(enc, head, learner, omap, learn=learn, update_every=args.update_every, window=args.window,
                  map_strength=args.map_strength, seed=seed)
    torch.manual_seed(seed)
    env = TRexEnv(seed, frames_per_decision=args.k)
    phases = []
    for _ in range(3):
        before = (env.tally.deaths, env.tally.cleared, env.tally.frames)
        env.play(agent, args.train_frames // 3)
        d, c, f = (env.tally.deaths - before[0], env.tally.cleared - before[1], env.tally.frames - before[2])
        phases.append({"deaths_per_min": round(d / (f / 3600), 3), "clear_rate": round(c / max(1, c + d), 4)})
    # evaluation: greedy, frozen, unseen courses
    agent.learn, agent.greedy, agent.eps = False, True, 0.0
    evals = []
    for es in args.eval_seeds:
        e = TRexEnv(es, frames_per_decision=args.k, oracle=True)
        e.play(agent, args.eval_frames)
        evals.append(summarize(e.tally))
    tot = {k: sum(e[k] for e in evals) for k in ("deaths", "cleared", "frames")}
    agree = [e["oracle_agreement"] for e in evals if e["oracle_agreement"] is not None]
    by_type: dict[str, list[int]] = {}
    for e in evals:
        for k, v in e["by_type"].items():
            acc = by_type.setdefault(k, [0, 0]); acc[0] += v[0]; acc[1] += v[1]
    ls = learner.stats if learner else None
    return {"arm": name, "seed": seed, "train_phases": phases, "train": summarize(env.tally),
            "eval": {"deaths_per_min": round(tot["deaths"] / (tot["frames"] / 3600), 3),
                     "clear_rate": round(tot["cleared"] / max(1, tot["cleared"] + tot["deaths"]), 4),
                     "oracle_agreement": round(float(np.mean(agree)), 4) if agree else None,
                     "by_type": by_type},
            "learner": ({"updates": ls.updates, "published": ls.published, "rejected": ls.rejected,
                         "rollbacks": ls.rollbacks} if ls else None),
            "map_rows": len(omap) if omap is not None else 0}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["hash", "clm"], default="hash")
    ap.add_argument("--dim", type=int, default=256, help="hash encoder width")
    ap.add_argument("--ckpt", help="--encoder clm: base CLM checkpoint (e.g. $(clm-download))")
    ap.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--train-frames", type=int, default=60_000)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--eval-frames", type=int, default=7200)
    ap.add_argument("--k", type=int, default=4, help="frames per decision")
    ap.add_argument("--update-every", type=int, default=40, help="training episodes per LoRA update")
    ap.add_argument("--window", type=int, default=8, help="terminal credit window (decisions)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--map-k", type=int, default=64)
    ap.add_argument("--map-tau", type=float, default=0.05)
    ap.add_argument("--map-strength", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "trex_online.json"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    rows, bases = [], {}
    for seed in args.seeds:
        t0 = time.time()
        if args.encoder == "clm":
            enc = ClmEncoder(args.emb_url)
            base_ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            S = torch.tensor(enc.embed([f"Dino runner game. Nothing ahead. Speed {v}." for v in range(6, 14)]))
            opts = torch.tensor(enc.embed([OPTIONS[a] for a in ACTIONS]))
            anchor = (S, opts.unsqueeze(0).expand(len(S), -1, -1).contiguous(), torch.ones(len(S), 3, dtype=torch.bool))
            bases[seed] = {"base": args.ckpt}
        else:
            enc = HashEncoder(args.dim)
            base_ck, anchor, info = build_base(enc, seed)
            bases[seed] = info
            print(f"[base] seed {seed}: {json.dumps(info)} ({time.time() - t0:.0f}s)", flush=True)
        for arm in args.arms:
            t1 = time.time()
            row = run_arm(arm, enc, base_ck, anchor, args, seed)
            row["seconds"] = round(time.time() - t1, 1)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("arm", "seed", "train_phases", "eval", "learner", "seconds")}),
                  flush=True)

    print("\narm                    train clear% (1st→last third)   eval deaths/min   eval clear%   oracle agree")
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm]
        m = lambda f: float(np.mean([f(r) for r in rs]))  # noqa: E731
        s = lambda f: float(np.std([f(r) for r in rs]))  # noqa: E731
        print(f"{arm:22s} {100 * m(lambda r: r['train_phases'][0]['clear_rate']):5.1f} → "
              f"{100 * m(lambda r: r['train_phases'][-1]['clear_rate']):5.1f}"
              f"            {m(lambda r: r['eval']['deaths_per_min']):6.2f}±{s(lambda r: r['eval']['deaths_per_min']):.2f}"
              f"      {100 * m(lambda r: r['eval']['clear_rate']):5.1f}±{100 * s(lambda r: r['eval']['clear_rate']):.1f}"
              f"     {100 * m(lambda r: r['eval']['oracle_agreement'] or 0):5.1f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "bases": bases, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
