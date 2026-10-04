#!/usr/bin/env python3
"""Phase 0: retrain CLM's DeepSWE verifier head from success *and* failure.

CLM's ``train/finetune.py --task clm`` trains on every step of every trajectory as a positive
(state, action) pair; the per-step ``reward`` in the embedding metadata is only used to stratify
folds.  This script trains on the same embeddings, warm-started from the same checkpoint, and
writes a head in CLM's checkpoint format, so CLM's own ``evaluation/bon_eval.py`` scores it
unchanged against the published 31/38 (81.6%) on heldout-38.

Objectives (``--objective``)
  clm              CLM's loss, reproduced: bidirectional InfoNCE over all steps, same-(task, step)
                   rows masked.  The control: same code path, reward ignored.
  success          the same loss on the steps of passing trajectories only.
  outcome          CLM's loss on all steps + a task-level contrast between trajectories: within each
                   training task, every passing trajectory should outscore every failing one, with
                   the score bon_eval selects by (mean step cosine over the final ``--window`` steps):
                   ``softplus(-(score_pass - score_fail) / tau)``.  This is the success/failure
                   distribution per task used as the contrast.
  outcome-success  ``success`` + the trajectory contrast.

``--lora-rank R`` trains rank-R adapters on the heads (merged on export); ``--lora-rank 0`` fine-tunes
the heads in full.  Model selection uses best-of-N on held-in validation tasks (never heldout-38).

    hf download Contrastive-LM/deepswe-clm-heads-8k --local-dir heads/deepswe
    python <CLM>/preprocessing/hf_embeddings.py download Contrastive-LM/deepswe-clm-train-embeddings-8k --out data/deepswe_train
    python experiments/deepswe_outcome.py --emb-dir data/deepswe_train --init-ckpt "$(clm-download)" \\
        --holdout-tasks heads/deepswe/heldout_tasks.json --objective outcome --out-dir runs/deepswe/outcome
    python <CLM>/evaluation/bon_eval.py --hf-dataset Contrastive-LM/deepswe-clm-embeddings-8k \\
        --checkpoint runs/deepswe/outcome/best_head.pt --tasks-file heads/deepswe/heldout_tasks.json --n 4 --window 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoevolve.lora import LoRAHeads, atomic_save  # noqa: E402

OBJECTIVES = ("clm", "success", "outcome", "outcome-success")


# --------------------------------------------------------------------------- data
class Data:
    """One embedding dir (``state_embeddings.pt``, ``action_embeddings.pt``, ``metadata.json``),
    kept on the host in fp16; trajectories ordered by ``step_idx``."""

    def __init__(self, emb_dir: str):
        self.S = torch.load(os.path.join(emb_dir, "state_embeddings.pt"), map_location="cpu")
        self.A = torch.load(os.path.join(emb_dir, "action_embeddings.pt"), map_location="cpu")
        self.samples = json.load(open(os.path.join(emb_dir, "metadata.json")))["samples"]
        if not (len(self.S) == len(self.A) == len(self.samples)):
            raise ValueError(f"{emb_dir}: embedding and metadata lengths differ")
        steps = defaultdict(list)
        for i, s in enumerate(self.samples):
            steps[s["trajectory_id"]].append((s["step_idx"], i))
        self.traj = {t: [i for _, i in sorted(v)] for t, v in steps.items()}
        self.task_of = {t: self.samples[v[0]]["task_id"] for t, v in self.traj.items()}
        self.group_of = {t: (self.task_of[t], self.samples[v[0]].get("config") or "default")
                         for t, v in self.traj.items()}
        self.passed = {t: float(self.samples[v[0]].get("reward") or 0.0) > 0.5 for t, v in self.traj.items()}
        codes: dict = {}
        self.code = torch.tensor([codes.setdefault((s["task_id"], s["step_idx"]), len(codes)) for s in self.samples])

    def steps_of(self, trajs, final: int | None = None):
        return [i for t in trajs for i in (self.traj[t][-final:] if final else self.traj[t])]


def split_tasks(data: Data, holdout: set, val_frac: float, seed: int):
    tasks = sorted({t for t in data.task_of.values() if t not in holdout})
    random.Random(seed).shuffle(tasks)
    n_val = min(len(tasks) - 1, max(1, round(val_frac * len(tasks))))
    return set(tasks[n_val:]), set(tasks[:n_val])


# --------------------------------------------------------------------------- losses
def clm_loss(model, s, a, codes):
    """train/finetune.py's _clm_loss: bidirectional InfoNCE, same-(task, step) rows masked."""
    zs, za = model.encode_states(s), model.encode_actions(a)
    logits = model.scale * zs @ za.t()
    labels = torch.arange(len(s), device=s.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    fwd = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    bwd = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (fwd + bwd) / 2


def trajectory_scores(model, data, trajs, window, device):
    """bon_eval's selection score: mean step cosine over each trajectory's final ``window`` steps."""
    idx, owner = [], []
    for k, t in enumerate(trajs):
        steps = data.traj[t][-window:]
        idx += steps; owner += [k] * len(steps)
    idx_t = torch.tensor(idx)
    s, a = data.S[idx_t].to(device).float(), data.A[idx_t].to(device).float()
    cos = (model.encode_states(s) * model.encode_actions(a)).sum(-1)
    owner_t = torch.tensor(owner, device=device)
    tot = torch.zeros(len(trajs), device=device).index_add_(0, owner_t, cos)
    cnt = torch.zeros(len(trajs), device=device).index_add_(0, owner_t, torch.ones_like(cos))
    return tot / cnt


def pair_loss(model, data, groups, window, tau, device, max_traj=16, rng=None):
    """Within each sampled (task, config) group: every passing trajectory above every failing one."""
    losses = []
    for trajs in groups:
        trajs = list(trajs)
        if rng and len(trajs) > max_traj:
            trajs = rng.sample(trajs, max_traj)
        p = [t for t in trajs if data.passed[t]]
        f = [t for t in trajs if not data.passed[t]]
        if not p or not f:
            continue
        sc = trajectory_scores(model, data, p + f, window, device)
        diff = sc[:len(p)].unsqueeze(1) - sc[len(p):].unsqueeze(0)
        losses.append(F.softplus(-diff / tau).mean())
    return torch.stack(losses).mean() if losses else None


# --------------------------------------------------------------------------- evaluation
def best_of_n(candidates, n):
    """Exact expected reward of picking the top score among a uniform N-subset (ties uniform);
    the same estimator as CLM's evaluation/bon_eval.py."""
    blocks = defaultdict(list)
    for score, reward in candidates:
        blocks[score].append(reward)
    choose = lambda c: math.comb(c, n) if c >= n else 0  # noqa: E731
    denom = choose(len(candidates))
    selected, lower = 0.0, 0
    for _, rewards in sorted(blocks.items()):
        prob = (choose(lower + len(rewards)) - choose(lower)) / denom
        selected += prob * sum(rewards) / len(rewards)
        lower += len(rewards)
    return selected


@torch.no_grad()
def evaluate(model, data, groups, window, n, device):
    model.eval()
    sel, mixed, auc_hits, auc_n = 0.0, 0, 0.0, 0
    for trajs in groups.values():
        trajs = sorted(trajs)
        sc = trajectory_scores(model, data, trajs, window, device).tolist()
        sel += best_of_n([(s, int(data.passed[t])) for s, t in zip(sc, trajs)], min(n, len(trajs)))
        p = [s for s, t in zip(sc, trajs) if data.passed[t]]
        f = [s for s, t in zip(sc, trajs) if not data.passed[t]]
        if p and f:
            mixed += 1
            auc_hits += sum((x > y) + 0.5 * (x == y) for x in p for y in f)
            auc_n += len(p) * len(f)
    model.train()
    return {"bon": sel / max(1, len(groups)), "pair_auc": auc_hits / max(1, auc_n), "mixed_groups": mixed,
            "groups": len(groups)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True, help="training embedding dir (hf_embeddings.py download output)")
    ap.add_argument("--init-ckpt", required=True, help="warm start, e.g. $(clm-download)")
    ap.add_argument("--holdout-tasks", default=None, help="tasks never trained on (heldout_tasks.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--objective", choices=OBJECTIVES, default="outcome")
    ap.add_argument("--lora-rank", type=int, default=16, help="0 = full head fine-tune")
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="default: one pass over the training steps")
    ap.add_argument("--batch", type=int, default=512, help="InfoNCE rows per step")
    ap.add_argument("--pair-groups", type=int, default=8, help="(task, config) groups per step for the pair loss")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--pair-tau", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=12, help="final steps scored, as bon_eval --window")
    ap.add_argument("--n", type=int, default=4, help="best-of-N budget for validation, as bon_eval --n")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 (LoRA) / 2e-4 (full)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = Data(args.emb_dir)
    raw = json.load(open(args.holdout_tasks)) if args.holdout_tasks else []
    holdout = set(raw.get("tasks", raw.get("heldout_tasks", [])) if isinstance(raw, dict) else raw)
    train_tasks, val_tasks = split_tasks(data, holdout, args.val_frac, args.seed)
    groups = defaultdict(list)
    for t, g in data.group_of.items():
        groups[g].append(t)
    train_groups = {g: v for g, v in groups.items() if g[0] in train_tasks}
    val_groups = {g: v for g, v in groups.items() if g[0] in val_tasks}
    pair_groups = [v for v in train_groups.values()
                   if any(data.passed[t] for t in v) and not all(data.passed[t] for t in v)]
    use_success = args.objective in ("success", "outcome-success")
    use_pairs = args.objective in ("outcome", "outcome-success")
    train_trajs = [t for v in train_groups.values() for t in v if data.passed[t] or not use_success]
    rows = torch.tensor(data.steps_of(train_trajs))
    print(f"[phase0] {len(data.samples)} steps, {len(data.traj)} trajectories | tasks: train {len(train_tasks)}, "
          f"val {len(val_tasks)}, holdout {len(holdout)} | InfoNCE rows {len(rows)} | "
          f"mixed train groups {len(pair_groups)} | objective {args.objective}", flush=True)

    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    full = args.lora_rank == 0
    model = LoRAHeads(ck, rank=max(1, args.lora_rank), alpha=args.lora_alpha, train_scale=full).to(device)
    if full:
        for n_, p in model.named_parameters():
            p.requires_grad_("lora_" not in n_)
    params = model.trainable_parameters()
    lr = args.lr or (2e-4 if full else 1e-3)
    opt = torch.optim.AdamW(params, lr=lr)
    steps = args.steps_per_epoch or max(1, len(rows) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps * args.epochs, pct_start=0.1)
    print(f"[phase0] {'full heads' if full else f'LoRA r={args.lora_rank}'}: "
          f"{sum(p.numel() for p in params):,} trainable | {steps} steps/epoch | lr {lr}", flush=True)

    def export(epoch, metrics, name):
        blob = model.merged_checkpoint({"objective": args.objective, "epoch": epoch})
        blob["cfg"].update({"init_ckpt": os.path.basename(args.init_ckpt), "train_dir": args.emb_dir})
        blob["metrics"], blob["epoch"] = metrics, epoch
        atomic_save(blob, os.path.join(args.out_dir, name))

    m0 = evaluate(model, data, val_groups, args.window, args.n, device)
    history = [{"epoch": 0, **m0}]
    best, best_ep, bad = m0["bon"], 0, 0
    export(0, m0, "best_head.pt")
    print(f"[phase0] epoch 0 (init) val {json.dumps(m0)}", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = rows[torch.randperm(len(rows))]
        tot = {"infonce": 0.0, "pair": 0.0}
        for k in range(steps):
            idx = perm[(k * args.batch) % len(perm):][:args.batch]
            if len(idx) < 2:
                idx = perm[:args.batch]
            loss = clm_loss(model, data.S[idx].to(device).float(), data.A[idx].to(device).float(),
                            data.code[idx].to(device))
            tot["infonce"] += loss.item()
            if use_pairs and pair_groups:
                pl = pair_loss(model, data, rng.sample(pair_groups, min(args.pair_groups, len(pair_groups))),
                               args.window, args.pair_tau, device, rng=rng)
                if pl is not None:
                    loss = loss + args.pair_weight * pl
                    tot["pair"] += pl.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        m = evaluate(model, data, val_groups, args.window, args.n, device)
        m.update({k: v / steps for k, v in tot.items()}, epoch=ep, minutes=round((time.time() - t0) / 60, 2))
        history.append(m)
        print(f"[phase0] epoch {ep} {json.dumps(m)}", flush=True)
        export(ep, m, "final_head.pt")
        if m["bon"] > best + 1e-6:
            best, best_ep, bad = m["bon"], ep, 0
            export(ep, m, "best_head.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[phase0] early stop at epoch {ep} (best {best_ep})", flush=True)
                break
    summary = {"args": vars(args), "best_epoch": best_ep, "best_val_bon": best, "history": history}
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print(f"[phase0] best epoch {best_ep}: val best-of-{args.n} {best:.4f} -> {args.out_dir}/best_head.pt", flush=True)


if __name__ == "__main__":
    main()
