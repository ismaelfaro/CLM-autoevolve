"""The outcome map: a distribution of past successes and failures over (state, action) space.

Every labelled step the system lived through is kept as a point ``(state, action, outcome)`` in the
*frozen encoder's* space.  For a new decision, the map answers "when I (or a similar state) took
this action before, how often did it work?" with a kernel-weighted Beta posterior::

    w_i  = exp((cos(s, s_i) - 1) / tau_s) * exp((cos(a, a_i) - 1) / tau_a) * (1 + bonus * [task_i == task])
    p(a) = (a0 + sum_i w_i r_i) / (a0 + b0 + sum_i w_i),     n(a) = sum_i w_i  (evidence)

Exact revisits dominate (w = 1), similar states contribute less, and the ``task`` bonus lets a
task's own history count more than its neighbours' without ignoring them, so a new task borrows
from similar ones until it has history of its own.

The distribution enters CLM's contrast in two ways:

1. **At inference, instantly** (no training): ``bias()`` adds ``lambda * n/(n+kappa) * logit p`` to
   CLM's logits, so an action that keeps failing in this region is pushed down after a handful of
   failures, and an unknown region (n ~ 0) leaves CLM's prior alone.  Non-parametric episodic
   control (Blundell et al. 2016; Pritzel et al. 2017) and kNN-LM (Khandelwal et al. 2020) in
   CLM's terms.
2. **As training targets** (consolidation): ``soft_targets()`` turns the posterior over a
   decision's candidates into a distribution that the LoRA head is distilled towards
   (``objectives.map_distill``).  That gives every candidate a graded target, failures included,
   rather than one positive against in-batch negatives.  It's the complementary-learning-systems
   split (McClelland et al. 1995): a fast memory that learns from one episode, and a slow
   parametric model that generalises it.

Keys live in the frozen encoder space, so a head update never invalidates the map.
"""
from __future__ import annotations

import numpy as np
import torch


class OutcomeMap:
    def __init__(self, dim: int, capacity: int = 100_000, k: int = 64, tau_state: float = 0.05,
                 tau_action: float = 0.02, prior: tuple[float, float] = (1.0, 1.0), task_bonus: float = 1.0,
                 device: str = "cpu"):
        self.dim, self.capacity, self.k = dim, capacity, k
        self.tau_s, self.tau_a, self.a0, self.b0, self.task_bonus = tau_state, tau_action, *prior, task_bonus
        self.device = device
        self.S = torch.zeros(capacity, dim, device=device)
        self.A = torch.zeros(capacity, dim, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.w = torch.zeros(capacity, device=device)
        self.task = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.n = 0          # rows filled
        self.head = 0       # next row to overwrite (ring buffer: the newest experience wins)

    def __len__(self) -> int:
        return self.n

    def add(self, states, actions, outcomes, weights=None, tasks=None) -> None:
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device).reshape(-1, self.dim)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device).reshape(-1, self.dim)
        r = torch.as_tensor(np.asarray(outcomes), dtype=torch.float32, device=self.device).reshape(-1)
        w = (torch.ones_like(r) if weights is None
             else torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=self.device).reshape(-1))
        t = (torch.full_like(r, -1, dtype=torch.long) if tasks is None
             else torch.as_tensor(np.asarray(tasks), dtype=torch.long, device=self.device).reshape(-1))
        m = len(r)
        if m > self.capacity:
            s, a, r, w, t = (x[-self.capacity:] for x in (s, a, r, w, t))
            m = self.capacity
        rows = (self.head + torch.arange(m, device=self.device)) % self.capacity
        self.S[rows], self.A[rows], self.r[rows], self.w[rows], self.task[rows] = s, a, r, w, t
        self.head = (self.head + m) % self.capacity
        self.n = min(self.n + m, self.capacity)

    def add_labeled(self, items, task_of=None) -> None:
        """``experience.Labeled`` steps; ``task_of(decision) -> int`` (default ``decision.task``)."""
        if not items:
            return
        self.add([i.d.state for i in items], [i.d.candidates[i.d.chosen] for i in items],
                 [float(i.success) for i in items], [i.weight for i in items],
                 [(task_of(i.d) if task_of else (i.d.task if isinstance(i.d.task, int) else -1)) for i in items])

    @torch.no_grad()
    def query(self, s: torch.Tensor, cands: torch.Tensor, task: torch.Tensor | None = None):
        """``s`` [B, d], ``cands`` [B, K, d] -> (p_success [B, K], evidence [B, K])."""
        B, K = cands.shape[:2]
        if self.n == 0:
            prior = self.a0 / (self.a0 + self.b0)
            return torch.full((B, K), prior, device=s.device), torch.zeros(B, K, device=s.device)
        S, A, r, w, tk = (x[:self.n] for x in (self.S, self.A, self.r, self.w, self.task))
        sims = s.to(self.device).float() @ S.t()                                   # [B, n]
        vals, idx = sims.topk(min(self.k, self.n), dim=1)                          # [B, k]
        ws = torch.exp((vals - 1) / self.tau_s)                                    # [B, k]
        cos_a = torch.einsum("bkd,bjd->bkj", cands.to(self.device).float(), A[idx])  # [B, K, k]
        wk = ws.unsqueeze(1) * torch.exp((cos_a - 1) / self.tau_a) * w[idx].unsqueeze(1)
        if task is not None:
            same = (tk[idx] == task.to(self.device).view(-1, 1)) & (tk[idx] >= 0)
            wk = wk * (1 + self.task_bonus * same.float()).unsqueeze(1)
        succ = (wk * r[idx].unsqueeze(1)).sum(-1)
        tot = wk.sum(-1)
        p = (self.a0 + succ) / (self.a0 + self.b0 + tot)
        return p.to(s.device), tot.to(s.device)

    def bias(self, s, cands, task=None, strength: float = 2.0, kappa: float = 2.0) -> torch.Tensor:
        """Additive logit correction: ``strength * n/(n+kappa) * logit(p)``."""
        p, n = self.query(s, cands, task)
        p = p.clamp(1e-3, 1 - 1e-3)
        return strength * (n / (n + kappa)) * torch.log(p / (1 - p))

    def soft_targets(self, s, cands, mask, task=None, beta: float = 4.0, kappa: float = 2.0):
        """-> (target distribution over candidates [B, K], confidence [B] in [0, 1))."""
        p, n = self.query(s, cands, task)
        p = p.clamp(1e-3, 1 - 1e-3)
        logits = (beta * torch.log(p / (1 - p))).masked_fill(~mask, float("-inf"))
        conf = ((n / (n + kappa)) * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return torch.softmax(logits, -1), conf

    def task_profile(self, task: int) -> dict[str, float]:
        """Success rate and volume of one task's own history (for reporting)."""
        sel = self.task[:self.n] == task
        if not sel.any():
            return {"n": 0, "success": float("nan")}
        return {"n": int(sel.sum()), "success": float((self.r[:self.n][sel] * self.w[:self.n][sel]).sum()
                                                     / self.w[:self.n][sel].sum())}
