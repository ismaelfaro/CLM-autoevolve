"""Training signals from success and failure for a CLM scorer.

All losses take encoder embeddings (the frozen Qwen3-8B side) and a ``LoRAHeads`` model.

* ``outcome_infonce``   CLM's bidirectional in-batch InfoNCE, but only successful steps are
                        positives, and failed actions taken in the *same* state group stay in the
                        denominator as hard negatives (CLM's mid-training hard-negative form).
                        Other successes of the same group are masked, as ``train/finetune.py``
                        masks same ``(task_id, step_idx)`` rows.
* ``outcome_ce``        On the logged candidate set of each decision: cross-entropy towards the
                        executed action when it succeeded, unlikelihood ``-log(1 - p)`` when it
                        failed (Welleck et al. 2019).  Unlike the contrastive form it needs no
                        success at the same state to learn from a failure.
* ``pairwise_dpo``      DPO-style preference on (state, winner, loser), anchored to the frozen
                        base head: use it when one state has both a success and a failure.
* ``bandit_ppo``        Contextual-bandit policy gradient for typed Choice decisions: CLM's answer
                        is a softmax policy over the candidates, the logged probabilities are the
                        behaviour policy, and the clipped ratio keeps updates near it.
* ``anchor_kl``         KL(base || current) over candidate sets from a replay set: the
                        anti-forgetting term (with replay of the original training pairs).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .lora import LoRAHeads


def outcome_infonce(model: LoRAHeads, s: torch.Tensor, a: torch.Tensor, success: torch.Tensor,
                    group: torch.Tensor, weight: torch.Tensor | None = None,
                    hard_negative_boost: float = 0.0) -> torch.Tensor:
    """``s``, ``a``: [B, d]; ``success``: [B] bool; ``group``: [B] int (same state / decision point).

    Rows with ``success`` are anchors.  Columns: every action in the batch, except other
    successes of the anchor's group (they are also right, so not negatives).  Failed actions of
    the anchor's own group are the hard negatives; ``hard_negative_boost`` adds ``log(1 + boost)``
    to their logits (importance-weighting the hardest negatives, Robinson et al. 2021).
    ``weight`` scales each anchor's term (e.g. an advantage or a credit weight).
    """
    pos = success.bool()
    if pos.sum() == 0:
        return s.new_zeros(())
    zs, za = model.encode_states(s), model.encode_actions(a)
    logits = model.scale * zs @ za.t()
    same = group.unsqueeze(0) == group.unsqueeze(1)
    eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
    false_neg = same & pos.unsqueeze(0) & pos.unsqueeze(1) & ~eye     # other right answers
    hard = same & ~pos.unsqueeze(0)                                   # failed actions, same state
    if hard_negative_boost > 0:
        logits = logits + hard.float() * torch.log1p(torch.tensor(hard_negative_boost, device=s.device))
    labels = torch.arange(len(s), device=s.device)
    fwd = F.cross_entropy(logits.masked_fill(false_neg, float("-inf"))[pos], labels[pos], reduction="none")
    # action -> state: a successful action should pick out its own state among the batch's states;
    # every row of the same group *is* that state (failed tries included), so none is a negative
    bwd = F.cross_entropy(logits.t().masked_fill(same & ~eye, float("-inf"))[pos], labels[pos],
                          reduction="none")
    per = (fwd + bwd) / 2
    if weight is not None:
        w = weight[pos].float()
        return (per * w).sum() / w.sum().clamp(min=1e-8)
    return per.mean()


def outcome_ce(model: LoRAHeads, s: torch.Tensor, cands: torch.Tensor, mask: torch.Tensor,
               chosen: torch.Tensor, success: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """``-log p(chosen)`` for successes, ``-log(1 - p(chosen))`` for failures, over each decision's
    own candidates (``cands`` [B, K, d] padded, ``mask`` [B, K])."""
    logp = F.log_softmax(model.candidate_logits(s, cands).masked_fill(~mask, float("-inf")), -1)
    lp = logp.gather(1, chosen.unsqueeze(1)).squeeze(1)
    neg = torch.log1p(-lp.exp().clamp(max=1 - 1e-6))
    per = -torch.where(success.bool(), lp, neg)
    if weight is not None:
        return (per * weight).sum() / weight.sum().clamp(min=1e-8)
    return per.mean()


def pairwise_dpo(model: LoRAHeads, s: torch.Tensor, a_win: torch.Tensor, a_lose: torch.Tensor,
                 beta: float = 0.1) -> torch.Tensor:
    """``-log sigmoid(beta * [(f(s,a+) - f_ref(s,a+)) - (f(s,a-) - f_ref(s,a-))])``, f = scaled cosine."""
    with torch.no_grad(), model.reference():
        ref = model.pair_logits(s, a_win) - model.pair_logits(s, a_lose)
    cur = model.pair_logits(s, a_win) - model.pair_logits(s, a_lose)
    return -F.logsigmoid(beta * (cur - ref)).mean()


def bandit_ppo(model: LoRAHeads, s: torch.Tensor, cands: torch.Tensor, mask: torch.Tensor,
               chosen: torch.Tensor, logged_prob: torch.Tensor, advantage: torch.Tensor,
               clip: float = 0.2, temperature: float = 1.0) -> torch.Tensor:
    """``cands``: [B, K, d] padded, ``mask``: [B, K] valid; ``chosen``: [B] index executed;
    ``logged_prob``: [B] probability the serving head gave it; ``advantage``: [B] reward - baseline.

    Only the executed action's outcome is observed, so this is off-policy learning from logged
    bandit feedback (Swaminathan & Joachims 2015); the PPO clip bounds the importance ratio.
    """
    logits = model.candidate_logits(s, cands) / temperature
    logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), -1)
    lp = logp.gather(1, chosen.unsqueeze(1)).squeeze(1)
    ratio = torch.exp(lp - logged_prob.clamp(min=1e-6).log())
    adv = advantage.float()
    return -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()


def anchor_kl(model: LoRAHeads, s: torch.Tensor, cands: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean KL(pi_base || pi_current) of the answer distributions over replayed candidate sets."""
    with torch.no_grad(), model.reference():
        ref = F.log_softmax(model.candidate_logits(s, cands).masked_fill(~mask, float("-inf")), -1)
    cur = F.log_softmax(model.candidate_logits(s, cands).masked_fill(~mask, float("-inf")), -1)
    p = ref.exp()
    return (p * (ref - cur)).masked_fill(~mask, 0.0).sum(-1).mean()


def replay_infonce(model: LoRAHeads, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Plain CLM InfoNCE on replayed (state, gold action) pairs from the original training mix."""
    group = torch.arange(len(s), device=s.device)
    return outcome_infonce(model, s, a, torch.ones(len(s), dtype=torch.bool, device=s.device), group)
