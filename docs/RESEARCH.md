# CLM Autoevolve: learning CLM online from successes and failures with LoRA

Research note on building an automatic training loop for
[Contrastive Language Models (CLM)](https://github.com/Contrastive-LM/CLM): CLM makes decisions
(state → action), the environment reports success or failure, and CLM improves on the fly
through LoRA adapters. Read against CLM `main` as of 2026-10-04.

---

## 1. Summary

**It is feasible, and CLM's design makes it cheap.** CLM is a *frozen* Qwen3-8B encoder plus two
small trainable projection heads (~19M parameters; the 75 MB reference checkpoint). Learning
online only has to touch the heads. That has three consequences:

1. **Every embedding stays valid.** Updates never require re-encoding anything. An update is
   a few optimiser steps of a ~19M-parameter MLP on cached 4096-d vectors: milliseconds on a
   GPU, seconds on a CPU.
2. **Deployment needs no server change.** `clm-serve` already hot-reloads a checkpoint when
   its mtime changes, and its vector cache keys rows by head *generation*, so stale
   projections are dropped automatically. A LoRA adapter merged into the base weights
   (`W + (α/r)·BA`) is a standard CLM checkpoint. The prototype here does that merge, and its
   tests check that `clm.heads.HeadPair` serves the merged file with outputs identical to the
   adapter model.
3. **LoRA here buys stability, not compute.** The heads are small enough to fine-tune in full.
   The reasons to use LoRA (rank 16 ≈ 344K parameters, 1.8% of the heads) are: it limits
   forgetting during many small updates; each update is a ~1.3 MB adapter you can version,
   A/B test and roll back; and you can keep one adapter per environment over a shared base.
   Full-head fine-tuning should stay as a baseline.

**The current CLM recipe throws failures away.** In `train/finetune.py` the `reward` field
of each step is read only to stratify folds (lines 79–85). `_clm_loss` treats *every* step
as a positive, including steps from trajectories that failed. The loop proposed here starts
by putting that signal back: successes become positives, and failed actions become hard
negatives or unlikelihood targets.

**Recommendation:** build it in four phases, cheapest and most informative first:

| Phase | What | Needs | Answers |
|---|---|---|---|
| 0 | **Offline**: retrain the DeepSWE head with outcome-aware losses on the *existing* embeddings (which carry `reward`), evaluate with `bon_eval.py` on heldout-38 | 1 GPU, no new data | Does using failures beat the current 81.6% (31/38)? |
| 1 | **Prequential replay**: stream DeepSWE/T-Rex episodes in order, test-then-train with the online learner | same | Does *incremental* LoRA learning keep up with batch training, without forgetting? |
| 2 | **Live loop**: T-Rex `--no-shield` (deaths = failure) and typed-decisions as a bandit | clm-serve + vLLM | End-to-end gains, latency of the loop, safety of auto-promotion |
| 3 | LoRA on the **encoder** (only if heads plateau) | multi-GPU, re-embedding | Is the frozen encoder the ceiling? |

---

## 2. What in CLM matters for this

| Fact (CLM repo) | Consequence for online learning |
|---|---|
| Score = `exp(logit_scale) · cos(state_head(s), action_head(a))`; encoder frozen (`heads.py`) | Train only the heads; log embeddings, not text |
| Answer distribution = softmax over the candidates' scores (`engine.py`, `schema.py`) | CLM is literally a **softmax policy** over a candidate set: classic contextual-bandit / policy-gradient machinery applies, and the served `probabilities` are the propensities you need for off-policy correction |
| `temperature` request parameter in `(0, 100]` | Built-in knob for exploration (sample at T > 1) |
| `HeadPair.ensure()` reloads on mtime change; `generation` bumps | Publishing = atomic file replace. No restart |
| `--model NAME=PATH`, `--ckpt-dir` | Serve `clm-online` *beside* `clm-latest` for shadow/A-B |
| Vector cache keyed `"{name}@{generation}"` | New head ⇒ cache misses for that head only; base head's cache stays hot |
| `Embedder` LRU-caches every normalised embedding it served | Recording a decision's embeddings costs no extra encoder call |
| Training = bidirectional InfoNCE; same `(task_id, step_idx)` masked; hard-negative variant in mid-training | The outcome-aware loss is a small change to the existing loss |
| **40% replay** of original data kept hard-negative top-1 at 68.5% vs **56.2%** without (README) | Replay is mandatory in the online loop, and they've already measured how much it matters |
| `docs/FINETUNING.md` "autofinetune": an agent edits `finetune.py`, keeps/discards by held-out score | That is an *outer* loop over the **recipe**. This proposal is the *inner* loop over **data**. They compose (§8) |
| Released head trained against Qwen3-8B last-token pooling, 2048 tokens served (8192 for DeepSWE training) | Train on *serving-time* embeddings to avoid train/serve skew in truncation and templating |

---

## 3. Framing: what "learning from success and failure" means for CLM

At each decision CLM sees a state `s`, a candidate set `{a_1..a_K}`, and returns
`π(a|s) = softmax_k(scale · cos(f(s), g(a_k)))`. The agent executes one action, and eventually
the environment says whether the episode succeeded. So:

* **Single decisions with immediate feedback** (routing, typed Choice, tool selection) are a
  **contextual bandit with logged propensities**: you see the reward only for the action
  taken.
* **Multi-step episodes with a terminal outcome** (SWE agents, games, computer use) add
  **credit assignment**: which steps caused the failure?
* **Best-of-N verification** (CLM's strongest results) gives **pairs**: at the same task,
  some trajectories pass and some fail, which is ideal for contrastive/preference losses.

### 3.1 Objectives (all implemented in `autoevolve/objectives.py`)

| Objective | Signal used | Use when | Notes |
|---|---|---|---|
| `outcome_infonce` | successes = positives; failed actions at the *same* decision point = hard negatives | Many decisions per state (best-of-N, revisited states) | CLM's own loss with a success mask; other successes of the same group are masked like `finetune.py` masks `(task, step)`. `hard_negative_boost` up-weights the failures (Robinson et al. 2021) |
| `outcome_ce` | CE toward a successful action; **unlikelihood** `−log(1−p)` for a failed one, over the logged candidate set | Any Choice-style decision | Learns from a failure even with no success at that state. In the toy below, it's the objective that helped most |
| `bandit_ppo` | `(r − baseline) ×` clipped importance ratio `π_θ/π_logged` | Logged bandit feedback, stochastic serving | Needs the served probabilities; off-policy correct (Swaminathan & Joachims 2015; PPO clip). Noisier; needs exploration |
| `pairwise_dpo` | `(state, winner, loser)` margin vs the frozen base head | Pass/fail pairs at the same state | DPO (Rafailov et al. 2023) with the scaled cosine as the "log-likelihood"; the base head is the reference (adapters off, no extra copy) |
| `anchor_kl` + `replay_infonce` | KL(base‖current) on replayed candidate sets; InfoNCE on original pairs | Always | Anti-forgetting; replay weight 0.4 mirrors the README's measured mix |

### 3.2 Credit assignment (`autoevolve/experience.py`)

| Strategy | Failed episode labels | Risk |
|---|---|---|
| `all` | every step negative | Teaches CLM that the many *correct* early steps were wrong |
| `terminal` (default) | last `window` steps negative, earlier steps unlabelled; successes label all steps | Misses early root causes; `window` should match the domain (DeepSWE BoN eval uses a 12-step window) |
| `discounted` | weight `γ^(T−1−t)` | Smooth compromise |
| dense `step_reward` | per-step labels from a verifier, test runner or safety shield. Always wins | Best when available. T-Rex's shield overriding an unsafe action is a free negative label; failing unit tests are a free per-step signal in SWE |

Better credit assignment (a learned value baseline, Contrastive RL's goal-conditioned critic,
Eysenbach et al. 2022) is the natural next step once the simple schemes are measured.

---

## 4. Where to put the LoRA

| Option | Trainable | Re-embed on update? | Serving change | Forgetting risk | Verdict |
|---|---|---|---|---|---|
| **A. LoRA on projection heads** | ~344K (r=16) | No | None (merge → hot reload) | Low | **Start here** |
| B. Full head fine-tune | ~19M | No | None | Higher with small online batches | Baseline to beat |
| C. Adapter bank: one head-LoRA per env/tenant | 344K each | No | `--model env=PATH` per adapter, or merge on demand | Isolated per env | When several environments share one server |
| D. LoRA on the Qwen3-8B encoder | ~10–40M | **Yes**: every cached and stored embedding becomes stale | vLLM must serve the pooling model with LoRA (verify support for your vLLM version); cache flush | Highest | Phase 3 only, if heads plateau |

LoRA parameter count on the reference head shape (`4096→1536→1536→512`, two heads), rank `r`:
`2 · r · [(4096+1536) + (1536+1536) + (1536+512)] = 21,504 · r` ⇒ r=8: 172K, r=16: 344K,
r=32: 688K, against ~18.9M base parameters.

Why A over B even though B is affordable: LoRA learns less and forgets less (Biderman et al.
2024). That is the right trade for many small, noisy, self-generated updates. Low rank also
caps how far one bad batch can move the model. Merging gives B's serving cost: zero. For
continual learning across very different environments, orthogonal-subspace LoRA (O-LoRA,
Wang et al. 2023) or option C avoid adapters interfering with each other.

---

## 5. System design

```
            ┌───────────────────────── serve ─────────────────────────┐
 agent ───► │ clm-serve  :8700   clm-latest (base)   clm-online (LoRA-merged) │──► vLLM Qwen3-8B :8090
   ▲        └───────────────┬─────────────────────────────────▲──────┘     (frozen encoder)
   │ action                 │ decision log                     │ atomic replace → mtime hot-reload
   │                        ▼                                  │
 environment ──outcome──► Experience buffer ──► Learner ──► Gate ──► publish  (versions/adapter_vNNNN.pt)
 (tests, game, user)      · embeddings (s, cands)  · LoRA steps    · held-out logged episodes
                          · chosen, probs, head gen · + 40% replay  · fixed regression suites
                          · credit assignment       · + KL anchor   · promote or roll back
```

### 5.1 Components

1. **Decision recorder.** For each answer, store: episode id, step, the state's and the
   candidates' encoder embeddings, the chosen index, the served probability vector, the
   model name and head generation, and optionally the texts (for debugging and re-embedding
   if the encoder changes). In-process (`clm.Engine`), embeddings come for free from
   `engine.embedder.embed(texts)`, which is an LRU hit. Over HTTP, CLM has no decision-id or
   feedback API today. The cleanest extension is for `/v1/systemone` to return a
   `decision_id` (server logs the vectors) plus a new `POST /v1/feedback {episode_id |
   decision_id, reward}`. That is a small fork of `server.py`/`engine.py`.
2. **Experience buffer + credit assignment.** Episodes stay open until an outcome arrives,
   then turn into labelled steps (§3.2). Reservoir-bounded store, a per-decision-point
   running-mean baseline for advantages, and a stream of fresh steps.
3. **Learner.** Every *N* episodes (or *M* labelled steps), a few AdamW steps on a batch that
   is half fresh experience and half buffer samples, plus replay of original training pairs
   and the KL anchor.
4. **Gate.** Score the candidate adapter on held-out logged episodes (every 5th episode is
   withheld from training) and on fixed regression suites: Nemotron hard-negative top-1, the
   typed-decisions validation split, and the DeepSWE heldout-38 BoN rate. Publish only on
   improvement; otherwise **roll the live weights back** to the last promoted adapter, so a
   bad update can't compound. Same keep/discard discipline as `docs/FINETUNING.md`.
5. **Publisher.** Merge, `torch.save` to a temp file, `os.replace` onto the served path.
   Keep every promoted adapter and an append-only log, so rollback means re-publishing an
   older one.
6. **Exploration.** Argmax serving yields propensity-1 logs and no counterfactual
   information. Serve the online model with sampling (temperature ≥ 1 or ε-greedy) on a
   fraction of traffic and log the probabilities actually used.

### 5.2 Cadence

Fine-grained (every few episodes) is affordable because updates are tiny. The binding
constraints are **statistical, not computational**: enough fresh outcomes per update for
the gate to tell signal from noise, and enough held-out episodes to gate on. Start with
updates every 25–100 episodes and gate every update. Lengthen the interval if the rejection
rate is high.

---

## 6. Risks and how the design handles them

| Risk | Mitigation |
|---|---|
| **Self-confirmation loop**: training on its own choices reinforces its own mistakes (what ignoring outcomes does) | Outcome-labelled losses; exploration with logged propensities; gate on held-out outcomes |
| Catastrophic forgetting of general skill | LoRA low rank, 40% replay, KL anchor, regression suites in the gate |
| Credit misassignment in long episodes | `terminal`/`discounted` credit, dense step rewards where they exist; measure `all` as the negative control |
| Noisy or hackable outcomes (flaky tests, user clicks) | Gate on a trusted held-out suite, not just online reward; margin `promote_margin`; per-source reward weights |
| Non-stationarity (environment drifts) | Fresh/buffer mix; reservoir buffer; periodic adapter reset + re-learn from buffer |
| Evaluation leakage | Task-disjoint held-out sets (CLM already uses task-disjoint folds); never train on gate episodes |
| Frozen-encoder ceiling: if two states embed alike, no head can separate them | Track per-group accuracy where success and failure share near-identical state embeddings; that is the trigger for Phase 3 |
| Logged states may hold sensitive data | Store embeddings by default, texts opt-in; retention limits |

---

## 7. Experimental plan

### Phase 0: offline, on data CLM already ships (highest value per hour)

`Contrastive-LM/deepswe-clm-embeddings-8k` metadata carries `reward` per step, and the same
task has passing and failing trajectories, so hard negatives at the same `(task, step)` exist.

* **Baseline:** the published recipe (`train/finetune.py --task clm`), heldout-38, `--n 4 --window 12` ⇒ 31/38.
* **Arms:** (i) success-only positives; (ii) `outcome_infonce` with failed steps in the
  last 12 as hard negatives (credit window = eval window); (iii) + `pairwise_dpo` on pass/fail
  pairs per task; (iv) LoRA r ∈ {8, 16, 32} vs full-head; all warm-started from `CLM_v0.1-8B.pt`.
* **Eval:** unchanged `bon_eval.py` (same folds, `--n`, `--window`) and Nemotron
  hard-negative top-1 for forgetting.
* **Note:** the BoN score is the *mean cosine over the final window*. Training failures down
  in exactly that window aligns the training signal with the selection rule.

### Phase 1: prequential (test-then-train) replay

Order the DeepSWE tasks (or T-Rex runs). For each block, first evaluate the current adapter
on it, then train on it. Report the cumulative BoN rate vs frozen and vs batch-trained
heads, plus forgetting on the regression suites. This isolates "online" from "more data".

### Phase 2: live loops

* **T-Rex** (`examples/t_rex`), run with `--no-shield` so the model's answer stands. Episode
  = run until a death; failure = death; dense negatives = decisions the shield *would* have
  overridden (the planner labels them unsafe). Train seeds disjoint from gate seeds; metric =
  deaths per 60 s and agreement with the planner. One caveat: the shipped prompt (`labeled`)
  writes "Safe/Unsafe/Best" into the options, so the task is near-trivial for the model.
  Use a prompt without the labels to make learning measurable.
* **Typed decisions as a bandit** (`LocalLLaMA/typed-decisions`). Reveal reward only for
  the chosen option (`gold == chosen`) and compare `outcome_ce`, `bandit_ppo` and
  full-information fine-tuning (the upper bound).

### Phase 3: encoder LoRA (conditional)

Only if Phase 0–2 show a plateau attributable to the encoder (see §6). Requires
re-embedding the stored states after each encoder update. Batch these updates (nightly), not
online.

### Metrics, in every phase

Online success rate (prequential), held-out success / BoN rate, regret vs oracle,
forgetting suites, gate accept/reject rate, update latency, encoder tokens spent (should be
~0 extra).

---

## 8. Relation to `docs/FINETUNING.md` (autofinetune)

CLM's repo already describes an *agent* that edits `train/finetune.py` in a loop and
keeps/discards by held-out score. That loop searches over **recipes** (losses, schedules,
head shapes) on a fixed dataset. This proposal is the **data loop**: fixed recipe, growing
experience. They compose naturally. The inner loop runs continuously. The outer loop
periodically proposes recipe changes (`LearnerConfig`: rank, losses, credit window, replay
weight), evaluated by prequential replay of the logged experience (Phase 1). Changes that
win are promoted.

---

## 9. Synthetic sanity check (prototype in this repo)

`experiments/synthetic_online.py` runs a toy world with no GPU. States and actions are fixed
random vectors that stand in for the frozen encoder's embeddings. The right action among
K=4 candidates is `argmax sᵀW*a`. The base head is pre-trained, like CLM, on a related but
shifted rule, so it starts imperfect. Episodes have 3 steps and end at the first wrong
action, and only the episode outcome is reported. LoRA r=8, an update every 25 episodes,
gated on held-out logged episodes, 40% replay, KL 0.05. Each arm sees the identical episode
stream; 3 seeds.

RESULTS_TABLE

**This validates the mechanics, not CLM-8B.** The toy encoder is 32-d with a low-rank rule,
and absolute numbers mean nothing. What carries over is the *ordering*: whether
outcome-aware objectives beat ignoring outcomes, and whether credit assignment matters.

---

## 10. The prototype

| File | What |
|---|---|
| `autoevolve/lora.py` | `LoRAHeads`: wraps a CLM checkpoint's heads with LoRA, `reference()` context (adapters off = frozen base for KL/DPO), `merged_checkpoint()` in CLM's exact format, `atomic_save()` |
| `autoevolve/objectives.py` | `outcome_infonce`, `outcome_ce`, `bandit_ppo`, `pairwise_dpo`, `anchor_kl`, `replay_infonce` |
| `autoevolve/experience.py` | `Decision`, `assign_credit` (`all`/`terminal`/`discounted` + dense step rewards), `ExperienceBuffer`, `to_tensors` |
| `autoevolve/learner.py` | `OnlineLearner`: update (fresh + buffer + replay + KL), gate with rollback, publish with versioned adapters |
| `tests/test_autoevolve.py` | Includes: merged checkpoint served by `clm.heads.HeadPair` matches the adapter model; publish triggers hot reload (new generation); losses move the right way; gate rolls back |
| `experiments/synthetic_online.py` | The toy comparison above |

```bash
pip install torch numpy pytest contrastive-lm   # or: pip install --no-deps -e <CLM clone>
python -m pytest -q tests
python experiments/synthetic_online.py --seeds 0 1 2
```

Wiring it to a real server (Phase 2):

```bash
clm-serve --model clm-online=heads/clm-online.pt     # base stays clm-latest
```
```python
learner = OnlineLearner(torch.load(clm_download_path, weights_only=False), "heads/clm-online.pt",
                        LearnerConfig(rank=16, losses=("infonce", "ce")), replay=(S, A), anchor=anchor)
# per decision:  buffer.record(Decision(ep, t, s_emb, cand_embs, chosen, probs, group))
# per episode:   buffer.close(ep, outcome); every N episodes: learner.update(buffer); learner.maybe_publish(gate)
```

---

## 11. Decisions needed

1. **First environment**: Phase 0 on DeepSWE (no new infrastructure, directly comparable to
   CLM's headline number) is recommended. T-Rex is the best *live* demo.
2. **Server API**: fork CLM to add `decision_id` + `/v1/feedback`, or record client-side
   with the in-process `Engine` (no fork; ties the learner to the agent process).
3. **Hardware**: Phase 0 needs one GPU with the DeepSWE embedding dataset; Phases 2–3 need
   the vLLM encoder running.

## References

* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
* Biderman et al., *LoRA Learns Less and Forgets Less*, 2024.
* Wang et al., *Orthogonal Subspace Learning for Language Model Continual Learning* (O-LoRA), 2023.
* Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (EWC), 2017.
* Rolnick et al., *Experience Replay for Continual Learning*, 2019.
* Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018.
* Robinson et al., *Contrastive Learning with Hard Negative Samples*, 2021.
* Eysenbach et al., *Contrastive Learning as Goal-Conditioned Reinforcement Learning*, 2022.
* Swaminathan & Joachims, *Counterfactual Risk Minimization: Learning from Logged Bandit Feedback*, 2015.
* Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
* Rafailov et al., *Direct Preference Optimization*, 2023.
* Welleck et al., *Neural Text Generation with Unlikelihood Training*, 2019.
* Dawid, *Present Position and Potential Developments: The Prequential Approach*, 1984.
* Kwok et al., *Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making*, 2026 (CLM repo / blog).
