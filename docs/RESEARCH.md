# CLM Autoevolve: learning CLM online from successes and failures

Research note on an automatic training loop for
[Contrastive Language Models (CLM)](https://github.com/Contrastive-LM/CLM). CLM decides
(state → action), the environment reports success or failure, and CLM improves on the fly in two
ways: an **outcome map** (a distribution of past successes and failures that corrects CLM's
scores instantly) and **LoRA adapters** on its projection heads (consolidated, gated, hot-reloaded).
Read against CLM `main` as of 2026-10-04; the prototype and testbed are in this repository.

---

## 1. Summary

SUMMARY_RESULTS

**Why it is cheap for CLM specifically.** CLM is a *frozen* Qwen3-8B encoder plus two small
trainable projection heads (~19M parameters, the 75 MB reference checkpoint). Everything here
touches only the heads or lives in the encoder's frozen space:

1. **No embedding ever goes stale.** Logged decisions, the outcome map and the replay sets are
   encoder vectors, and they stay valid across every head update. An update is a few optimiser
   steps of a small MLP on cached vectors: milliseconds on a GPU.
2. **Deployment needs no server change.** `clm-serve` hot-reloads a checkpoint when its mtime
   changes, and its vector cache keys rows by head *generation*. A LoRA adapter merged into the
   base weights (`W + (α/r)·BA`) is a standard CLM checkpoint. The tests check that
   `clm.heads.HeadPair` serves the merged file with outputs identical to the adapter model, and
   that republishing bumps the generation.
3. **LoRA here buys stability and versioning, not compute.** The heads are small enough to
   fine-tune in full. The reasons to use LoRA are elsewhere: it bounds how far many small,
   noisy, self-generated updates can move the model; each update is a ~1.3 MB adapter you can
   gate, version and roll back; and you can keep one adapter per environment over a shared base.

**The current CLM recipe throws failures away.** In `train/finetune.py` the `reward` of each
step is read only to stratify folds (lines 79–85). `_clm_loss` treats every step as a positive,
including steps of failed trajectories. The T-Rex control arm that does the same (every executed
step a positive) got *worse* than not learning at all (§8).

**Recommended path:**

| Phase | What | Needs | Answers |
|---|---|---|---|
| ✅ T | **T-Rex testbed** (this repo): outcome map + gated LoRA from obstacle successes/crashes, CPU stand-in encoder | CPU | Do the mechanics work, which parts matter? (§8) |
| 0 | **Offline on DeepSWE**: retrain the head with outcome-aware losses on the *existing* embeddings (they carry `reward`), evaluate with `bon_eval.py` on heldout-38 | 1 GPU, no new data | Does using failures beat the published 81.6% (31/38)? |
| 1 | **Prequential replay**: stream DeepSWE tasks in order, test-then-train | same | Does incremental learning keep up with batch training without forgetting? |
| 2 | **Live loops**: T-Rex with the real CLM-8B (`--encoder clm`), typed-decisions as a bandit | clm-serve + vLLM | End-to-end gains with the real encoder |
| 3 | LoRA on the **encoder** (only if heads plateau) | multi-GPU, re-embedding | Is the frozen encoder the ceiling? |

---

## 2. Does it make sense to use the reward to keep CLM learning?

**Yes, under conditions that are easy to check.** The argument:

* **CLM already is a policy over a candidate set.** The served answer is
  `π(a|s) = softmax_k(scale·cos(f(s), g(a_k)))`, and its probabilities are returned with every
  answer. A success/failure signal on the executed action is exactly the feedback a softmax
  policy can learn from (contextual bandit with logged propensities; with episodes, RL with a
  terminal reward). Nothing has to be bolted on to make CLM "trainable from reward".
* **The reward is the quantity CLM is deployed to predict.** As a verifier (DeepSWE,
  Terminal-Bench) CLM's job is to rank the candidate that *will succeed* first. Training on
  "this state–action pair was taken" (today's recipe) is a proxy for that. Training on "this
  was taken *and it worked / failed*" is the target itself.
* **Failures are the cheapest hard negatives there are.** CLM's mid-training needed 30M
  synthetic hard negatives from Gemini to reach 69% hard-negative top-1. A deployed agent
  produces real ones for free: plausible actions, chosen by CLM itself, that failed.
* **The update is cheap enough to do continuously** (frozen encoder, small heads; §1).

**It does not make sense, or needs more machinery, when:**

| Condition | Why it matters | What to do |
|---|---|---|
| Outcomes are rare or very delayed | Too few labels per update for the gate to separate signal from noise | Batch updates (daily), or use dense proxies (tests, verifiers) |
| The reward is noisy or hackable (flaky tests, clicks) | The loop optimises the reward, not the intent | Gate on a trusted held-out suite, not only on online reward |
| Credit is unclear (long episodes, late failures) | Labels land on the wrong steps | Terminal-window / discounted credit, dense step signals (§3.2) |
| No exploration (greedy serving) | Propensity-1 logs: nothing is learned about the actions not taken | Sample from the answer distribution on a share of traffic, log the probabilities |
| The frozen encoder cannot tell the relevant states apart | No head can separate what the encoder merges | Detect (success and failure at near-identical embeddings), then Phase 3 |
| Labels are available anyway | Supervised fine-tuning is simpler and stronger | Use `train/finetune.py` with them |

**The one thing it must not do is learn from its own choices without the outcome.** That is a
self-confirmation loop. In the T-Rex testbed the "all-positive, ungated" control learns from
every executed step as if it were right, and it ends up dying more than the frozen base (§8).

---

## 3. Learning signals

At each decision CLM sees a state `s` and candidates `{a_1..a_K}`, the agent executes one, and an
outcome arrives later:

* **Single decisions with immediate feedback** (routing, typed Choice, tool selection): a
  contextual bandit with logged propensities.
* **Multi-step episodes with a terminal outcome** (SWE agents, games): add credit assignment.
* **Best-of-N verification**: passing and failing trajectories of the same task give pairs.

### 3.1 Objectives (`autoevolve/objectives.py`)

| Objective | Signal | Use when | Notes |
|---|---|---|---|
| `outcome_ce` | CE towards a successful action; **unlikelihood** `−log(1−p)` on a failed one, over the logged candidate set | Any Choice-style decision; **closed action sets** | Learns from a failure even with no success at that state. Used in the T-Rex runs |
| `outcome_infonce` | CLM's bidirectional InfoNCE with successes as positives and failed actions at the same decision point as hard negatives | Large/open action spaces (DeepSWE, typed decisions) | Masks other successes of the same group (as `finetune.py` masks `(task, step)`) and identical action texts (with 3 actions, in-batch "negatives" are mostly the positive itself) |
| `map_distill` | CE towards the outcome map's success distribution over the candidates | With the outcome map | Consolidates the fast memory into the LoRA (§4) |
| `bandit_ppo` | `(r − baseline) ×` clipped ratio `π_θ/π_logged` | Logged bandit feedback | Off-policy correct (Swaminathan & Joachims 2015; PPO clip); noisier |
| `pairwise_dpo` | `(s, winner, loser)` margin vs the frozen base head | Pass/fail pairs at the same state | DPO (Rafailov et al. 2023) with the scaled cosine as log-likelihood; base = adapters off |
| `anchor_kl`, `replay_infonce` | KL(base‖current) on replayed candidate sets; InfoNCE on original pairs | Always | Anti-forgetting; CLM's README measured 40% replay keeping hard-negative top-1 at 68.5% vs 56.2% |

### 3.2 Credit assignment (`autoevolve/experience.py`)

| Strategy | Failed episode labels | Risk |
|---|---|---|
| `all` | every step negative | Teaches CLM that the many correct early steps were wrong |
| `terminal` (default) | last `window` steps negative, earlier ones unlabelled; successes label all steps | Misses early root causes; set `window` per domain (T-Rex: 8 decisions ≈ one jump's airtime; DeepSWE BoN uses 12 steps) |
| `discounted` | weight `γ^(T−1−t)` | Smooth compromise |
| dense `step_reward` | per-step labels from a verifier, test runner or safety shield (always wins) | Best when available: T-Rex's shield overriding an action, a failing unit test |

---

## 4. The outcome map: the distribution of successes and failures as part of the contrast

The idea: keep **every** labelled `(state, action, outcome, task)` the system has lived through,
and use their *distribution*, by task and across similar tasks, to make CLM better at that task.

### 4.1 What it is (`autoevolve/outcome_map.py`)

Points live in the **frozen encoder space** (the same 4096-d vectors CLM already computes and
caches). For a new decision `(s, {a_k}, task)`:

```
w_i  = exp((cos(s, s_i) − 1)/τ_s) · exp((cos(a_k, a_i) − 1)/τ_a) · (1 + β·[task_i = task])
p_k  = (α0 + Σ w_i r_i) / (α0 + β0 + Σ w_i)          success posterior of action k here
n_k  = Σ w_i                                          evidence
```

* An **exact revisit** counts fully (w = 1); a **similar state** counts less; a dissimilar one
  not at all. Unknown regions fall back to the prior (p = ½, n ≈ 0), which leaves CLM alone.
* The **task bonus** makes a task's own history count more than its neighbours'. A *new* task
  still borrows from *similar* tasks (by state similarity) until it has history of its own.
  That is the "one task or similar tasks" behaviour. In T-Rex, tasks are obstacle types
  ("2 large cacti", "low bird", …).
* A ring buffer keeps the newest experience (non-stationarity).

### 4.2 How the distribution enters the contrast

CLM's decision *is* a contrast: one state against K candidates, softmax over scaled cosines.
The map adds evidence to that contrast in two ways:

1. **At inference, instantly**: `bias_k = λ · n_k/(n_k+κ) · logit(p_k)` is added to CLM's logit
   for candidate k. An action that has kept failing in this region is pushed down after a
   handful of failures, with no gradient step, and the push grows with the evidence. This is
   episodic control (Blundell et al. 2016; Pritzel et al. 2017) and kNN-LM (Khandelwal et
   al. 2020) in CLM's terms.
2. **As the training target** (consolidation): `soft_targets()` turns the posterior over a
   decision's candidates into a distribution, and `map_distill` trains the LoRA head towards it,
   weighted by evidence. Every candidate gets a graded target, failed ones included, instead of
   one positive against in-batch negatives. The parametric head then generalises what the map
   only remembers.

This is the complementary-learning-systems split (McClelland et al. 1995): a fast,
non-parametric memory that learns from one episode, and a slow parametric model that
consolidates it. For CLM it is natural: the vectors are already computed and cached (`clm-serve`
already reserves a GPU vector arena for them), so a map query is a matrix–vector product next to
the head.

### 4.3 Possible extensions

* **Hard-negative mining from the map**: the actions with the lowest posterior in similar states
  are the best negatives for `outcome_infonce` in open action spaces (DeepSWE).
* **Per-task prototypes**: success/failure centroids per task in projection space, as a
  task-conditioned contrast direction.
* **The map as a router**: when a new task has no history, use the task with the most similar
  state distribution (map neighbourhood) to pick which per-task adapter to load.

---

## 5. Where to put the LoRA

| Option | Trainable | Re-embed on update? | Serving change | Forgetting risk | Verdict |
|---|---|---|---|---|---|
| **A. LoRA on projection heads** | ~344K (r=16) | No | None (merge → hot reload) | Low | **Start here** (used throughout) |
| B. Full head fine-tune | ~19M | No | None | Higher with small online batches | Baseline to beat |
| C. Adapter bank: one head-LoRA per env/task | 344K each | No | `--model env=PATH` per adapter | Isolated | Several environments on one server |
| D. LoRA on the Qwen3-8B encoder | ~10–40M | **Yes**: all stored embeddings and the map go stale | vLLM must serve the pooling model with LoRA (check your version) | Highest | Phase 3 only |

On the reference head shape (`4096→1536→1536→512`, two heads), rank `r` costs
`2·r·[(4096+1536)+(1536+1536)+(1536+512)] = 21,504·r` parameters (r=16: 344K, 1.8% of ~18.9M).
LoRA learns less and forgets less (Biderman et al. 2024), which is the right trade for many
small, noisy updates. O-LoRA (Wang et al. 2023) or option C keep adapters for very different
environments from interfering.

---

## 6. System design

```
            ┌──────────────────────────── serve ────────────────────────────┐
 agent ───► │ clm-serve :8700   clm-latest (base)   clm-online (LoRA-merged)  │──► vLLM Qwen3-8B (frozen)
   ▲        └──────────┬──────────────────────────────────────▲──────────────┘
   │ action      logits│+ outcome-map bias                     │ atomic replace → mtime hot-reload
   │                   ▼                                       │
 environment ─outcome─► Experience buffer ──► Learner (candidate LoRA) ──► Gate ──► publish (versions/)
 (tests, game, user)    · embeddings, chosen,   · CE / unlikelihood         · candidate vs served,
                        · probs, head gen, task · + map distill + KL/replay   same held-out episodes (SNIPS)
                        · credit assignment ──► Outcome map (frozen space)  · promote / keep / roll back
```

1. **Decision recorder**: episode id, step, state and candidate embeddings, chosen index,
   served probabilities, head generation, task id. In-process (`clm.Engine`) the embeddings
   are an LRU hit. Over HTTP, CLM needs a small extension: `/v1/systemone` returning a
   `decision_id`, plus `POST /v1/feedback {decision_id | episode_id, reward}`.
2. **Experience buffer + credit assignment** (§3.2). Every 5th episode is held out for the gate
   and kept out of the map and the training set.
3. **Outcome map** (§4), updated on every closed training episode.
4. **Learner**: a *candidate* adapter keeps training across updates (fresh + buffered steps,
   chosen objectives, KL anchor/replay); a separate *served* adapter acts.
5. **Gate**: scores candidate and served adapters on the **same** held-out logged episodes with
   a self-normalised importance-sampling estimate of the success rate (using the same map bias
   the policy acts with). It promotes only on improvement and resets the candidate only after
   `rollback_patience` clearly-worse gates in a row. Add fixed regression suites for production
   (Nemotron hard-negative top-1, typed-decisions validation, DeepSWE heldout-38).
6. **Publisher**: merge, temp file, `os.replace` onto the served path; every promoted adapter and
   an append-only log are kept for rollback.
7. **Exploration**: sample from the answer distribution (temperature, ε floor) and log the
   probabilities actually used.

*A note on the gate design.* The first prototype compared each candidate with a score the
incumbent got earlier on a smaller validation set, and reset the weights after every rejection.
It rejected 100% of updates in the toy run, so no gain could accumulate. Comparing both adapters
on the same data, keeping the candidate training, and rolling back only on sustained regression
fixed it. Treat that failure as a design requirement.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Self-confirmation loop** (training on own choices without outcomes) | Outcome-labelled losses only; the gate; exploration with logged propensities. Measured in §8 |
| Catastrophic forgetting | LoRA low rank, KL anchor, replay, regression suites in the gate |
| Credit misassignment | `terminal`/`discounted` credit, dense step signals |
| Map over-confidence (stale or out-of-distribution memories) | Evidence-weighted bias (`n/(n+κ)`); ring buffer; prior for empty regions |
| Noisy/hackable outcomes | Gate on trusted held-out data; `promote_margin`; per-source reward weights |
| Evaluation leakage | Held-out episodes never enter the training set or the map; task-disjoint suites |
| Frozen-encoder ceiling | Watch for success and failure at near-identical embeddings → Phase 3 |
| Logged states may hold sensitive data | Store embeddings by default, texts opt-in; retention limits |

---

## 8. T-Rex testbed results

**Setup** (`testbeds/trex/`). CLM's T-Rex game (engine and planner vendored from
`examples/t_rex`), headless, lockstep, a decision every 4 frames. The **neutral prompt** describes
only the scene ("2 large cacti ahead, close, about 100 px away. Speed 9.") and the actions ("Jump:
leap up and over what is ahead."), unlike the example's `labeled` prompt, which writes
"Safe/Unsafe/Best" into the options. Each obstacle encounter is an episode: cleared = success,
crash = failure. Each life starts at a random speed in [6, 10], so birds (speed ≥ 8.5) appear in
every life. The planner is used only as an evaluation oracle and to build the base head.

* **Base head** (stand-in for "CLM before this task"): heads trained on the planner's choices on an
  **easy course** (single cacti only: no groups, no birds) with a CPU hash encoder (256-d).
* **Online phase**: 60,000 frames (~17 game-minutes, hundreds of lives) on one training course.
  Sampling from the answer distribution (ε = 0.05); LoRA r=8 updated every 10 training episodes;
  terminal credit, window 8; KL anchor 0.05; gate on held-out logged episodes.
* **Evaluation**: greedy, no learning, map frozen, on 5 **unseen** courses × 2 minutes.
* 5 seeds (base head, training course and initialisation differ per seed).

TREX_RESULTS

---

## 9. Synthetic check

`experiments/synthetic_online.py`: states and actions are fixed random vectors (a 32-d stand-in
encoder), and the right action among K=4 is `argmax sᵀW*a` with a low-rank `W*`. The base head is
pre-trained on a related but shifted rule; episodes have 3 steps and end at the first wrong
action; only the episode outcome is reported. LoRA r=8, an update every 25 episodes, gated
(SNIPS), 40% replay, KL 0.05; identical episode streams per arm; 3 seeds.

SYNTH_RESULTS

---

## 10. Experimental plan (next)

### Phase 0: offline on DeepSWE (highest value per hour)

`Contrastive-LM/deepswe-clm-embeddings-8k` metadata carries `reward` per step, and tasks have both
passing and failing trajectories, so same-`(task, step)` hard negatives exist.

* **Baseline:** published recipe, heldout-38, `--n 4 --window 12` ⇒ 31/38.
* **Arms:** success-only positives; `outcome_infonce` with failed steps in the last 12 as hard
  negatives (credit window = eval window); + `pairwise_dpo` on pass/fail pairs; + map-mined hard
  negatives; LoRA r ∈ {8, 16, 32} vs full head; all warm-started from `CLM_v0.1-8B.pt`.
* **Eval:** unchanged `bon_eval.py` (same folds, `--n`, `--window`) and Nemotron hard-negative
  top-1 for forgetting.
* The BoN score is the *mean cosine over the final window*, so training failures down in that
  window aligns training with selection.

### Phase 1: prequential replay

Order tasks; evaluate each block before training on it. Report cumulative BoN rate vs frozen and
vs batch training, plus forgetting.

### Phase 2: live loops with the real encoder

* `python testbeds/trex/run_online.py --encoder clm --ckpt "$(clm-download)"`: the same arms with
  Qwen3-8B embeddings and the released head as the base (zero-shot, neutral prompt).
* Typed decisions as a bandit (`LocalLLaMA/typed-decisions`): reveal reward only for the chosen
  option; compare `outcome_ce`, `bandit_ppo`, map + LoRA and full-information fine-tuning
  (upper bound).

### Phase 3: encoder LoRA (conditional)

Only if a plateau is attributable to the encoder. Re-embed stored states and rebuild the map
after each encoder update, in nightly batches.

---

## 11. Relation to `docs/FINETUNING.md` (autofinetune)

CLM's repo describes an *agent* that edits `train/finetune.py` in a loop and keeps or discards by
held-out score: a search over **recipes** on fixed data. This is the **data loop**: fixed recipe,
growing experience. They compose. The inner loop runs continuously. The outer loop proposes
`LearnerConfig` / map changes (rank, losses, credit window, map strength, KL), evaluates them by
replaying the logged experience (Phase 1), and promotes the winners.

---

## 12. Decisions needed

1. **Phase 0 on DeepSWE next?** Recommended: no new infrastructure, directly comparable to CLM's
   headline number.
2. **Server API:** fork CLM to add `decision_id` + `/v1/feedback` (and optionally the map bias
   inside `Engine.answer`), or record client-side with the in-process `Engine`.
3. **Hardware:** Phase 0 needs one GPU with the DeepSWE embedding dataset; Phase 2 needs the vLLM
   encoder running.

## References

* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
* Biderman et al., *LoRA Learns Less and Forgets Less*, 2024.
* Wang et al., *Orthogonal Subspace Learning for Language Model Continual Learning* (O-LoRA), 2023.
* Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (EWC), 2017.
* Rolnick et al., *Experience Replay for Continual Learning*, 2019.
* McClelland, McNaughton & O'Reilly, *Why there are complementary learning systems in the hippocampus and neocortex*, 1995.
* Blundell et al., *Model-Free Episodic Control*, 2016.
* Pritzel et al., *Neural Episodic Control*, 2017.
* Khandelwal et al., *Generalization through Memorization: Nearest Neighbor Language Models*, 2020.
* Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018.
* Robinson et al., *Contrastive Learning with Hard Negative Samples*, 2021.
* Eysenbach et al., *Contrastive Learning as Goal-Conditioned Reinforcement Learning*, 2022.
* Swaminathan & Joachims, *Counterfactual Risk Minimization: Learning from Logged Bandit Feedback*, 2015.
* Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
* Rafailov et al., *Direct Preference Optimization*, 2023.
* Welleck et al., *Neural Text Generation with Unlikelihood Training*, 2019.
* Dawid, *Present Position and Potential Developments: The Prequential Approach*, 1984.
* Kwok et al., *Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making*, 2026 (CLM repo / blog).
