# CLM-autoevolve

**Let a [Contrastive Language Model (CLM)](https://github.com/Contrastive-LM/CLM) get better at a
task from its own successes and failures, while it is being used.**

CLM is a *System One* model: a frozen Qwen3-8B encoder embeds a state and each candidate action,
two small projection heads (~19M parameters) map them into a shared space, and the answer is a
softmax over their cosine scores. This repository adds the loop around it:

```
 state ─► CLM (base head + LoRA) ─► action ─► environment ─► success / failure
   ▲         ▲            ▲                                      │
   │         │ logit bias │ hot-reload (merged checkpoint)       │
   │    Outcome map    Learner + gate ◄── experience buffer ◄────┘
   │   (fast memory)   (slow LoRA consolidation)
```

* **Outcome map**: every labelled `(state, action, outcome)` is kept in the frozen encoder's
  space. For a new decision it gives a kernel-weighted posterior of "how often did this action
  work here, or in states like this, or in this task". It corrects CLM's logits *immediately*,
  with no training.
* **LoRA learner**: adapters on the projection heads, trained on outcomes: cross-entropy towards
  actions that worked, unlikelihood on actions that failed, optional distillation towards the
  map's distribution, plus a KL anchor/replay against forgetting.
* **Gate**: a candidate adapter is promoted only when it beats the served one on held-out logged
  episodes (self-normalised importance sampling). A sustained regression rolls it back.
* **Zero server changes**: a promoted adapter is merged (`W + (α/r)·BA`) into a standard CLM
  checkpoint and written atomically. `clm-serve` hot-reloads it by mtime, and its vector cache
  invalidates itself.

The full analysis, design and results are in **[docs/RESEARCH.md](docs/RESEARCH.md)**.

## Results on the T-Rex testbed

CLM's own T-Rex game (`examples/t_rex`), played headless with a **neutral prompt**: the scene and
the actions only, no "Safe/Unsafe" labels, so the model has to learn what works. Each obstacle is
an episode (cleared = success, crash = failure). The base head knows only an easy course (single
cacti). The online phase is 60k frames on one course, and evaluation is greedy on 5 unseen
courses. CPU stand-in encoder, 5 seeds: see [docs/RESEARCH.md §8](docs/RESEARCH.md#8-t-rex-testbed-results).

| Arm | Deaths / min | Obstacles cleared | Low birds cleared (new to the base) |
|---|---|---|---|
| Frozen base | 8.00 ± 0.61 | 72.6% | 50% |
| Outcome map only (no training) | 4.12 ± 1.52 | 89.4% | 78% |
| Gated LoRA | 2.82 ± 1.06 | 93.3% | 95% |
| **Outcome map + gated LoRA** | **0.80 ± 0.43** | **98.5%** | **100%** |
| Reward ignored (every executed step a positive, ungated) | 9.86 ± 1.12 | 58.1% | 54% |

The map and the LoRA add up (10× fewer deaths than the base), the largest gains are on obstacle
types the base never saw, and learning from the system's own choices *without* the outcome is
worse than not learning at all.

## Quickstart

```bash
pip install torch numpy pytest
pip install --no-deps contrastive-lm     # only clm.heads is needed on CPU; full install for serving
python -m pytest -q tests                # 20 tests: LoRA export through clm.heads.HeadPair, map, gate, testbed

python testbeds/trex/run_online.py                       # T-Rex, CPU, ~20 min for 7 arms x 5 seeds
python testbeds/trex/run_online.py --seeds 0 --arms frozen map lora+map --train-frames 30000   # quick look
python experiments/synthetic_online.py --seeds 0 1 2     # toy bandit world, objectives compared
```

With the real CLM-8B (GPU, vLLM encoder running as in the CLM README):

```bash
python testbeds/trex/run_online.py --encoder clm --ckpt "$(clm-download)" --seeds 0
```

Serving a learned head next to the base one:

```bash
clm-serve --model clm-online=heads/clm-online.pt       # republished files hot-reload
```

```python
from autoevolve import OnlineLearner, LearnerConfig, OutcomeMap, ExperienceBuffer, Decision, snips

omap = OutcomeMap(dim=4096)
learner = OnlineLearner(base_ckpt, "heads/clm-online.pt",
                        LearnerConfig(rank=16, losses=("ce", "distill")), anchor=anchor, outcome_map=omap)
buffer = ExperienceBuffer(strategy="terminal", window=8)
# per decision: buffer.record(Decision(episode, step, state_emb, cand_embs, chosen, probs, group, task=task_id))
# per episode:  labeled = buffer.close(episode, success); omap.add_labeled(labeled)
# every N episodes: learner.update(buffer); learner.maybe_publish(lambda m: snips(m, held_out_items))
```

## Layout

| Path | What |
|---|---|
| `autoevolve/lora.py` | `LoRAHeads`: LoRA on a CLM checkpoint's heads, `reference()` (frozen base), merged export in CLM's format, `atomic_save` |
| `autoevolve/outcome_map.py` | `OutcomeMap`: success/failure distribution over (state, action, task); `bias()` at inference, `soft_targets()` for training |
| `autoevolve/objectives.py` | `outcome_ce`, `outcome_infonce`, `map_distill`, `bandit_ppo`, `pairwise_dpo`, `anchor_kl`, `replay_infonce` |
| `autoevolve/experience.py` | `Decision`, credit assignment (`all` / `terminal` / `discounted` / dense step rewards), `ExperienceBuffer` |
| `autoevolve/learner.py` | `OnlineLearner`: candidate vs served adapter, gate, rollback, versioned publish; `snips` evaluator |
| `autoevolve/encoders.py` | `ClmEncoder` (Qwen3-8B via vLLM, as CLM serves) and `HashEncoder` (CPU stand-in) |
| `testbeds/trex/` | Headless T-Rex (engine and planner vendored from CLM, Apache-2.0), `run_online.py` experiment |
| `experiments/synthetic_online.py` | Toy contextual-bandit world for comparing objectives |
| `experiments/results/` | JSON results of the runs reported in the docs |
| `docs/RESEARCH.md` | The research note |

## Status

Research prototype. The T-Rex and synthetic results use a CPU stand-in encoder, not Qwen3-8B. They
show the mechanics work and how the pieces compare. They are not a claim about CLM-8B's numbers.
The next step is Phase 0 in the research note: the same outcome-aware training on CLM's published
DeepSWE embeddings, which already carry per-step rewards, against its 81.6% best-of-N result.
Those embeddings and the reference head are on the Hugging Face Hub (`Contrastive-LM/*`). Only the
heads train, so Phase 0 needs no encoder or GPU, just network access to `huggingface.co` (and its
file-download hosts) from wherever it runs.

## License

Apache 2.0. `testbeds/trex/engine.py` and `planner.py` are vendored unmodified (apart from a
header) from [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) `examples/t_rex`
(Apache-2.0), itself derived from [laya-vs-jev](https://github.com/virajbhartiya/laya-vs-jev)
(Apache-2.0).
