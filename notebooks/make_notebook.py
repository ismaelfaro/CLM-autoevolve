"""Builds notebooks/CLM_autoevolve_colab.ipynb:  python notebooks/make_notebook.py notebooks/CLM_autoevolve_colab.ipynb"""
import json
import sys

REPO = "ismaelfaro/CLM-autoevolve"
BRANCH = "claude/clm-online-lora-research"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(f"""
# CLM-autoevolve on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/CLM_autoevolve_colab.ipynb)

Can a [Contrastive Language Model (CLM)](https://github.com/Contrastive-LM/CLM) keep improving from its own
**successes and failures**? This notebook runs everything in this repository:

| Part | What | Runtime | Time |
|---|---|---|---|
| 0 | Setup + the 22 unit tests | any (CPU is fine) | ~3 min |
| A | **T-Rex testbed** (CPU stand-in encoder): outcome map + gated LoRA vs frozen | any | ~5 min |
| B | Synthetic toy: which learning objectives work | any | ~5 min |
| C | **T-Rex with the real CLM-8B** (Qwen3-8B encoder + released head) | GPU ≥ 24 GB (**L4** or **A100**) | ~30–60 min |
| D | **Phase 0, DeepSWE**: retrain CLM's verifier head with the reward in the loop, scored by CLM's own `bon_eval.py` against its published 31/38 | GPU recommended, high-RAM helps | ~30–90 min |

Parts A and B reproduce the results in [`docs/RESEARCH.md`](https://github.com/{REPO}/blob/{BRANCH}/docs/RESEARCH.md)
at a smaller scale. Parts C and D are the first runs with the real model and data, never run before
this notebook: treat their numbers as new results.

**Runtime:** *Runtime → Change runtime type*. CPU is enough for Parts 0–B; pick **L4** or **A100** for C and D.
Secrets (key icon on the left), both optional: `GITHUB_TOKEN` only if the repository is private;
`HF_TOKEN` only if Hugging Face rate-limits anonymous downloads.
""")

md("## 0. Setup")
code(f"""
REPO, BRANCH = "{REPO}", "{BRANCH}"
QUICK = False   # True: smallest settings everywhere (a smoke test of the whole notebook)

import os, subprocess, sys
os.makedirs("/content", exist_ok=True); os.chdir("/content")
token = None
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")      # only needed for a private repository
except Exception:
    pass
url = f"https://{{token}}@github.com/{{REPO}}.git" if token else f"https://github.com/{{REPO}}.git"
if not os.path.isdir("CLM-autoevolve"):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, "CLM-autoevolve"], check=True)
if not os.path.isdir("CLM"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Contrastive-LM/CLM.git", "CLM"], check=True)
os.chdir("/content/CLM-autoevolve")
sys.path.insert(0, "/content/CLM-autoevolve")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
""")
code("""
# torch, numpy, transformers, huggingface_hub, pyarrow, pandas and matplotlib ship with Colab.
# The clm package is installed without its serving deps (vLLM, FastAPI): only its heads code is needed here.
!pip install -q --no-deps /content/CLM
!pip install -q pytest
try:
    from google.colab import userdata
    os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
except Exception:
    pass
""")
code("""
import shutil, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
print(f"disk free: {shutil.disk_usage('/content').free / 1e9:.0f} GB")
!free -g | head -2
""")
code("""
!python -m pytest -q tests
""")

md("""
### Plot helpers
One measure per chart (deaths per minute on unseen courses; lower is better). Learning arms in blue, the two
references (frozen base, and learning with the reward ignored) in gray, values labelled on the bars.
""")
code("""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt

LEARN, REF, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
REFERENCE_ARMS = {"frozen", "all-positive-ungated"}

def trex_table(path):
    rows = json.load(open(path))["rows"]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    out = []
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        dm = [r["eval"]["deaths_per_min"] for r in rs]
        out.append({"arm": a, "seeds": len(rs), "deaths/min": np.mean(dm), "± std": np.std(dm),
                    "cleared %": 100 * np.mean([r["eval"]["clear_rate"] for r in rs]),
                    "train cleared % (first→last third)": f"{100*np.mean([r['train_phases'][0]['clear_rate'] for r in rs]):.1f} → "
                                                          f"{100*np.mean([r['train_phases'][-1]['clear_rate'] for r in rs]):.1f}"})
    return pd.DataFrame(out).set_index("arm").round(2)

def by_type(path):
    rows = json.load(open(path))["rows"]
    agg = {}
    for r in rows:
        for k, (ok, bad) in r["eval"]["by_type"].items():
            a = agg.setdefault(r["arm"], {}).setdefault(k, [0, 0]); a[0] += ok; a[1] += bad
    return pd.DataFrame({arm: {k: round(100 * v[0] / max(1, sum(v))) for k, v in t.items()} for arm, t in agg.items()})

def plot_trex(path, title):
    t = trex_table(path)
    arms, m, e = list(t.index), t["deaths/min"].values, t["± std"].values
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(arms) + 1.2))
    ax.barh(arms, m, xerr=e if t["seeds"].max() > 1 else None, height=0.6,
            color=[REF if a in REFERENCE_ARMS else LEARN for a in arms], error_kw={"ecolor": MUTED, "lw": 1})
    for i, a in enumerate(arms):
        ax.text(m[i] + (e[i] if t["seeds"].max() > 1 else 0) + 0.1, i,
                f"{m[i]:.2f}  ({t.loc[a, 'cleared %']:.1f}% cleared)", va="center", color=INK, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, max(m + e) * 1.45)
    ax.set_xlabel("deaths per minute on unseen courses (lower is better)", color=MUTED)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    plt.tight_layout(); plt.show()
    return t
""")
md("The published 5-seed result (from `docs/RESEARCH.md`), for reference:")
code("""
plot_trex("experiments/results/trex_online.json", "T-Rex, CPU stand-in encoder, 5 seeds (published run)")
""")

md("""
## A. T-Rex testbed (CPU stand-in encoder)

CLM's T-Rex game, headless, with a **neutral prompt** (the scene and the actions only, no "Safe/Unsafe" hints).
Each obstacle is an episode: cleared = success, crash = failure. The base head was trained only on single cacti
(an "easy course"); it never saw groups or birds. During the online phase the agent learns from its own clears and
crashes; evaluation is greedy on unseen courses.

Arms: `frozen` · `map` (outcome map only, no training) · `lora` (gated LoRA from success/failure) · `lora+map` ·
`all-positive-ungated` (every executed step treated as correct = the reward ignored).
""")
code("""
A_FRAMES = 6000 if QUICK else 30000
!python testbeds/trex/run_online.py --seeds 0 --arms frozen map lora lora+map all-positive-ungated \\
    --train-frames {A_FRAMES} --eval-seeds 100 101 --eval-frames 3600 --update-every 10 --out runs/trex_quick.json
""")
code("""
display(plot_trex("runs/trex_quick.json", "T-Rex, CPU stand-in encoder, 1 seed (this run)"))
display(by_type("runs/trex_quick.json"))  # % of obstacles cleared, by obstacle type
""")

md("""
## B. Synthetic toy: which objectives work

States/actions are random vectors; the right action is `argmax sᵀW*a`; the base head learned a shifted rule.
Compares: frozen, learning with the reward ignored, successes only, CE + unlikelihood on failures (`ce`), and the
same with every step of a failed episode blamed (`ce-credit-all`).
""")
code("""
B_EPISODES = 300 if QUICK else 900
!python experiments/synthetic_online.py --seeds 0 --episodes {B_EPISODES} \\
    --arms frozen all-positive-ungated success-only ce ce-credit-all --out runs/synthetic_quick.json
""")

md("""
## C. T-Rex with the real CLM-8B (GPU ≥ 24 GB)

The real encoder (Qwen3-8B, run in-process with `transformers`: last-token pooling, as `clm-serve`'s vLLM embedder)
and the **released CLM head** (`CLM_v0.1-8B.pt`, downloaded from Hugging Face) as the base, zero-shot on the
neutral prompt. First run downloads ~16 GB of weights. Embeddings are cached per text, so later arms are faster.

`--encoder clm --emb-url ...` does the same against a vLLM pooling server, as in CLM's README, if you prefer it.
""")
code("""
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 22e9, \\
    "Part C needs an L4 or A100 runtime (Qwen3-8B in bf16 needs ~17 GB of GPU memory)."
C_FRAMES = 6000 if QUICK else 30000
!python testbeds/trex/run_online.py --encoder hf --seeds 0 --arms frozen map lora lora+map \\
    --train-frames {C_FRAMES} --eval-seeds 100 101 --eval-frames 3600 --update-every 10 --rank 16 --lr 1e-3 \\
    --out runs/trex_clm8b.json
""")
code("""
display(plot_trex("runs/trex_clm8b.json", "T-Rex, real CLM-8B (Qwen3-8B + released head), 1 seed"))
display(by_type("runs/trex_clm8b.json"))
""")

md("""
## D. Phase 0: DeepSWE with the reward in the loop

CLM's `train/finetune.py` trains its DeepSWE verifier on every step as a positive; the pass/fail `reward` stored with
each step is not used by the loss. `experiments/deepswe_outcome.py` trains on the **same published embeddings**,
from the **same** warm start, with:

* `clm`: CLM's loss reproduced (the control),
* `outcome`: CLM's loss + a per-task contrast: within each training task, passing trajectories must outscore
  failing ones, with the exact score `bon_eval.py` selects by (mean step cosine over the final 12 steps).

Every head is scored by CLM's **own, unmodified** `evaluation/bon_eval.py` on the same 38 held-out tasks
(`--n 4 --window 12`), next to the published head (31/38 = 81.6%) and the zero-shot base.
Only the projection heads train (embeddings are precomputed), so the GPU is for speed, not memory.
""")
code("""
from huggingface_hub import snapshot_download
from clm.heads import download
HEADS = snapshot_download("Contrastive-LM/deepswe-clm-heads-8k", local_dir="heads/deepswe")
INIT = download()                      # CLM_v0.1-8B.pt, the warm start (and the zero-shot reference)
print(sorted(os.listdir(HEADS)), "\\nwarm start:", INIT)
os.makedirs("runs/phase0", exist_ok=True)
BON = "python /content/CLM/evaluation/bon_eval.py --hf-dataset Contrastive-LM/deepswe-clm-embeddings-8k " \\
      "--tasks-file heads/deepswe/heldout_tasks.json --n 4 --window 12"
""")
md("References: the published DeepSWE head (expected 31/38) and the zero-shot base head.")
code("""
!{BON} --checkpoint heads/deepswe/best_head.pt --output runs/phase0/bon_published.json
!{BON} --checkpoint {INIT} --output runs/phase0/bon_zero-shot.json
""")
md("The training embeddings (size printed after download; they stay on disk under `data/`).")
code("""
!python /content/CLM/preprocessing/hf_embeddings.py download Contrastive-LM/deepswe-clm-train-embeddings-8k --out data/deepswe_train
!du -sh data/deepswe_train
""")
md("""
Train and score each objective. `STEPS` bounds an epoch so a run fits a Colab session; raise it (or drop
`--steps-per-epoch`) for a full pass. Add `"success"` / `"outcome-success"` to `OBJECTIVES`, or `--lora-rank 0`
(full heads), for more arms.
""")
code("""
OBJECTIVES = ["clm", "outcome"]
STEPS, EPOCHS = (20, 1) if QUICK else (300, 10)
for obj in OBJECTIVES:
    !python experiments/deepswe_outcome.py --emb-dir data/deepswe_train --init-ckpt {INIT} \\
        --holdout-tasks heads/deepswe/heldout_tasks.json --objective {obj} --lora-rank 16 \\
        --epochs {EPOCHS} --steps-per-epoch {STEPS} --out-dir runs/phase0/{obj}
    !{BON} --checkpoint runs/phase0/{obj}/best_head.pt --output runs/phase0/bon_{obj}.json
""")
code("""
import glob
rows = []
for f in sorted(glob.glob("runs/phase0/bon_*.json")):
    d = json.load(open(f)); sel = next(iter(d["selectors"].values()))
    rows.append({"head": os.path.basename(f)[4:-5], "resolved": f"{sel['resolved']}/{sel['n_tasks']}",
                 "best-of-4 %": round(100 * sel["rate"], 1), "random pick %": round(100 * d["random_pick"], 1),
                 "oracle %": round(100 * d["oracle_any"], 1)})
pd.DataFrame(rows).set_index("head")
""")
md("""
Optional: CLM's own training script as a second control (it should land near the `clm` row):
""")
code("""
# !python /content/CLM/train/finetune.py --task clm --emb-dir data/deepswe_train --init-ckpt {INIT} \\
#     --holdout-tasks heads/deepswe/heldout_tasks.json --batch 512 --out-dir runs/phase0/clm_official
# !{BON} --checkpoint runs/phase0/clm_official/best_head.pt --output runs/phase0/bon_clm_official.json
""")

md("## Save the results")
code("""
!zip -qr /content/clm_autoevolve_runs.zip runs -x "*.pt"
try:
    from google.colab import files
    files.download("/content/clm_autoevolve_runs.zip")
except Exception:
    print("results in /content/clm_autoevolve_runs.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
