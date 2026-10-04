import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_deepswe  # noqa: E402
from clm.heads import HeadPair  # noqa: E402

from autoevolve.lora import random_checkpoint  # noqa: E402
from experiments.deepswe_outcome import Data, best_of_n  # noqa: E402


def test_best_of_n_matches_brute_force():
    rng = random.Random(0)
    for _ in range(50):
        c = [(rng.choice([0.1, 0.2, 0.3, 0.4]), rng.randint(0, 1)) for _ in range(rng.randint(2, 7))]
        n = rng.randint(1, len(c))
        # every N-subset equally likely; ties broken uniformly among the top-scoring members
        from itertools import combinations
        subsets = list(combinations(range(len(c)), n))
        exp = 0.0
        for sub in subsets:
            top = max(c[i][0] for i in sub)
            winners = [c[i][1] for i in sub if c[i][0] == top]
            exp += sum(winners) / len(winners)
        assert math.isclose(best_of_n(c, n), exp / len(subsets), rel_tol=1e-9)


def test_phase0_trains_and_exports_a_servable_clm_head(tmp_path):
    d = fake_deepswe.make(str(tmp_path / "emb"), hidden=32, tasks=30, trajs=4, steps=8)
    init = tmp_path / "init.pt"
    torch.save(random_checkpoint(width=64, depth=3, proj=16, hidden=32, seed=0), init)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "deepswe_outcome.py"),
                        "--emb-dir", d, "--init-ckpt", str(init), "--holdout-tasks", os.path.join(d, "heldout.json"),
                        "--objective", "outcome", "--epochs", "2", "--batch", "64", "--window", "4",
                        "--lora-rank", "4", "--device", "cpu", "--out-dir", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    summary = json.load(open(out / "summary.json"))
    assert summary["history"][0]["epoch"] == 0 and len(summary["history"]) >= 2
    hp = HeadPair("phase0", str(out / "best_head.pt"), device="cpu").ensure()
    zs, za = hp.project(np.random.randn(3, 32).astype(np.float32), np.random.randn(2, 32).astype(np.float32))
    assert zs.shape == (3, 16) and za.shape == (2, 16)
    data = Data(d)
    held = set(json.load(open(os.path.join(d, "heldout.json"))))
    assert held and all(t.startswith("task") for t in held)
    assert len(data.traj) == 30 * 4 and all(len(v) == 8 for v in data.traj.values())
