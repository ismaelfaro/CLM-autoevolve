"""A small embedding dir in the exact format CLM's DeepSWE datasets use (hf_embeddings.py), with a
planted signal: passing trajectories' final steps have actions aligned with their states."""
import json
import os

import numpy as np
import torch


def make(out_dir, hidden=64, tasks=40, trajs=6, steps=16, seed=0):
    r = np.random.default_rng(seed)
    W = r.standard_normal((hidden, hidden)) / np.sqrt(hidden)
    S, A, samples = [], [], []
    for t in range(tasks):
        n_pass = int(r.integers(0, trajs + 1))
        for j in range(trajs):
            passed = j < n_pass
            for k in range(steps):
                s = r.standard_normal(hidden)
                late = k >= steps - 6
                a = (s @ W if (passed and late) else 0) + r.standard_normal(hidden) * (0.8 if late else 1.0)
                S.append(s / np.linalg.norm(s)); A.append(a / np.linalg.norm(a))
                samples.append({"trajectory_id": f"task{t}__traj{j}", "step_idx": k, "task_id": f"task{t}",
                                "reward": float(passed), "model": "fake", "config": "default"})
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.tensor(np.stack(S), dtype=torch.float16), os.path.join(out_dir, "state_embeddings.pt"))
    torch.save(torch.tensor(np.stack(A), dtype=torch.float16), os.path.join(out_dir, "action_embeddings.pt"))
    json.dump({"num_samples": len(samples), "hidden_size": hidden, "samples": samples},
              open(os.path.join(out_dir, "metadata.json"), "w"))
    json.dump([f"task{t}" for t in range(tasks - 8, tasks)], open(os.path.join(out_dir, "heldout.json"), "w"))
    return out_dir
