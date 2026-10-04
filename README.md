# CLM-autoevolve

Research and a prototype for an automatic training loop for
[Contrastive Language Models (CLM)](https://github.com/Contrastive-LM/CLM). CLM makes
state → action decisions, the environment reports success or failure, and CLM improves on the
fly through LoRA adapters on its projection heads. Each adapter is merged and published as a
standard checkpoint that `clm-serve` hot-reloads.

* **Research note:** [docs/RESEARCH.md](docs/RESEARCH.md)
* **Prototype:** `autoevolve/`: LoRA heads with merged export, outcome-aware objectives,
  experience buffer with credit assignment, and an online learner with a gate and rollback.
* **Toy experiment:** `experiments/synthetic_online.py`

```bash
pip install torch numpy pytest contrastive-lm
python -m pytest -q tests
python experiments/synthetic_online.py --seeds 0 1 2
```
