import os
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clm.heads import HeadPair, make_head

from autoevolve import (Decision, ExperienceBuffer, HashEncoder, LearnerConfig, LoRAHeads, OnlineLearner, OutcomeMap,
                        assign_credit, snips)
from autoevolve.experience import Labeled
from autoevolve import objectives as obj
from autoevolve.lora import atomic_save, random_checkpoint

HID = 128


@pytest.fixture
def ckpt():
    return random_checkpoint(hidden=HID, seed=0)


def _base_heads(ck):
    cfg = ck["cfg"]
    kw = dict(width=cfg["width"], depth=cfg["depth"], proj=cfg["projection_dim"], activation="gelu",
              layernorm=cfg["layernorm"], residual=False, hidden=cfg["hidden_size"])
    sh, ah = make_head(**kw), make_head(**kw)
    sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    return sh.eval(), ah.eval()


def _perturb(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def test_adapter_starts_as_noop(ckpt):
    m = LoRAHeads(ckpt, rank=8).eval()
    sh, _ = _base_heads(ckpt)
    x = torch.randn(5, HID)
    assert torch.allclose(m.state_head(x), sh(x), atol=1e-6)


def test_only_adapters_train(ckpt):
    m = LoRAHeads(ckpt, rank=8)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    trainable = sum(p.numel() for p in m.trainable_parameters())
    total = sum(p.numel() for p in m.parameters())
    assert trainable < 0.2 * total


def test_merged_checkpoint_serves_identically_through_clm_headpair(ckpt, tmp_path):
    m = LoRAHeads(ckpt, rank=8).eval()
    _perturb(m)
    path = tmp_path / "clm-online.pt"
    atomic_save(m.merged_checkpoint({"version": 1}), str(path))
    hp = HeadPair("clm-online", str(path), device="cpu").ensure()
    s, a = np.random.randn(4, HID).astype(np.float32), np.random.randn(6, HID).astype(np.float32)
    zs, za = hp.project(s, a)
    with torch.no_grad():
        ref_s = m.encode_states(torch.from_numpy(s)).numpy()
        ref_a = m.encode_actions(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(zs, ref_s, atol=1e-5)
    np.testing.assert_allclose(za, ref_a, atol=1e-5)
    assert hp.scale == pytest.approx(float(m.scale), rel=1e-6)


def test_publish_triggers_hot_reload(ckpt, tmp_path):
    path = str(tmp_path / "clm-online.pt")
    m = LoRAHeads(ckpt, rank=8).eval()
    atomic_save(m.merged_checkpoint(), path)
    hp = HeadPair("clm-online", path, device="cpu").ensure()
    gen = hp.generation
    _perturb(m, seed=3)
    time.sleep(0.01)
    atomic_save(m.merged_checkpoint(), path)
    os.utime(path, (time.time() + 1, time.time() + 1))   # coarse-mtime filesystems
    hp.ensure()
    assert hp.generation == gen + 1                       # new generation => vector cache misses


def test_reference_context_is_base(ckpt):
    m = LoRAHeads(ckpt, rank=8, train_scale=True).eval()
    _perturb(m)
    with torch.no_grad():
        m.logit_scale += 0.5
    sh, _ = _base_heads(ckpt)
    x = torch.randn(3, HID)
    with torch.no_grad(), m.reference():
        assert torch.allclose(m.state_head(x), sh(x), atol=1e-6)
        assert float(m.logit_scale) == pytest.approx(float(ckpt["logit_scale"]))
    assert not torch.allclose(m.state_head(x), sh(x), atol=1e-4)


def _dec(step, group=0, reward=None):
    return Decision("e", step, np.zeros(4), np.zeros((2, 4)), 0, np.array([.5, .5]), group, step_reward=reward)


def test_credit_assignment():
    ds = [_dec(t) for t in range(5)]
    assert [l.success for l in assign_credit(ds, 1.0, "terminal")] == [True] * 5
    fail = assign_credit(ds, 0.0, "terminal", window=2)
    assert [l.d.step for l in fail] == [3, 4] and not any(l.success for l in fail)
    disc = assign_credit(ds, 0.0, "discounted", gamma=0.5)
    assert [l.weight for l in disc] == [0.0625, 0.125, 0.25, 0.5, 1.0]
    dense = assign_credit([_dec(0, reward=1.0), _dec(1)], 0.0, "terminal")
    assert [l.success for l in dense] == [True, False]


def test_outcome_infonce_pushes_failure_below_success(ckpt):
    torch.manual_seed(0)
    m = LoRAHeads(ckpt, rank=8)
    s = torch.randn(1, HID).repeat(2, 1)
    a = torch.randn(2, HID)
    noise_s, noise_a = torch.randn(30, HID), torch.randn(30, HID)
    S, A = torch.cat([s, noise_s]), torch.cat([a, noise_a])
    ok = torch.tensor([True, False] + [True] * 30)
    grp = torch.tensor([0, 0] + list(range(1, 31)))
    opt = torch.optim.Adam(m.trainable_parameters(), lr=5e-3)

    def margin():
        with torch.no_grad():
            return float(m.pair_logits(s[:1], a[:1]) - m.pair_logits(s[:1], a[1:]))
    before = margin()
    for _ in range(30):
        loss = obj.outcome_infonce(m, S, A, ok, grp)
        opt.zero_grad(); loss.backward(); opt.step()
    assert margin() > before + 1.0


def test_bandit_ppo_raises_prob_of_rewarded_action(ckpt):
    torch.manual_seed(0)
    m = LoRAHeads(ckpt, rank=8)
    s, c = torch.randn(1, HID), torch.randn(1, 3, HID)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        p0 = F.softmax(m.candidate_logits(s, c), -1)[0]
    opt = torch.optim.Adam(m.trainable_parameters(), lr=1e-3)
    for _ in range(3):
        loss = obj.bandit_ppo(m, s, c, mask, torch.tensor([2]), p0[2:3], torch.tensor([1.0]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p1 = F.softmax(m.candidate_logits(s, c), -1)[0]
    assert p1[2] > p0[2]


def test_gate_promotes_only_a_better_candidate(ckpt, tmp_path):
    lr = OnlineLearner(ckpt, str(tmp_path / "h.pt"), LearnerConfig(rank=4))
    _perturb(lr.model)
    score = lambda m: float(m.adapter_state()["state_head.inp.lora_B"].abs().sum())  # noqa: E731
    assert lr.maybe_publish(score) is True                    # perturbed beats the zero adapter
    assert torch.equal(lr.serving.adapter_state()["state_head.inp.lora_B"],
                       lr.model.adapter_state()["state_head.inp.lora_B"])
    assert (tmp_path / "h.pt").exists() and (tmp_path / "versions" / "adapter_v0001.pt").exists()
    served = lr.serving.adapter_state()
    with torch.no_grad():
        for n, p in lr.model.named_parameters():
            if "lora_B" in n:
                p.mul_(0.5)                                       # candidate now scores lower
    assert lr.maybe_publish(score) is False
    for k, v in lr.serving.adapter_state().items():
        assert torch.equal(v, served[k])                      # serving untouched


def test_gate_rolls_back_after_sustained_regression(ckpt):
    lr = OnlineLearner(ckpt, None, LearnerConfig(rank=4, rollback_patience=2, rollback_tolerance=0.0))
    _perturb(lr.model)
    lr.maybe_publish(lambda m: 1.0 if m is lr.model else 0.0)
    served = lr.serving.adapter_state()
    _perturb(lr.model, seed=7)
    worse = lambda m: 0.0 if m is lr.model else 1.0  # noqa: E731
    lr.maybe_publish(worse)
    assert not torch.equal(lr.model.adapter_state()["state_head.inp.lora_B"], served["state_head.inp.lora_B"])
    lr.maybe_publish(worse)                                   # second bad gate in a row => reset
    assert lr.stats.rollbacks == 1
    assert torch.equal(lr.model.adapter_state()["state_head.inp.lora_B"], served["state_head.inp.lora_B"])


def test_ungated_updates_go_straight_to_serving(ckpt):
    lr = OnlineLearner(ckpt, None, LearnerConfig(rank=4, gate=False, losses=("ce",), batch=8, steps_per_update=3,
                                                 replay_weight=0, kl_weight=0))
    buf = ExperienceBuffer()
    rng = np.random.default_rng(0)
    for e in range(6):
        buf.record(Decision(f"e{e}", 0, rng.standard_normal(HID), rng.standard_normal((3, HID)), 0,
                            np.full(3, 1 / 3), group=e))
        buf.close(f"e{e}", 1.0)
    lr.update(buf)
    assert lr.stats.published == 1
    assert torch.equal(lr.serving.adapter_state()["state_head.inp.lora_B"],
                       lr.model.adapter_state()["state_head.inp.lora_B"])


def test_outcome_map_posterior_and_bias():
    m = OutcomeMap(dim=4, k=16, tau_state=0.05, tau_action=0.02)
    s = np.eye(4)[0]; good, bad = np.eye(4)[1], np.eye(4)[2]
    m.add([s] * 6, [good] * 3 + [bad] * 3, [1, 1, 1, 0, 0, 0], tasks=[0] * 6)
    q = torch.tensor(s, dtype=torch.float32)[None]
    c = torch.tensor(np.stack([good, bad, np.eye(4)[3]]), dtype=torch.float32)[None]
    p, n = m.query(q, c)
    assert p[0, 0] > 0.75 and p[0, 1] < 0.25 and abs(float(p[0, 2]) - 0.5) < 0.05   # unseen action: prior
    assert n[0, 0] == pytest.approx(3.0, rel=1e-3) and float(n[0, 2]) < 1e-3
    b = m.bias(q, c)
    assert b[0, 0] > 0 > b[0, 1] and abs(float(b[0, 2])) < 1e-2
    far = torch.tensor(np.eye(4)[3], dtype=torch.float32)[None]
    assert float(m.query(far, c)[1].max()) < 1e-6           # dissimilar state: no evidence borrowed
    t, conf = m.soft_targets(q, c, torch.ones(1, 3, dtype=torch.bool))
    assert t[0].argmax() == 0 and 0 < float(conf[0]) < 1


def test_outcome_map_task_bonus_and_ring_buffer():
    m = OutcomeMap(dim=2, capacity=4, k=8, task_bonus=3.0)
    s, a = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    m.add([s, s], [a, a], [1, 0], tasks=[0, 1])
    q, c = torch.tensor(s, dtype=torch.float32)[None], torch.tensor(a, dtype=torch.float32)[None, None]
    p0, _ = m.query(q, c, task=torch.tensor([0]))
    p1, _ = m.query(q, c, task=torch.tensor([1]))
    assert p0[0, 0] > 0.5 > p1[0, 0]                        # each task leans on its own history
    m.add([s] * 5, [a] * 5, [0] * 5)
    assert len(m) == 4 and float(m.query(q, c)[0][0, 0]) < 0.25   # oldest rows overwritten


def test_hash_encoder_is_deterministic_and_similarity_aware():
    e1, e2 = HashEncoder(dim=64), HashEncoder(dim=64)
    a, b, c = e1.embed(["large cactus ahead, close", "large cactus ahead, far", "bird at head height"])
    assert np.allclose(a, e2.embed(["large cactus ahead, close"])[0])
    assert float(a @ b) > float(a @ c)


def test_snips_prefers_the_policy_that_picks_successes(ckpt):
    m = LoRAHeads(ckpt, rank=4).eval()
    rng = np.random.default_rng(0)
    items = []
    for e in range(40):
        st, cands = rng.standard_normal(HID), rng.standard_normal((3, HID))
        ch = int(rng.integers(3))
        d = Decision(f"e{e}", 0, st, cands, ch, np.full(3, 1 / 3), group=e)
        items.append(Labeled(d, success=bool(e % 2), weight=1.0))
    good = lambda b: torch.where(b["success"], 20.0, -20.0)[:, None] * F.one_hot(b["chosen"], 3)  # noqa: E731
    bad = lambda b: -good(b)  # noqa: E731
    assert snips(m, items, good) > 0.9 > 0.1 > snips(m, items, bad)


def test_learner_update_runs_with_all_losses(ckpt):
    cfg = LearnerConfig(rank=4, losses=("infonce", "bandit", "dpo"), batch=16, steps_per_update=2)
    replay = (torch.randn(64, HID), torch.randn(64, HID))
    anchor = (torch.randn(32, HID), torch.randn(32, 3, HID), torch.ones(32, 3, dtype=torch.bool))
    lr = OnlineLearner(ckpt, None, cfg, replay=replay, anchor=anchor)
    buf = ExperienceBuffer()
    rng = np.random.default_rng(0)
    for e in range(8):
        for t in range(3):
            buf.record(Decision(f"e{e}", t, rng.standard_normal(HID), rng.standard_normal((3, HID)),
                                int(rng.integers(3)), np.full(3, 1 / 3), group=t))
        buf.close(f"e{e}", float(e % 2))
    logs = lr.update(buf)
    assert {"infonce", "bandit", "replay", "kl"} <= set(logs)
