import os
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clm.heads import HeadPair, make_head

from autoevolve import Decision, ExperienceBuffer, LearnerConfig, LoRAHeads, OnlineLearner, assign_credit
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


def test_gate_rolls_back_a_worse_update(ckpt, tmp_path):
    lr = OnlineLearner(ckpt, str(tmp_path / "h.pt"), LearnerConfig(rank=4))
    scores = iter([0.5, 0.4])                  # the candidate is scored first, then the base adapter
    _perturb(lr.model)
    assert lr.maybe_publish(lambda _: next(scores)) is True
    good = lr.model.adapter_state()
    _perturb(lr.model, seed=9)
    assert lr.maybe_publish(lambda _: 0.1) is False
    for k, v in lr.model.adapter_state().items():
        assert torch.equal(v, good[k])         # rolled back to the last promoted adapter
    assert (tmp_path / "h.pt").exists() and (tmp_path / "versions" / "adapter_v0001.pt").exists()


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
