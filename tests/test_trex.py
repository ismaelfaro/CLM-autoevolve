import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoevolve import HashEncoder, LoRAHeads, OutcomeMap  # noqa: E402
from autoevolve.lora import random_checkpoint  # noqa: E402
from testbeds.trex.env import ACTIONS, QUESTION, EasyCourse, TRexEnv, describe  # noqa: E402
from testbeds.trex.run_online import Agent  # noqa: E402


class Recorder:
    def __init__(self, policy):
        self.policy, self.decisions, self.outcomes, self.discarded = policy, [], [], []

    def decide(self, text, label, ep):
        self.decisions.append((text, label, ep))
        return self.policy(text)

    def outcome(self, ep, ok):
        self.outcomes.append((ep, ok))

    def discard(self, ep):
        self.discarded.append(ep)


def test_describe_never_names_the_answer():
    env = TRexEnv(3)
    seen = []
    rec = Recorder(lambda t: seen.append(t) or "run")
    env.play(rec, 1500)
    assert seen and all(t.endswith(QUESTION) for t in seen)
    assert not any(w in t.lower() for t in seen for w in ("safe", "best", "collision"))
    text, o = describe(env.game)
    assert (o is None) == ("Nothing ahead" in text)


def test_running_only_crashes_and_episodes_close_once():
    env = TRexEnv(5, speed_range=None)
    rec = Recorder(lambda t: "run")
    tally = env.play(rec, 3000)
    assert tally.deaths >= 1
    eps = [e for e, _ in rec.outcomes]
    assert len(eps) == len(set(eps))                     # one outcome per obstacle encounter
    assert sum(not ok for _, ok in rec.outcomes) <= tally.deaths
    decided = {e for _, _, e in rec.decisions}
    assert set(eps) | set(rec.discarded) <= decided


def test_oracle_policy_mostly_survives_easy_course():
    env = TRexEnv(7, course=EasyCourse(7), oracle=True, speed_range=None)
    from testbeds.trex.planner import snapshot

    class Oracle(Recorder):
        def decide(self, text, label, ep):
            return env.planner.plan(snapshot(env.game, env.held), (0, 0), (env.k, env.k)).best
    tally = env.play(Oracle(None), 3600)
    assert tally.cleared > 10 and tally.deaths <= 1


def test_agent_learns_into_the_outcome_map():
    enc = HashEncoder(64)
    head = LoRAHeads(random_checkpoint(width=64, proj=16, hidden=64), rank=1).eval()
    omap = OutcomeMap(dim=64)
    agent = Agent(enc, head, None, omap, seed=0)
    env = TRexEnv(11)
    env.play(agent, 4000)
    assert len(omap) > 0 and len(agent.val.items) > 0
    assert set(np.unique([i.d.chosen for i in agent.train.items])) <= set(range(len(ACTIONS)))


def test_acting_temperature_tames_a_large_logit_scale():
    import math
    import torch
    from testbeds.trex.run_online import act_temperature
    ck = random_checkpoint(width=64, proj=16, hidden=64)
    assert act_temperature(LoRAHeads(ck, rank=1), "auto") == 1.0        # stand-in heads: unchanged
    ck["logit_scale"] = torch.tensor(math.log(100.0))                    # a confident head, like CLM's
    head = LoRAHeads(ck, rank=1).eval()
    t = act_temperature(head, "auto")
    assert t == 7.0 and act_temperature(head, "2.5") == 2.5
    s, c = torch.randn(64, 64), torch.randn(64, 3, 64)
    raw = torch.softmax(head.candidate_logits(s, c), -1).max(-1).values.mean()
    tamed = torch.softmax(head.candidate_logits(s, c) / t, -1).max(-1).values.mean()
    assert tamed < raw                                                   # less one-hot: room to explore
