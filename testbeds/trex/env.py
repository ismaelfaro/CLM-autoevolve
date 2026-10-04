"""Headless T-Rex for online learning: CLM decides, obstacles are cleared or crashed into.

The game is CLM's own T-Rex example (vendored ``engine.py``), played in lockstep: the game waits
for every answer, and a decision is asked every ``frames_per_decision`` frames.

Unlike the example's ``labeled`` prompt, which writes "Safe… Best" into the options, the state here
only describes the scene and the options only describe the actions.  The model has to *learn*
which action works where, which is the point of a learning testbed:

    state:  Dino runner game. 2 large cacti ahead, close, about 100 px away. Speed 9.
            Which action should the dinosaur take now?
    jump:   Jump: leap up and over what is ahead.
    duck:   Duck: crouch low under it, or drop fast while in the air.
    run:    Run: keep running and wait.

Episodes are obstacle encounters: every decision taken while an obstacle is the nearest one ahead
belongs to that obstacle's episode, which ends in success when the dino is past it and in
failure when the dino hits it.  A crash restarts the course at once (no 1.5 s pause).  Each
life starts at a random speed in ``speed_range`` so birds (speed >= 8.5) appear in every life,
not only after 40 s of survival.

The vendored planner is used **only** as an oracle for evaluation (agreement with its best
action) and for building the stand-in base head; it never labels training data in the
online loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .engine import GROUND_Y, TREX_WIDTH, Game, Spec
from .planner import Planner, snapshot

ACTIONS = ("jump", "duck", "run")
OPTIONS = {
    "jump": "Jump: leap up and over what is ahead.",
    "duck": "Duck: crouch low under it, or drop fast while in the air.",
    "run": "Run: keep running and wait.",
}
QUESTION = "Which action should the dinosaur take now?"


def closeness(d: float) -> str:
    return "very close" if d < 60 else "close" if d < 130 else "near" if d < 220 else "far"


def describe(game: Game):
    """-> (state text, nearest obstacle ahead or None)."""
    trex = game.trex
    ahead = [o for o in game.obstacles if not o.remove and o.x + o.width > trex.x]
    if not ahead or not game.has_obstacles:
        return f"Dino runner game. Nothing ahead. Speed {round(game.speed)}. {QUESTION}", None
    o = ahead[0]
    d = max(0, o.x - (trex.x + TREX_WIDTH))
    text = (f"Dino runner game. {o.label[0].upper()}{o.label[1:]} ahead, {closeness(d)}, "
            f"about {25 * round(d / 25)} px away. Speed {round(game.speed)}.")
    if trex.jumping:
        up = GROUND_Y - trex.y
        text += f" The dino is in the air, about {15 * round(up / 15)} px up, " \
                f"{'falling' if trex.velocity > 0 else 'rising'}."
    elif trex.ducking:
        text += " The dino is ducking."
    return f"{text} {QUESTION}", o


class EasyCourse:
    """Single cacti only, no groups, no birds: the stand-in base head's 'pre-training' regime."""

    def __init__(self, seed):
        self.seed = seed

    def spec(self, game, index, source="random"):
        rng = random.Random(f"easy-{self.seed}-{game.run_index}-{index}")
        return Spec(rng.choice(["cactusSmall", "cactusLarge"]), 1, 0, rng.random(), False, source)


@dataclass
class Tally:
    frames: int = 0
    decisions: int = 0
    deaths: int = 0
    cleared: int = 0
    oracle_agree: int = 0
    oracle_seen: int = 0
    by_type: dict = field(default_factory=dict)       # label -> [cleared, crashed]

    def encounter(self, label: str, ok: bool):
        c = self.by_type.setdefault(label, [0, 0])
        c[0 if ok else 1] += 1
        if ok:
            self.cleared += 1


class TRexEnv:
    def __init__(self, seed: int, frames_per_decision: int = 4, speed_range=(6.0, 10.0), course=None,
                 oracle: bool = False):
        self.seed, self.k, self.speed_range = seed, frames_per_decision, speed_range
        self.game = Game(seed, course=course)
        self.rng = random.Random(f"speeds-{seed}")
        self.held = "run"
        self.planner = Planner() if oracle else None
        self.open: dict[int, str] = {}     # obstacle id -> episode id
        self.tally = Tally()
        self.game.press_jump()             # starts the run (the original's first key press)
        self._set_speed()

    def _set_speed(self):
        if self.speed_range:
            self.game.speed = self.rng.uniform(*self.speed_range)

    def episode_id(self, o) -> str:
        return f"{self.seed}-{self.game.run_index}-{o.id}"

    # the pilot's key handling (examples/t_rex/trex/pilot.py: apply / enforce)
    def apply(self, action: str):
        g = self.game
        if action == "jump":
            if not g.trex.jumping:
                if g.trex.ducking:
                    g.release_duck()
                g.press_jump()
            self.held = "run"
        else:
            self.held = action

    def enforce(self):
        g, t = self.game, self.game.trex
        if not g.playing or g.crashed:
            return
        if self.held == "duck":
            if t.jumping:
                if not t.speed_drop:
                    g.press_duck()
            elif not t.ducking:
                g.press_duck()
        elif t.speed_drop or not t.jumping and t.ducking:
            g.release_duck()

    def play(self, agent, frames: int):
        """Run ``frames`` game frames with ``agent.decide(text, task, episode) -> action`` and
        ``agent.outcome(episode, success)``; returns the tally."""
        g, since = self.game, self.k
        end = self.tally.frames + frames
        while self.tally.frames < end:
            if g.crashed:
                first = g.obstacles[0] if g.obstacles else None
                if first is not None:
                    self.tally.encounter(first.label, False)
                    ep = self.open.pop(first.id, None)
                    if ep:
                        agent.outcome(ep, False)
                for ep in self.open.values():      # decisions about obstacles never reached
                    agent.discard(ep)
                self.open.clear()
                self.tally.deaths += 1
                g.restart()
                self.held, since = "run", self.k
                self._set_speed()
                continue
            for o in g.obstacles:
                if o.x + o.width < g.trex.x and o.id in self.open:
                    self.tally.encounter(o.label, True)
                    agent.outcome(self.open.pop(o.id), True)
            if since >= self.k and g.playing:
                text, o = describe(g)
                if o is None:
                    action = "run"
                else:
                    ep = self.open.setdefault(o.id, self.episode_id(o))
                    action = agent.decide(text, o.label, ep)
                    self.tally.decisions += 1
                    if self.planner is not None:
                        best = self.planner.plan(snapshot(g, self.held), (0, 0), (self.k, self.k)).best
                        airborne = g.trex.jumping
                        self.tally.oracle_seen += 1
                        self.tally.oracle_agree += (action == best or airborne and {action, best} <= {"jump", "run"})
                self.apply(action)
                since = 0
            self.enforce()
            g.step()
            since += 1
            self.tally.frames += 1
        return self.tally
