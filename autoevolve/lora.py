"""LoRA adapters on CLM projection heads, exported back to a plain CLM checkpoint.

CLM scores a (state, action) pair as ``exp(logit_scale) * cos(state_head(s), action_head(a))``
where ``s`` and ``a`` are embeddings from a frozen Qwen3-8B encoder.  Only the two heads
(``hidden -> width -> ... -> proj`` MLPs, ~20M parameters together) are trainable, so the
adapter goes on their ``nn.Linear`` layers; the encoder, and therefore every cached embedding,
stays valid.

``LoRAHeads.merged_checkpoint()`` folds ``W + (alpha / r) * B @ A`` into each layer and returns a
dict in exactly the format ``clm.heads.HeadPair`` loads, so ``clm-serve --ckpt PATH`` (or
``--model NAME=PATH``) picks an update up through its mtime hot-reload without any change to
the server, and its vector cache invalidates itself through the head generation.
"""
from __future__ import annotations

import contextlib
import copy
import math
import os
import tempfile
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.heads import HIDDEN, PROJ_DIM, make_head

TARGETS = ("inp", "hidden", "out")


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) * x A^T B^T`` with ``base`` frozen; B starts at zero so the
    adapter is a no-op until trained."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return y

    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += self.scaling * (self.lora_B @ self.lora_A)
        return lin


def _swap(module: nn.Module, name: str, rank: int, alpha: float, dropout: float) -> None:
    child = getattr(module, name)
    if isinstance(child, nn.ModuleList):
        for i, lin in enumerate(child):
            child[i] = LoRALinear(lin, rank, alpha, dropout)
    else:
        setattr(module, name, LoRALinear(child, rank, alpha, dropout))


def _lora_layers(module: nn.Module) -> Iterator[LoRALinear]:
    return (m for m in module.modules() if isinstance(m, LoRALinear))


class LoRAHeads(nn.Module):
    """State + action heads from a CLM checkpoint with LoRA on their linear layers.

    ``train_scale`` also lets the temperature move; LayerNorms stay frozen (they are part of
    the base checkpoint, and the merged export must be the base plus a low-rank delta).
    """

    def __init__(self, checkpoint: dict, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
                 targets: tuple[str, ...] = TARGETS, train_scale: bool = False):
        super().__init__()
        self.base_ckpt = checkpoint
        cfg = dict(checkpoint["cfg"])
        self.head_kw = dict(width=cfg["width"], depth=cfg["depth"],
                            proj=checkpoint.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
                            activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
                            residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", HIDDEN))
        self.state_head, self.action_head = make_head(**self.head_kw), make_head(**self.head_kw)
        self.state_head.load_state_dict(checkpoint["state_head"])
        self.action_head.load_state_dict(checkpoint["action_head"])
        for p in self.parameters():
            p.requires_grad_(False)
        for head in (self.state_head, self.action_head):
            for t in targets:
                _swap(head, t, rank, alpha, dropout)
        self.logit_scale = nn.Parameter(torch.as_tensor(checkpoint["logit_scale"]).float().clone(),
                                        requires_grad=train_scale)
        self.lora_cfg = {"rank": rank, "alpha": alpha, "dropout": dropout, "targets": list(targets),
                         "train_scale": train_scale}

    # ------------------------------------------------------------------ scoring
    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def encode_states(self, s: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_head(s.float()), dim=-1)

    def encode_actions(self, a: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.action_head(a.float()), dim=-1)

    def pair_logits(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """[B, d] x [B, d] -> [B] scaled cosine of matched rows."""
        return self.scale * (self.encode_states(s) * self.encode_actions(a)).sum(-1)

    def candidate_logits(self, s: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """[B, d] x [B, K, d] -> [B, K], the logits behind a System One Choice."""
        zs = self.encode_states(s)
        zc = self.encode_actions(cands.reshape(-1, cands.shape[-1])).view(*cands.shape[:2], -1)
        return self.scale * torch.einsum("bh,bkh->bk", zs, zc)

    @contextlib.contextmanager
    def reference(self):
        """Evaluate as the frozen base checkpoint (adapters off), for KL / DPO anchors."""
        layers = list(_lora_layers(self))
        saved = self.logit_scale.data.clone()
        for m in layers:
            m.enabled = False
        self.logit_scale.data.copy_(torch.as_tensor(self.base_ckpt["logit_scale"]).float())
        try:
            yield self
        finally:
            for m in layers:
                m.enabled = True
            self.logit_scale.data.copy_(saved)

    # ------------------------------------------------------------------ adapters
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        """Only the trained tensors (a few hundred KB at rank 16): the unit to version and roll back."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if "lora_" in k or (k == "logit_scale" and self.logit_scale.requires_grad)}

    def load_adapter(self, state: dict[str, torch.Tensor]) -> None:
        missing = set(state) - set(self.state_dict())
        if missing:
            raise KeyError(f"adapter keys not in this model: {sorted(missing)[:3]}")
        self.load_state_dict(state, strict=False)

    def reset_adapter(self) -> None:
        for m in _lora_layers(self):
            nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
            nn.init.zeros_(m.lora_B)
        self.logit_scale.data.copy_(torch.as_tensor(self.base_ckpt["logit_scale"]).float())

    # ------------------------------------------------------------------ export
    def _merged_head(self, head: nn.Module) -> dict[str, torch.Tensor]:
        plain = make_head(**self.head_kw)
        merged = copy.deepcopy(head)
        for name in TARGETS:
            child = getattr(merged, name)
            if isinstance(child, nn.ModuleList):
                for i, m in enumerate(child):
                    if isinstance(m, LoRALinear):
                        child[i] = m.merged()
            elif isinstance(child, LoRALinear):
                setattr(merged, name, child.merged())
        plain.load_state_dict(merged.state_dict())
        return {k: v.detach().cpu() for k, v in plain.state_dict().items()}

    def merged_checkpoint(self, extra: dict | None = None) -> dict:
        """A standard CLM checkpoint (``state_head``/``action_head``/``logit_scale``/``cfg``)."""
        cfg = dict(self.base_ckpt["cfg"])
        cfg["autoevolve"] = {**self.lora_cfg, **(extra or {})}
        return {"state_head": self._merged_head(self.state_head),
                "action_head": self._merged_head(self.action_head),
                "logit_scale": self.logit_scale.detach().cpu().clone(), "cfg": cfg}


def atomic_save(obj: dict, path: str) -> None:
    """Write next to ``path`` and rename over it, so a hot-reloading server never reads half a file."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def random_checkpoint(width: int = 256, depth: int = 3, proj: int = 64, hidden: int = 128,
                      layernorm: bool = True, seed: int = 0) -> dict:
    """A small checkpoint in the CLM format, for tests and the synthetic experiments."""
    torch.manual_seed(seed)
    kw = dict(width=width, depth=depth, proj=proj, activation="gelu", layernorm=layernorm,
              residual=False, hidden=hidden)
    sh, ah = make_head(**kw), make_head(**kw)
    return {"state_head": sh.state_dict(), "action_head": ah.state_dict(),
            "logit_scale": torch.tensor(math.log(1 / 0.07)),
            "cfg": {"width": width, "depth": depth, "projection_dim": proj, "hidden_size": hidden,
                    "activation": "gelu", "layernorm": layernorm, "residual": False}}
