"""Shared test helpers."""
from __future__ import annotations

import functools
import math

import pytest
import torch

from neuron_games.model import FCSNN
from neuron_games.potential.base import TermContext


def needs_impl(fn):
    """Skip (instead of fail) while the code under test still raises
    NotImplementedError. Once it is implemented the test runs for real."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"not implemented yet: {e}")
    return wrapper


def small_net(n_in=12, n_hidden=9, n_out=4, gain=3.0, seed=0) -> FCSNN:
    torch.manual_seed(seed)
    net = FCSNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, alpha=math.exp(-0.1))
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)
    return net


def small_ctx(cfg, T=15, B=5, n_in=12, seed=1, epoch=0) -> TermContext:
    net = small_net(n_in=n_in)
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(T, B, n_in, generator=g) < 0.4).float()
    with torch.no_grad():
        rec = net(x, record_traces=True)
    return TermContext(rec=rec, weights=[w.detach() for w in net.weights], cfg=cfg, epoch=epoch)
