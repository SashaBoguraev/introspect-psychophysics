"""The vectors that get injected, as {layer: {concept: tensor}}.

  concept  mean-diff concept vectors (Lindsey): residual at the last token of "Tell me about {word}" minus the
           mean over the baseline words; unnormalized
  random   control: random directions with the concept vectors' per-dimension spread, each rescaled to the norm of
           the concept vector it stands in for, so only the direction differs
"""
import torch

from stimuli import CONCEPTS

KINDS = ["concept", "random"]
LAYERS = [16, 31, 40, 53]  # default injection layers (of Gemma3-27B's 62)


def random_like(vecs, seed):
    """Random directions with the concept vectors' per-dimension std, norm-matched to each concept vector."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l, vl in vecs.items():
        std = torch.stack(list(vl.values())).std(0)
        out[l] = {}
        for c, v in vl.items():
            r = torch.randn(v.shape, generator=g) * std
            out[l][c] = r * (v.norm() / r.norm())
    return out


def build(m, kinds, layers, seed=0):
    """({kind: {layer: {concept: vec}}}, {layer: mean residual norm}) for the 50 concepts."""
    mean_diff, resid_norm = m.concept_vectors(CONCEPTS, layers)
    of_kind = {"concept": lambda: mean_diff, "random": lambda: random_like(mean_diff, seed)}
    return {k: of_kind[k]() for k in kinds}, resid_norm
