"""Gemma3-27B and the injection hook.

The vectors and the hook follow Macar et al. 2026 (github.com/safety-research/introspection-mechanisms), which itself
follows Lindsey 2026 (arXiv:2601.01828).
"""
import contextlib
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stimuli import BASELINE_WORDS

MODEL = "google/gemma-3-27b-it"


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


class Model:
    def __init__(self, device: str = "cuda:0"):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
        self.layers = self.model.model.language_model.layers
        self.lm_head = self.model.get_output_embeddings()
        self.device = next(self.model.parameters()).device

    def chat(self, messages) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def user_prompt(self, text: str) -> str:
        """Chat prompt for a single user turn."""
        return self.chat([{"role": "user", "content": text}])

    def encode(self, prompts):
        # Chat-templated strings already contain BOS.
        return self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

    @torch.no_grad()
    def last_token_acts(self, prompts, layer_ids, batch_size=64) -> dict:
        """Residual stream (decoder-layer output) at the final prompt token, for each layer."""
        out = {l: [] for l in layer_ids}
        for i in range(0, len(prompts), batch_size):
            cache = {}
            hooks = [self.layers[l].register_forward_hook(capture(cache, l)) for l in layer_ids]
            try:
                self.model(**self.encode(prompts[i:i + batch_size]), use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            for l in layer_ids:
                out[l].append(cache[l][:, -1, :].float().cpu())
        return {l: torch.cat(v) for l, v in out.items()}

    def concept_vectors(self, concepts, layer_ids, template="Tell me about {word}"):
        """Returns ({layer: {concept: vec}}, {layer: mean residual norm of baseline prompts}).

        vec = act(concept) - mean(act(baseline)) at the last prompt token, unnormalized.
        """
        fmt = lambda w: self.user_prompt(template.format(word=w))
        c_acts = self.last_token_acts([fmt(w) for w in concepts], layer_ids)
        b_acts = self.last_token_acts([fmt(w) for w in BASELINE_WORDS], layer_ids)
        vecs = {l: {w: c_acts[l][i] - b_acts[l].mean(0) for i, w in enumerate(concepts)} for l in layer_ids}
        return vecs, {l: b_acts[l].norm(dim=-1).mean().item() for l in layer_ids}


def hidden(output):
    """The residual stream from a decoder layer's output (a tensor or a tuple starting with one)."""
    return output[0] if isinstance(output, tuple) else output


def capture(cache, l):
    """Forward hook that stores layer l's output in cache[l]."""
    def hook(module, inp, output):
        cache[l] = hidden(output).detach()
    return hook


@contextlib.contextmanager
def inject(model: Model, layer: int, vecs: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
    """Add vecs[b] to the residual stream after `layer` at padded positions [start[b], end[b]).

    vecs: [B, d] (already scaled by strength; zero rows = no injection). Single-token decoding steps (seq_len == 1)
    come after the injected span and are left alone.
    """
    vecs = vecs.to(model.device, torch.bfloat16)
    start, end = start.to(model.device), end.to(model.device)

    def hook(module, inp, output):
        h = hidden(output)
        B, T, _ = h.shape
        if T == 1:
            return output
        pos = torch.arange(T, device=h.device)[None, :]
        mask = (pos >= start[:, None]) & (pos < end[:, None])
        h = h + mask[..., None].to(h.dtype) * vecs[:, None, :]
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

