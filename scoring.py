"""The trial and how it is scored: one shared context with the vector added over one passage, then many final
questions, each scored from the context's KV cache by next-token probabilities or by teacher-forcing an answer.

  user:       PREAMBLE (setup)
  assistant:  Ok.
  user:       "Trial 1:\n\nPassage 1:\n<p>\n\nPassage 2:\n<p>\n\n<question>"   <- same passage twice; vector added
                                                                             over one copy's tokens

The vector reaches the question only through the KV cache (Pearson-Vogel et al. 2026), so every question sees exactly
the same injected activations.
"""
import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from model import Model, inject
from stimuli import TEST, TEST_EXTRA, surface_forms

PREAMBLE = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. In each trial I will show you the same passage twice. While you read one "
    "copy, I will inject a thought about a specific word into your mind. While you read the other copy, I will not "
    "inject anything."
)
# the same setup for the --different variant, where the two passages are not the same text
PREAMBLE_DIFFERENT = PREAMBLE.replace(
    "In each trial I will show you the same passage twice. While you read one copy, I will inject a thought about a "
    "specific word into your mind. While you read the other copy, I will not inject anything.",
    "In each trial I will show you two passages. While you read one of them, I will inject a thought about a "
    "specific word into your mind. While you read the other one, I will not inject anything.")
assert PREAMBLE_DIFFERENT != PREAMBLE
ANSWER_WORD = " Answer with just the word."
QUESTIONS = {
    "detect": "In which passage was a thought injected, {p1} or {p2}? Answer with just the {pkind}.",
    "identify": "Which word was the injected thought about?\n{options}\nAnswer with just the {okind}.",
    "say": "Before we continue, say a word." + ANSWER_WORD,
    "include": "Say the word that was injected." + ANSWER_WORD,
    "exclude": "Say any word except the one that was injected." + ANSWER_WORD,
}
WORD_QUESTIONS = ["say", "include", "exclude"]  # answered with a word: scored by teacher-forcing its surface forms
OTHER_QUESTIONS = ["include", "exclude"]  # also scored for words of concepts that were not injected
# --filler: uninjected text after the second copy, so an injection there is not right before the question
FILLER = ("That is the end of both copies of the passage. Nothing further was injected after this point. "
          "Please take a moment, and then answer the question below.")
# passage labels: (first, second, what to answer with). Gemma answers "1" with P > 0.99999 without injection.
PASSAGE_LABELS = {"12": ("1", "2", "number"), "AB": ("A", "B", "letter"), "XY": ("X", "Y", "letter")}
LETTERS = "ABCD"  # option slots in the output (pA..pD); shown to the model as Scorer.olabels
PASSAGES = TEST + TEST_EXTRA  # --passages N sweeps the first N; the 12 of TEST come first
PARTNERS = 30  # --different: the second passage of a trial is drawn from the first PARTNERS passages


def logit(p):
    p = np.clip(p, 1e-300, 1 - 1e-15)  # say-a-word baselines go down to ~1e-15
    return np.log(p) - np.log1p(-p)


def contexts(passages, different=False):
    """Each passage shown twice (a = b), x which copy is injected; the copies differ only by the injection.
    different=True: each passage is followed by another one instead, drawn at random from the first PARTNERS test
    passages (the draw depends only on the passage, so it stays the same when more passages are swept later)."""
    partner = lambda i: random.Random(f"partner|{i}").choice([j for j in range(PARTNERS) if j != i])
    return [dict(pair=i, a=i, b=partner(i) if different else i, inj=pos) for i in range(len(passages)) for pos in (1, 2)]


class Scorer:
    """Runs the shared context once (with injection) and scores many continuations from its KV cache."""

    def __init__(self, m: Model, passages, labels="12", filler=False, different=False):
        self.m, self.passages = m, passages
        self.preamble = PREAMBLE_DIFFERENT if different else PREAMBLE
        self.filler = f"\n\n{FILLER}" if filler else ""
        self.p1, self.p2, self.pkind = PASSAGE_LABELS[labels]
        # identification options are numbered when the passages are lettered, so the two never share a label
        self.olabels, self.okind = ("ABCD", "letter") if labels == "12" else ("1234", "number")
        self.softcap = getattr(m.model.config.get_text_config(), "final_logit_softcapping", None)
        self.base = m.model.model  # decoder stack returning normed last hidden state
        self._prefix, self._cont = {}, {}
        # tokens that would continue a word: the raw token starts with a letter or digit, i.e. no word-start
        # marker (SentencePiece "▁", byte-level BPE "Ġ") and no punctuation, newline or special token
        vocab = m.tok.convert_ids_to_tokens(list(range(len(m.tok))))
        self.cont_mask = torch.zeros(m.lm_head.weight.shape[0], dtype=torch.bool, device=m.device)
        self.cont_mask[:len(vocab)] = torch.tensor([bool(v) and v[0].isalnum() for v in vocab])
        self.digit_ids = [self._single(d) for d in (self.p1, self.p2)]
        self.letter_ids = [self._single(x) for x in self.olabels]

    def question(self, name, options=""):
        return QUESTIONS[name].format(p1=self.p1, p2=self.p2, pkind=self.pkind, options=options, okind=self.okind)

    def _single(self, s):
        ids = self.m.tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        return ids[0]

    def messages(self, a, b, question):
        """Setup, "Ok.", then one user turn: "Trial 1:", both passages, and the question."""
        trial = (f"Trial 1:\n\nPassage {self.p1}:\n{self.passages[a]}\n\nPassage {self.p2}:\n{self.passages[b]}"
                 f"{self.filler}\n\n{question}")
        return [{"role": "user", "content": self.preamble}, {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": trial}]

    def prefix(self, a, b):
        """Token ids of the shared context (through the end of passage 2) and the token spans of the two passages."""
        if (a, b) not in self._prefix:
            full = self.m.chat(self.messages(a, b, "QUESTION"))
            s = full[:full.rindex(self.passages[b]) + len(self.passages[b])]
            enc = self.m.tok(s, add_special_tokens=False, return_offsets_mapping=True)
            spans = []
            for text in (self.passages[a], self.passages[b]):
                lo = s.index(text, spans[-1][2] if spans else 0)
                hi = lo + len(text)
                toks = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < hi and y > lo]
                spans.append((toks[0], toks[-1] + 1, hi))
            self._prefix[(a, b)] = (s, enc["input_ids"], [sp[:2] for sp in spans])
        return self._prefix[(a, b)]

    def continuation(self, a, b, question, answer=""):
        """Ids of the final user turn + generation prompt (+ answer), tokenized as in the full prompt, and the number
        of ids before the answer."""
        key = (a, b, question, answer)
        if key not in self._cont:
            s, pre, _ = self.prefix(a, b)
            full = self.m.chat(self.messages(a, b, question))
            assert full.startswith(s), "chat template does not extend the prefix"
            q = self.m.tok(full[len(s):], add_special_tokens=False)["input_ids"]
            assert pre + q == self.m.tok(full, add_special_tokens=False)["input_ids"], "tokenization differs at the split"
            ans = self.m.tok(answer, add_special_tokens=False)["input_ids"] if answer else []
            self._cont[key] = (q + ans, len(q))
        return self._cont[key]

    def logits(self, h):
        # float32 unembedding: bf16 logits are rounded in steps of 0.125 at |logit| ~ 20, as large as the effects
        if not hasattr(self, "_W"):
            self._W = self.m.lm_head.weight.float()
        x = h.float() @ self._W.T
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x

    @torch.no_grad()
    def run(self, trials, layer, cont_batch=64):
        """trials: dicts with a, b, vec ([d], scaled; None = no injection), inj (1|2), conts [(ids, n_q), ...].

        Returns per trial a list, per continuation, of dict(lp_answer, lp_boundary, p_digits, p_letters):
          lp_answer    log P of the teacher-forced answer tokens (0 if there is no answer)
          lp_boundary  log P that the next token does not continue the answer's word
          p_digits     P of the two passage labels as the next token
          p_letters    P of the four option labels as the next token
        """
        m = self.m
        # 1. the shared contexts, left-padded into one batch, with the vector added over the injected passage
        pre = [self.prefix(t["a"], t["b"]) for t in trials]
        L = max(len(p[1]) for p in pre)
        B = len(trials)
        ids = torch.full((B, L), m.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        start, end = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
        for i, (t, (_, p, spans)) in enumerate(zip(trials, pre)):
            off = L - len(p)
            ids[i, off:] = torch.tensor(p)
            mask[i, off:] = 1
            s0, s1 = spans[t["inj"] - 1]
            start[i], end[i] = off + s0, off + s1
        ids, mask = ids.to(m.device), mask.to(m.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        d = m.lm_head.weight.shape[1]
        vecs = torch.stack([t["vec"] if t["vec"] is not None else torch.zeros(d) for t in trials])
        with inject(m, layer, vecs, start, end):
            out = self.base(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        n_real = mask.sum(-1)

        # 2. every continuation of every trial, in chunks, each on a copy of its own trial's rows of the cache
        flat = [(i, j, c) for i, t in enumerate(trials) for j, c in enumerate(t["conts"])]
        res = [[None] * len(t["conts"]) for t in trials]
        for k in range(0, len(flat), cont_batch):
            chunk = flat[k:k + cont_batch]
            rows = torch.tensor([i for i, _, _ in chunk], device=m.device)
            T = max(len(c[0]) for _, _, c in chunk)
            cid = torch.full((len(chunk), T), m.tok.pad_token_id, dtype=torch.long, device=m.device)
            cmask = torch.zeros((len(chunk), T), dtype=torch.long, device=m.device)
            for r, (_, _, (c, _)) in enumerate(chunk):
                cid[r, :len(c)] = torch.tensor(c)
                cmask[r, :len(c)] = 1
            sub = copy.copy(cache)
            sub.layers = [copy.copy(l) for l in cache.layers]
            sub.batch_select_indices(rows)
            full_mask = torch.cat([mask[rows], cmask], 1)
            cpos = n_real[rows][:, None] + torch.arange(T, device=m.device)[None]
            h = self.base(input_ids=cid, attention_mask=full_mask, position_ids=cpos, past_key_values=sub,
                          use_cache=True).last_hidden_state
            # positions to read: from the last question token through the last answer token
            for r, (i, j, (c, n_q)) in enumerate(chunk):
                lp = F.log_softmax(self.logits(h[r, n_q - 1:len(c)]), -1)
                ans = torch.tensor(c[n_q:], device=m.device)
                last = lp[-1]
                res[i][j] = dict(
                    lp_answer=lp[torch.arange(len(ans)), ans].sum().item() if len(ans) else 0.0,
                    lp_boundary=torch.log1p(-last[self.cont_mask].exp().sum().clamp(max=1 - 1e-7)).item(),
                    p_digits=last[self.digit_ids].exp().tolist(),
                    p_letters=last[self.letter_ids].exp().tolist(),
                )
            del sub, h
        del cache, out
        return res


def word_conts(sc, a, b, question, concept):
    """One continuation per surface form of the concept's word, as the answer to `question`."""
    return [sc.continuation(a, b, question, f) for f in surface_forms(concept)]


def word_prob(results):
    """Sum over surface forms of P(form, then a word boundary)."""
    return float(sum(math.exp(r["lp_answer"] + r["lp_boundary"]) for r in results))
