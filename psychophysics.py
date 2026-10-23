"""Psychophysics of concept injection in Gemma3-27B: priming-calibrated strength, 2IFC detection, 4AFC identification,
and include/exclude process dissociation (Jacoby 1991; Debner & Jacoby 1994). Token probabilities only: no sampled
answers and no classifier. The trial is laid out in scoring.py; only its final question changes:

  detect    "In which passage was a thought injected, 1 or 2?"      P(correct) / (P(1) + P(2)); chance 50%
  identify  "Which word was the injected thought about? A) B) C) D)" 4 shuffled concepts; chance 25%
  say       "Before we continue, say a word."                        priming probe; P(concept word)
  include   "Say the word that was injected."                        Jacoby: I = C + A(1 - C)
  exclude   "Say any word except the one that was injected."         Jacoby: E = A(1 - C)
            control C = I - E; automatic A = E / (1 - C)

P(word) sums teacher-forced probabilities over surface forms (bread/Bread, trumpet/trumpets/...) and requires the
word to end there (the next token does not continue it). Priming is a change in log-odds against the same context
with no injection.

Other words: a strong vector of any kind makes rare words more likely, whatever the instruction. To separate that
from an effect on the injected concept, include and exclude are also scored for --other-words other concepts' words
(drawn per concept and passage from the other 49, never injected in that trial) in the same injected context.

  python psychophysics.py calibrate   # GPU: for each (layer, concept), sweep alpha on a log grid over CALIBRATION
                                      #   passages and find alpha*, the strength at which the say-a-word probe's
                                      #   log-odds rise by --target (default ln 2) → alpha_<vectors>.csv
  python psychophysics.py sweep       # GPU: on held-out TEST passages, strength = k * alpha* for each k in
                                      #   --strengths, all five questions, for every label scheme in --labels
                                      #   → sweep_<vectors>_<labels>[_filler|_different].csv. Random vectors have no
                                      #   concept, so they use the paired concept vector's alpha* and are matched on size.
  python psychophysics.py summarize   # CPU: the raw CSVs above → the small tables the figures are drawn from
                                      #   (summary.csv, different/summary.csv, calibration_curve_<vectors>.csv); see analysis.py
  python psychophysics.py plot        # CPU: every figure, from those tables and alpha_<vectors>.csv only; see figures.py
  python psychophysics.py selftest    # GPU: cached-continuation scores must match a plain full-sequence forward pass

Outputs go to results/psychophysics/. Layers and strengths already on disk are skipped unless --overwrite is given, so
an interrupted run can simply be started again, and sweeps on several GPUs (CUDA_VISIBLE_DEVICES, --layers) can
share the output files.
"""
import argparse
import contextlib
import fcntl
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import vectors
from analysis import MIN_CALIBRATED, OUT, TARGET, alpha_path, summarize_all
from model import Model, inject, seed_everything
from scoring import (LETTERS, OTHER_QUESTIONS, PASSAGE_LABELS, PASSAGES, QUESTIONS, WORD_QUESTIONS, Scorer, contexts,
                     logit, word_conts, word_prob)
from stimuli import CALIBRATION, CONCEPTS, TEST, TEST_EXTRA, names_concept, surface_forms

# ---------------------------------------------------------------- results files
@contextlib.contextmanager
def locked(path):
    """Exclusive use of a results file while it is read and rewritten (sweeps on several GPUs share the files)."""
    with open(f"{path}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def layers_done(path):
    """The layers a results file already covers."""
    return set(pd.read_csv(path, usecols=["layer"]).layer) if path.exists() else set()


def merge_layers(path, new, by=("layer",)):
    """Write `new` to the CSV at `path`, replacing its rows for the layers in `new` and keeping its other layers, so
    layers can be added later or run in parallel. by=("layer", "k") replaces only the strengths in `new`, so a
    layer's strength grid can be filled in later."""
    by = list(by)
    with locked(path):
        if path.exists():
            old = pd.read_csv(path, dtype={"labels": str}, float_precision="round_trip")  # numbers kept exactly
            redone = pd.MultiIndex.from_frame(old[by]).isin(pd.MultiIndex.from_frame(new[by].drop_duplicates()))
            new = pd.concat([old[~redone], new])
        tmp = path.with_suffix(".tmp")  # written beside it and moved into place, so a reader never sees half a file
        new.sort_values("layer", kind="stable").to_csv(tmp, index=False)
        os.replace(tmp, path)


def sweep_path(kind, labels, filler, different=False):
    return OUT / f"sweep_{kind}_{labels}{'_filler' if filler else ''}{'_different' if different else ''}.csv"


# ---------------------------------------------------------------- calibrate
def calibrate(m, args, kind, vecs, resid, layers):
    """alpha* per (layer, concept): the strength at which the say-a-word probe's log-odds rise by --target."""
    sc = Scorer(m, CALIBRATION)
    ctx = contexts(CALIBRATION)
    alphas = [0.0] + list(2.0 ** np.arange(args.alpha_min, args.alpha_max + 1e-9, 0.5))
    for layer in layers:
        rows = []
        jobs = [(c, a, x) for c in CONCEPTS for a in alphas for x in ctx if a > 0 or x["inj"] == 1]
        for i in tqdm(range(0, len(jobs), args.batch_size), desc=f"calibrate L{layer}"):
            batch = jobs[i:i + args.batch_size]
            trials = [dict(a=x["a"], b=x["b"], inj=x["inj"], vec=vecs[layer][c] * a if a else None,
                           conts=word_conts(sc, x["a"], x["b"], sc.question("say"), c)) for c, a, x in batch]
            for (c, a, x), r in zip(batch, sc.run(trials, layer)):
                rows.append(dict(layer=layer, concept=c, alpha=a, pair=x["pair"], inj=x["inj"], p_say=word_prob(r)))
        df = pd.DataFrame(rows)
        df["vec_norm"] = [vecs[l][c].norm().item() for l, c in zip(df.layer, df.concept)]
        df["resid_norm"] = df.layer.map(resid)
        df, alpha = fit_alpha(df, args.target)
        merge_layers(OUT / f"calibration_{kind}.csv", df)
        merge_layers(alpha_path(kind), alpha)
        print(f"wrote layer {layer} to {alpha_path(kind)}")


def fit_alpha(df, target):
    """(calibration trials with their log-odds rise, alpha* per (layer, concept)). alpha* is the first crossing of the
    mean log-odds rise over `target`, interpolated in log alpha."""
    df = df.copy()
    df["logodds"] = logit(df.p_say)
    base = df[df.alpha == 0].groupby(["layer", "concept", "pair"]).logodds.mean().rename("base")
    df = df.join(base, on=["layer", "concept", "pair"])
    df["delta"] = df.logodds - df.base
    curve = df[df.alpha > 0].groupby(["layer", "concept", "alpha"]).delta.mean().reset_index()
    rec = []
    for (l, c), g in curve.groupby(["layer", "concept"]):
        g = g.sort_values("alpha")
        above = np.nonzero(g.delta.values >= target)[0]
        a_star = np.nan
        if len(above):
            k = above[0]
            if k == 0:
                a_star = g.alpha.values[0]
            else:  # interpolate in log alpha
                a0, a1 = np.log(g.alpha.values[k - 1]), np.log(g.alpha.values[k])
                d0, d1 = g.delta.values[k - 1], g.delta.values[k]
                a_star = float(np.exp(a0 + (target - d0) / (d1 - d0) * (a1 - a0)))
        first = df[(df.layer == l) & (df.concept == c)].iloc[0]
        rec.append(dict(layer=l, concept=c, alpha_star=a_star, vec_norm=first.vec_norm, resid_norm=first.resid_norm,
                        base_p=1 / (1 + np.exp(-base.loc[l, c].mean())), max_delta=g.delta.max()))
    a = pd.DataFrame(rec)
    a["inj_norm_ratio"] = a.alpha_star * a.vec_norm / a.resid_norm
    print(a.groupby("layer")[["alpha_star", "inj_norm_ratio", "base_p", "max_delta"]]
          .describe(percentiles=[0.5]).T.to_string())
    print("uncalibrated (never reached target):", a[a.alpha_star.isna()][["layer", "concept"]].values.tolist())
    return df, a


# ---------------------------------------------------------------- sweep
def sweep(m, args, kind, labels, vecs, layers):
    """Every question at strength k * alpha*, for each k in --strengths, at each of `layers`. Strengths a layer
    already has in the output file are skipped (unless --overwrite), so a coarse grid can be run first and the
    rest filled in later."""
    path = sweep_path(kind, labels, args.filler, args.different)
    have = set() if args.overwrite or not path.exists() else set(
        pd.read_csv(path, usecols=["layer", "k", "pair"]).drop_duplicates().itertuples(index=False, name=None))
    cal = pd.read_csv(alpha_path("concept" if kind == "random" else kind))
    a_star = {(r.layer, r.concept): r.alpha_star for r in cal.itertuples()}
    sc = Scorer(m, PASSAGES, labels, args.filler, args.different)
    ctx = contexts(PASSAGES[:args.passages], args.different)
    rows = []
    for layer in layers:
        n_cal = int(cal[cal.layer == layer].alpha_star.notna().sum())
        if n_cal < MIN_CALIBRATED:  # the layer cannot be calibrated: leave it out
            print(f"layer {layer}: only {n_cal} of {len(CONCEPTS)} concepts calibrated (< {MIN_CALIBRATED}); not swept")
            continue
        todo = [(k, x) for k in args.strengths for x in ctx if (layer, k, x["pair"]) not in have]
        if len(todo) < len(args.strengths) * len(ctx):
            print(f"{path}: layer {layer} already has {len(args.strengths) * len(ctx) - len(todo)} of "
                  f"{len(args.strengths) * len(ctx)} strength × context cells (--overwrite to redo)")
        jobs = [(c, k, x) for c in args.concepts if not np.isnan(a_star.get((layer, c), np.nan)) for k, x in todo
                if not (k == 0 and x["inj"] == 2)]  # no vector: identical to inj=1, scored once and used for both
        for i in tqdm(range(0, len(jobs), args.batch_size), desc=f"sweep L{layer}"):
            batch = jobs[i:i + args.batch_size]
            trials, meta = [], []
            for c, k, x in batch:
                conts, mt = sweep_trial(sc, args, c, x)
                alpha = k * a_star[(layer, c)]
                trials.append(dict(a=x["a"], b=x["b"], inj=x["inj"], vec=vecs[layer][c] * alpha if k else None,
                                   conts=conts))
                meta.append(dict(layer=layer, concept=c, k=k, alpha=alpha, pair=x["pair"], second=x["b"],
                                 inj=x["inj"], **mt))
            for mt, r in zip(meta, sc.run(trials, layer)):
                row = dict(mt, vectors=kind, labels=labels, filler=int(args.filler), **read_sweep(r, mt))
                rows.append(row)
                if mt["k"] == 0:  # the no-injection context stands for both injection positions
                    rows.append(dict(row, inj=2))
        if rows:
            merge_layers(path, pd.DataFrame(rows), by=("layer", "k", "pair"))  # layer by layer: kept if interrupted
    print(f"wrote {path}: {len(rows)} rows for layers {layers}")


def sweep_trial(sc, args, c, x):
    """The continuations scored for concept c in context x, in this order: detect, identify, each word question for
    each surface form of c, then include and exclude for each surface form of each other concept's word. Also the
    trial's metadata (options, other words). The options and other words depend only on the concept and passage,
    so they are the same at both injected positions and every k."""
    rng = random.Random(f"{c}|{x['pair']}")
    opts = [c] + rng.sample([o for o in CONCEPTS if o != c], 3)
    rng.shuffle(opts)
    options = "\n".join(f"{sc.olabels[i]}) {o}" for i, o in enumerate(opts))
    qs = {q: sc.question(q, options) for q in QUESTIONS}
    a, b = x["a"], x["b"]
    conts = [sc.continuation(a, b, qs["detect"]), sc.continuation(a, b, qs["identify"])]
    for q in WORD_QUESTIONS:
        conts += word_conts(sc, a, b, qs[q], c)
    others = random.Random(f"other|{c}|{x['pair']}").sample([o for o in CONCEPTS if o != c], args.other_words)
    for q in OTHER_QUESTIONS:
        for o in others:
            conts += word_conts(sc, a, b, qs[q], o)
    return conts, dict(correct_letter=LETTERS[opts.index(c)], options="|".join(opts), others="|".join(others))


def read_sweep(r, mt):
    """One sweep row's scores from Scorer.run's results for the continuations of sweep_trial."""
    p1, p2 = r[0]["p_digits"]
    out = dict(p1=p1, p2=p2, **{f"p{L}": p for L, p in zip(LETTERS, r[1]["p_letters"])})
    nf = len(surface_forms(mt["concept"]))
    for qi, q in enumerate(WORD_QUESTIONS):
        out[f"p_{q}"] = word_prob(r[2 + qi * nf: 2 + (qi + 1) * nf])
    at = 2 + len(WORD_QUESTIONS) * nf
    for q in OTHER_QUESTIONS:  # mean log-odds over the other words
        lo = []
        for o in filter(None, mt["others"].split("|")):
            n = len(surface_forms(o))
            lo.append(logit(word_prob(r[at:at + n])))
            at += n
        out[f"lo_{q}_other"] = float(np.mean(lo)) if lo else np.nan
    return out


# ---------------------------------------------------------------- self-test
def selftest(m, args, vecs):
    """Cached-continuation scores must match a plain full-sequence forward with the same injection."""
    sc = Scorer(m, TEST)
    layer = args.layers[0]
    c = "Bread"
    trials = []
    for a, b, inj in [(0, 1, 1), (4, 5, 2), (10, 11, 1)]:
        conts = [sc.continuation(a, b, sc.question("detect"))] + word_conts(sc, a, b, sc.question("say"), c)
        trials.append(dict(a=a, b=b, inj=inj, vec=vecs[layer][c] * 4.0, conts=conts))
    fast = sc.run(trials, layer, cont_batch=3)
    for t, fr in zip(trials, fast):
        s, pre, spans = sc.prefix(t["a"], t["b"])
        for (ids, n_q), f in zip(t["conts"], fr):
            full = torch.tensor([pre + ids], device=m.device)
            s0, s1 = spans[t["inj"] - 1]
            with inject(m, layer, t["vec"][None], torch.tensor([s0]), torch.tensor([s1])), torch.no_grad():
                lg = m.model(input_ids=full).logits[0].float()
            lp = F.log_softmax(lg, -1)
            n0 = len(pre) + n_q - 1
            slow = sum(lp[n0 + i, a].item() for i, a in enumerate(ids[n_q:]))
            print(f"{m.tok.decode(ids[n_q:]) or m.tok.decode(ids[-3:])!r:14} cached={f['lp_answer']:.4f} full={slow:.4f} "
                  f"P(1,2) cached={np.round(f['p_digits'], 4)} full={np.round(lp[n0, sc.digit_ids].exp().tolist(), 4)}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calibrate", "sweep", "summarize", "plot", "selftest"])
    ap.add_argument("--vectors", nargs="+", choices=vectors.KINDS, default=vectors.KINDS)
    ap.add_argument("--layers", type=int, nargs="+", default=vectors.LAYERS)
    ap.add_argument("--labels", nargs="+", choices=list(PASSAGE_LABELS), default=list(PASSAGE_LABELS),
                    help="sweep: passage label schemes")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4, 8, 16, 32],
                    help="sweep: multiples k of each concept's alpha* (not raw multipliers)")
    ap.add_argument("--filler", action="store_true",
                    help="sweep: put a neutral sentence between passage 2 and the question, so neither copy is adjacent to it")
    ap.add_argument("--different", action="store_true",
                    help="sweep: show two different passages (a random partner for each) instead of one passage twice "
                         "→ sweep_<vectors>_<labels>_different.csv")
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS, help="sweep: only these concepts (piloting)")
    ap.add_argument("--other-words", type=int, default=4,
                    help="sweep: other concepts whose words are also scored under include and exclude (0: none)")
    ap.add_argument("--target", type=float, default=TARGET, help="calibrate: say-probe log-odds rise")
    ap.add_argument("--alpha-min", type=float, default=-4, help="calibrate: log2 of the smallest strength")
    ap.add_argument("--alpha-max", type=float, default=4, help="calibrate: log2 of the largest strength")
    ap.add_argument("--passages", type=int, default=len(TEST),
                    help=f"sweep: how many test passages (the original {len(TEST)}, up to {len(TEST) + len(TEST_EXTRA)}); "
                         "passages a layer and strength already have are skipped")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="redo outputs that are already on disk")
    args = ap.parse_args()
    for p in CALIBRATION + PASSAGES:
        hits = [c for c in CONCEPTS if names_concept(p, c)]
        assert not hits, f"passage names {hits}: {p[:60]}"
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if args.stage == "summarize":
        summarize_all()
    elif args.stage == "plot":
        import figures  # plotnine is needed only here
        figures.plot()
    elif args.stage == "selftest":
        m = Model(args.device)
        selftest(m, args, vectors.build(m, ["concept"], args.layers[:1], args.seed)[0]["concept"])
    elif args.stage == "calibrate":
        # random vectors have no concept to calibrate on: they take the concept vectors' alpha*
        todo = []
        for kind in args.vectors:
            if kind == "random":
                continue
            path = alpha_path(kind)
            layers = [l for l in args.layers if args.overwrite or l not in layers_done(path)]
            if len(layers) < len(args.layers):
                print(f"{path}: layers {sorted(set(args.layers) - set(layers))} already there (--overwrite to redo)")
            todo.append((kind, layers))
        if any(layers for _, layers in todo):
            m = Model(args.device)
            vecs, resid = vectors.build(m, [k for k, _ in todo], args.layers, args.seed)
            for kind, layers in todo:
                calibrate(m, args, kind, vecs[kind], resid, layers)
    else:  # sweep: sweep() skips the strengths and passages already on disk
        m = Model(args.device)
        vecs, _ = vectors.build(m, args.vectors, args.layers, args.seed)
        for kind in args.vectors:
            for labels in args.labels:
                sweep(m, args, kind, labels, vecs[kind], args.layers)


if __name__ == "__main__":
    main()
