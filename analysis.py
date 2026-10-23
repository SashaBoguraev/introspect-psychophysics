"""From the raw per-trial CSVs of psychophysics.py to the small tables every figure is drawn from:

  summary.csv                    mean and 95% bootstrap CI over concepts, per vectors × filler × labels × layer × k ×
                                 measure (all same-passage sweeps)
  different/summary.csv          the same for the --different sweeps beside the same-passage sweeps restricted to
                                 what those have, with a `passages` column
  calibration_curve_<kind>.csv   mean rise in say-a-word log-odds per (layer, concept, alpha) on the calibration passages
  per_concept.csv                the per-concept values summary.csv is computed from (not committed: 5 MB)

Measures (per concept over the 24 held-out contexts, then mean and 95% bootstrap CI over concepts):
  priming      rise in the log-odds of saying the concept, vs the same context without injection
  detect       2IFC percent correct, averaged over both injected positions. Gemma often prefers one label whatever
               is injected, which pulls this toward 50%.
  detect_pair  bias-free 2IFC: share of passages where log P(2)/P(1) is higher when passage 2 is injected than when
               passage 1 is; detect_shift is the same comparison in nats
  identify     4AFC percent correct among the four options. Its CI is over each concept's change from its own
               no-injection level (added to the mean no-injection level), since some concepts' words are favoured as
               answers whatever is injected
  include/exclude  rise in log-odds of the concept under "say the injected word" / "say any word except it"
  include_other/exclude_other  the same rise for words of concepts that were not injected (mean over those words)
  include_specific/exclude_specific  the concept's rise minus the other words' rise
  control, automatic  Jacoby's C (percentage points) and A (percent)
  detect_mass, first_label, alpha   the two labels' total probability, the share of it on the first label, and the
               raw strength (diagnostics)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import OTHER_QUESTIONS, logit

OUT = Path(__file__).parent / "results" / "psychophysics"
TARGET = math.log(2)  # calibration: the rise in say-a-word log-odds that defines alpha*
MIN_CALIBRATED = 25  # a layer is swept only if at least this many of the 50 concepts reach the target there
MEASURES = {  # measure: chance level (None: no chance level)
    "priming": 0.0, "detect": 50.0, "detect_pair": 50.0, "identify": 25.0, "include": 0.0, "exclude": 0.0,
    "control": 0.0, "automatic": None,
}
OTHER_MEASURES = [f"{q}_{w}" for w in ("other", "specific") for q in OTHER_QUESTIONS]
KEYS = ["vectors", "filler", "labels", "layer"]
PASSAGE_KINDS = ["Same Passage Twice", "Two Different Passages"]  # the `passages` column of different/summary.csv


def alpha_path(kind):
    return OUT / f"alpha_{kind}.csv"


def calibration_curve_path(kind):
    return OUT / f"calibration_curve_{kind}.csv"


def boot(x, n=2000, seed=0):
    """(mean, lower, upper): mean of the non-NaN values with a 95% bootstrap interval."""
    x = np.asarray(x[~np.isnan(x)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(r, [2.5, 97.5])


def load_sweeps():
    """Every sweep on disk (all vector kinds and label schemes, with and without filler). The --different sweeps
    are a separate experiment and are left out."""
    return pd.concat([pd.read_csv(p, dtype={"labels": str}) for p in sorted(OUT.glob("sweep_*.csv"))
                      if not p.stem.endswith("_different")])


def per_concept(df):
    """Every measure per concept and strength (averaged over passages and injected positions)."""
    df = df.reset_index(drop=True)
    df["filler"] = df["filler"].fillna(0).astype(int)  # the first sweeps predate --filler and have no such column
    base = df[df.k == 0].groupby(KEYS + ["concept", "pair"])[["p_say", "p_include", "p_exclude"]].mean()
    base = logit(base).add_prefix("base_lo_")
    df = df.join(base, on=KEYS + ["concept", "pair"])
    df["priming"] = logit(df.p_say) - df.base_lo_p_say
    two = df.p1 + df.p2
    df["detect"] = 100 * np.where(df.inj == 1, df.p1, df.p2) / two
    df["detect_mass"] = two
    df["first_label"] = 100 * df.p1 / two  # the label preference: share of the two labels' probability on the first
    letters = df[["pA", "pB", "pC", "pD"]].to_numpy()
    idx = df.correct_letter.map("ABCD".index).to_numpy()
    df["identify"] = 100 * letters[np.arange(len(df)), idx] / letters.sum(1)

    df["include"] = logit(df.p_include) - df.base_lo_p_include
    df["exclude"] = logit(df.p_exclude) - df.base_lo_p_exclude
    for q in OTHER_QUESTIONS:  # empty for sweeps run without other words
        lo = df[f"lo_{q}_other"] if f"lo_{q}_other" in df else pd.Series(np.nan, index=df.index)
        base_other = lo[df.k == 0].groupby([df[c] for c in KEYS + ["concept", "pair"]]).mean().rename("base_other")
        df[f"{q}_other"] = lo - df.join(base_other, on=KEYS + ["concept", "pair"]).base_other
        df[f"{q}_specific"] = df[q] - df[f"{q}_other"]
    lr = np.log(df.p2) - np.log(df.p1)  # log P(2)/P(1), from raw probabilities (P(1) ≈ 1)
    # bias-free pairing: the same passage, first copy injected vs second copy injected
    pr = df.assign(lr=lr).pivot_table(index=KEYS + ["concept", "k", "pair"], columns="inj", values="lr")
    pr["detect_shift"] = pr[2] - pr[1]
    pr["detect_pair"] = np.where(pr[1] == pr[2], np.nan, 100.0 * (pr[2] > pr[1]))
    pr = pr.reset_index()

    g = df.groupby(KEYS + ["concept", "k"])
    means = ["priming", "detect", "identify", "include", "exclude", "detect_mass", "first_label", "alpha", *OTHER_MEASURES]
    c = g[means].mean()
    I, E = g.p_include.mean(), g.p_exclude.mean()
    c["control"] = 100 * (I - E)
    c["automatic"] = 100 * E / (1 - (I - E))
    c["base_p_say"] = 1 / (1 + np.exp(-g.base_lo_p_say.mean()))
    # identification with each concept's own no-injection level swapped for the mean one: the same mean, but the CI
    # leaves out how much the model likes each concept's word as an answer (2-76% at k = 0), which is fixed across k
    base = c[c.index.get_level_values("k") == 0].identify.droplevel("k").rename("base_identify")
    c = c.join(base, on=KEYS + ["concept"])
    c["identify_paired"] = c.identify - c.base_identify + c.groupby(KEYS).base_identify.transform("mean")
    c = c.drop(columns="base_identify")
    c = c.join(pr.groupby(KEYS + ["concept", "k"])[["detect_pair", "detect_shift"]].mean())
    return c.reset_index()


def summarize(pc):
    """Mean and 95% bootstrap CI over concepts of every measure, in long format."""
    rows = []
    for (v, f, lab, l, k), g in pc.groupby(KEYS + ["k"]):
        for mkey in list(MEASURES) + ["detect_shift", "detect_mass", "first_label", "alpha"] + OTHER_MEASURES:
            if mkey in OTHER_MEASURES and g[mkey].isna().all():
                continue  # other words were not scored in this sweep
            m, lo, hi = boot(g["identify_paired" if mkey == "identify" else mkey].to_numpy())
            rows.append(dict(vectors=v, filler=f, labels=lab, layer=l, k=k, measure=mkey, mean=m, lo=lo, hi=hi,
                             n=g[mkey].notna().sum()))
    return pd.DataFrame(rows)


def onsets(s):
    """Smallest k > 0 at which the 95% CI excludes chance (and every larger k in the sweep does too)."""
    rows = []
    for (v, f, lab, l, mk), g in s[(s.k > 0) & s.measure.isin(list(MEASURES))].groupby(KEYS + ["measure"]):
        ch = MEASURES[mk]
        if ch is None:
            continue
        g = g.sort_values("k")
        above = (g.lo > ch).to_numpy()
        onset = next((kk for i, kk in enumerate(g.k) if above[i:].all()), np.nan)
        rows.append(dict(vectors=v, filler=f, labels=lab, layer=l, measure=mk, onset_k=onset))
    return pd.DataFrame(rows).pivot_table(index=KEYS, columns="measure", values="onset_k")


def load_different():
    """The --different sweeps (two different passages per trial) beside the same-passage sweeps restricted to what
    the --different sweeps have: the same 12 first passages and the same strengths, labels 1 / 2, no filler →
    summary of both with a `passages` column."""
    out = []
    for path in sorted(OUT.glob("sweep_*_12_different.csv")):
        diff = pd.read_csv(path, dtype={"labels": str})
        same = pd.read_csv(path.with_name(path.name.replace("_different", "")), dtype={"labels": str})
        same = same[(same.filler.fillna(0) == 0) & same.pair.isin(diff.pair.unique()) & same.k.isin(diff.k.unique())
                    & same.layer.isin(diff.layer.unique())]
        for name, df in zip(PASSAGE_KINDS, (same, diff)):
            out.append(summarize(per_concept(df)).assign(passages=name))
    return pd.concat(out) if out else pd.DataFrame()


def calibration_curve(kind):
    """The mean rise in say-a-word log-odds per (layer, concept, alpha > 0), over calibration passages and positions."""
    cal = pd.read_csv(OUT / f"calibration_{kind}.csv")
    return cal[cal.alpha > 0].groupby(["layer", "concept", "alpha"]).delta.mean().reset_index()


def summarize_all():
    for path in sorted(OUT.glob("calibration_*.csv")):
        kind = path.stem.removeprefix("calibration_")
        if kind.startswith("curve_"):
            continue
        calibration_curve(kind).to_csv(calibration_curve_path(kind), index=False)
        print(f"wrote {calibration_curve_path(kind)}")
    pc = per_concept(load_sweeps())
    pc.to_csv(OUT / "per_concept.csv", index=False)
    s = summarize(pc)
    s.to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT}/per_concept.csv, summary.csv\n\nonset k (first k from which the CI stays above chance):")
    print(onsets(s).to_string())
    d = load_different()
    if len(d):
        (OUT / "different").mkdir(exist_ok=True)
        d.to_csv(OUT / "different" / "summary.csv", index=False)
        print(f"wrote {OUT}/different/summary.csv")
