"""Every figure, drawn only from the small tables analysis.py writes (summary.csv, different/summary.csv,
calibration_curve_<kind>.csv) and alpha_<kind>.csv, so they can be redrawn without the raw sweeps. All are PDFs sized
for a 36 × 48 in poster; mean over concepts with a 95% bootstrap CI throughout.

  calibration/      dose_response_<kind>.pdf            each concept's dose-response, one panel per layer, with its α*
                    dose_response_layer16_<kind>.pdf    the same for layer 16 alone
                    dose_response_by_layer_<kind>.pdf   priming on the held-out passages against k, one line per layer
                    by_layer_alpha_<kind>.pdf           α* across layers (raw, and relative to the residual norm)
                    by_layer_response_<kind>.pdf        how many concepts calibrate, and the largest rise on the grid
  detection/        curves_12.pdf, by_layer_best.pdf
  identification/   curves_12.pdf, by_layer_best.pdf
  include_exclude/  curves_<kind>_12[_specific].pdf, by_layer_best.pdf, control_automatic_best.pdf
  different/        detection[_bias_free].pdf, identification.pdf, include_exclude_<kind>.pdf: the same passage twice
                    vs two different passages

curves_*: one panel per layer against strength k (multiples of each concept's α*), passages labelled 1 / 2, no
filler. by_layer_best: each evaluation across layers at its best k (see by_layer).

  python figures.py   (or python psychophysics.py plot)
"""
import re
import textwrap

import numpy as np
import pandas as pd
from matplotlib import colormaps, font_manager
from matplotlib.colors import to_hex
from plotnine import (
    aes, annotate, coord_cartesian, element_blank, element_rect, element_text, facet_wrap, geom_blank,
    geom_errorbar, geom_hline, geom_line, geom_point, geom_ribbon, geom_text, geom_vline, ggplot, guide_legend,
    guides, labs, position_dodge, scale_color_manual, scale_fill_manual, scale_linetype_manual, scale_x_continuous,
    scale_y_continuous, scale_y_log10, theme, theme_minimal,
)

from analysis import OUT, PASSAGE_KINDS, TARGET, alpha_path, boot, calibration_curve_path
from stimuli import CONCEPTS


# ---------------------------------------------------------------- look
def palatino():
    """Palatino if installed, else a metric-compatible clone."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Palatino", "Palatino Linotype", "TeX Gyre Pagella", "P052"):
        if name in have:
            return name
    return "serif"


FONT = palatino()
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
STRIP, BORDER = "#e3e2de", "#8a8984"  # grey facet headers, panel boxes
BG, STRIP_BG = "#F3EADB", "#E2D6C1"  # the cream of setup_figure.py, and its chat bubbles for facet headers
METHOD_NAMES = {"concept": "mean-diff concept vectors", "random": "random vectors"}
METHOD_COLORS = {"mean-diff concept vectors": "#2a78d6", "random vectors": "#eb6834"}
QUESTION_NAMES = {"include": "“say the injected word”", "exclude": "“say any word except it”"}
# green and purple: blue and orange already mean concept and random vectors
QUESTION_COLORS = {"“say the injected word”": "#1baf7a", "“say any word except it”": "#7b4fc4"}
# the four strength-curve figures (calibration, detection, identification, include/exclude) tile 2 × 2 in the bottom
# half of a 36 × 48 in portrait poster: each is drawn at its printed size, with text scaled up to match
POSTER_GRID = dict(width=21.5, height=7.65, scale=1.5)  # no titles: the poster labels them
GRID_THEME = dict(legend_position="bottom", legend_direction="horizontal", legend_justification="center",
                  legend_box="horizontal", legend_key_width=45, legend_box_spacing=0.01)


def title_case(s):
    """Title Case For Figure Text: short joining words stay lower-case, hyphenated parts are each capitalised."""
    small = {"a", "an", "the", "of", "and", "or", "vs", "in", "on", "to", "per"}
    cap = lambda w: re.sub(r"[A-Za-z]", lambda m: m.group().upper(), w, count=1)
    words = [w if (w.lower().strip("()“”") in small and i) or len(w) == 1 else "-".join(cap(x) for x in w.split("-"))
             for i, w in enumerate(s.split(" "))]
    return " ".join(words)


METHODS = {title_case(m): c for m, c in METHOD_COLORS.items()}  # the poster legends' names for the vector kinds
QUESTIONS = dict(zip(["Include", "Exclude"], QUESTION_COLORS.values()))


def base_theme(**kw):
    """theme_minimal without grid lines, Palatino, grey facet strips, boxed panels."""
    return theme_minimal() + theme(
        text=element_text(family=FONT, color=INK),
        axis_text=element_text(family=FONT, color=INK_2),
        plot_title=element_text(family=FONT, size=13, ha="center"),
        plot_caption=element_text(family=FONT, size=8, color=INK_2, ha="left"),
        strip_text=element_text(family=FONT, size=9, color=INK, margin={"t": 4, "b": 4}),
        strip_background=element_rect(fill=STRIP, color=BORDER, size=0.6),
        panel_border=element_rect(fill=None, color=BORDER, size=0.6),
        panel_grid=element_blank(),
        **kw,
    )


def poster_theme(height=5.6, legend=(1, 1), width=12, scale=1.0):
    """The look of the poster figures: 12 inches wide (or `width`), large bold text (font sizes × `scale`), the legend
    inside a corner of the panels (`legend` = that corner as (x, y), 1 = right/top)."""
    x, y = legend
    bold = lambda size, **kw: element_text(family=FONT, weight="bold", size=size * scale, **kw)
    return (
        base_theme(figure_size=(width, height), legend_position="inside", legend_justification=(x, y),
                   legend_position_inside=(0.005 + 0.99 * x, 0.015 + 0.97 * y), legend_direction="vertical",
                   legend_background=element_rect(fill=BG, alpha=0.7, color="none"),
                   plot_background=element_rect(fill=BG, color="none"))
        + theme(strip_background=element_rect(fill=STRIP_BG, color=BORDER, size=0.6),
                text=element_text(family=FONT, weight="bold", color=INK),
                plot_title=bold(20, ha="center"),
                axis_title=bold(17),
                axis_text=bold(14, color=INK),
                strip_text=bold(15, margin={"t": 6 * scale, "b": 6 * scale}),
                legend_text=bold(14),
                legend_title=bold(14))
    )


def save(plot, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, dpi=160, verbose=False)
    print(f"wrote {path}")


# ---------------------------------------------------------------- shared pieces
def layer_panels(layers):
    """"Layer 16", "Layer 31", ... as an ordered categorical, for facetting by injection layer."""
    return pd.Categorical("Layer " + pd.Series(layers).astype(str).to_numpy(),
                          [f"Layer {l}" for l in sorted(pd.unique(layers))], ordered=True)


def k_axis(d, ks):
    """d with an x column for a log2 strength axis on which k = 0 sits one step left of the smallest k > 0, and the
    scale that labels it."""
    zero = np.log2(min(k for k in ks if k > 0)) - 1
    d = d.assign(x=np.where(d.k == 0, zero, np.log2(d.k.where(d.k > 0, 1))))
    return d, scale_x_continuous(breaks=[zero if k == 0 else np.log2(k) for k in ks], labels=[f"{k:g}" for k in ks])


def method_column(d):
    return pd.Categorical(d.vectors.map(METHOD_NAMES).map(title_case), list(METHODS), ordered=True)


def main_runs(s):
    """The rows of summary.csv the curve figures show: passages labelled 1 / 2, no filler."""
    return s[(s["labels"] == "12") & (s.filler == 0) & s["mean"].notna()]


def strength_curves(d, ks, color, colors, ylab, chance, group=None, linetype=None, linetypes=None, extra=()):
    """`mean` against strength k, a line per `color` level (and per `linetype` level), a panel per layer, at poster
    grid size."""
    d, x_scale = k_axis(d, ks)
    d = d.assign(panel=layer_panels(d.layer))
    shown = [c for c in colors if c in set(d[color])]
    style = dict(color=color, fill=color, **({"group": group} if group else {}), **({"linetype": linetype} if linetype else {}))
    p = (
        ggplot(d, aes("x", "mean", **style))
        + geom_hline(yintercept=chance, linetype="dashed", color=INK_3, size=0.5)
        + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.12, color=None, linetype="solid", show_legend=False)
        + geom_line(size=1.5)
        + geom_point(size=2.4, show_legend=False)
        + facet_wrap("panel", nrow=-(-d.layer.nunique() // 4))
        + x_scale
        + scale_color_manual(values=colors, breaks=shown)
        + scale_fill_manual(values=colors, breaks=shown)
        + labs(x="Strength k (× α*)", y=ylab, color="", fill="", linetype="")
        + poster_theme(**POSTER_GRID)
        + theme(**GRID_THEME)
    )
    if linetypes:
        p += scale_linetype_manual(values=linetypes)
    for layer in extra:
        p += layer
    return p


# ---------------------------------------------------------------- strength curves
STRENGTH_CURVES = {  # measure: (folder, y label, chance)
    "detect": ("detection", "Detection (% Correct)", 50),
    "identify": ("identification", "Identification (% Correct)", 25),
}


def strength_figure(s, measure):
    """One measure (detection: % correct, not the bias-free measure) against strength, concept and random vectors
    together → <folder>/curves_12.pdf. Only the strengths every layer has for both kinds of vector."""
    folder, ylab, chance = STRENGTH_CURVES[measure]
    d = main_runs(s)
    d = d[d.measure == measure]
    done = d.pivot_table(index="k", columns=["vectors", "layer"], values="mean").notna()
    ks = sorted(done.index[done.all(axis=1)])
    if len(ks) < 2:
        return
    d = d[d.k.isin(ks)].assign(method=method_column)
    p = strength_curves(d, ks, "method", METHODS, ylab, chance, extra=[
        scale_y_continuous(breaks=range(chance, 101, 25)),
        coord_cartesian(ylim=(chance, 100)),  # dips just below chance run off the bottom rather than being dropped
    ])
    save(p, OUT / folder / "curves_12.pdf")


def include_exclude_figure(s, kind, specific=False):
    """Include and exclude against strength → include_exclude/curves_<kind>_12.pdf. Only the strengths every layer
    has complete results for (the concept word and the control words, under both questions). Solid = the injected
    concept's word, dashed = control words. specific: one line per question, the concept's rise minus the control
    words' → curves_<kind>_12_specific.pdf."""
    need = ["include", "exclude", "include_other", "exclude_other"]
    d = main_runs(s)
    d = d[d.vectors == kind]
    shown = d[d.measure.isin(["include_specific", "exclude_specific"] if specific else need)]
    d = d[d.measure.isin(need)]
    done = d.groupby(["layer", "k"]).measure.nunique().unstack("layer") == len(need)  # k × layer: all four there
    done = done[[l for l in done.columns if done[l].sum() > 1]]  # layers with control words at some strength
    ks = sorted(done.index[done.all(axis=1)]) if len(done.columns) else []
    if len(ks) < 2:  # no control words for this vector kind yet
        return
    d = shown[shown.k.isin(ks) & shown.layer.isin(done.columns)]
    words = ["Injected Concept", "Control Words"]
    d = d.assign(question=pd.Categorical(np.where(d.measure.str.startswith("include"), "Include", "Exclude"),
                                         list(QUESTIONS), ordered=True),
                 word=pd.Categorical(np.where(d.measure.str.endswith("_other"), words[1], words[0]), words, ordered=True))
    ylab = "Rise in Log-Odds\n(Concept − Control Words)" if specific else "Rise in Log-Odds"
    p = strength_curves(d, ks, "question", QUESTIONS, ylab, 0, group="measure",
                        **({} if specific else dict(linetype="word", linetypes=dict(zip(words, ["solid", "dashed"])))))
    save(p, OUT / "include_exclude" / f"curves_{kind}_12{'_specific' if specific else ''}.pdf")


DIFFERENT_CURVES = {  # measure: (file, y label, chance)
    "detect": ("detection", "Detection (% Correct)", 50),
    "detect_pair": ("detection_bias_free", "Bias-Free Detection\n(% of Passages)", 50),
    "identify": ("identification", "Identification (% Correct)", 25),
}


def different_figures(s):
    """Same passage twice (solid) vs two different passages (dashed) → different/detection.pdf,
    detection_bias_free.pdf, identification.pdf (concept and random vectors) and include_exclude_<kind>.pdf (the
    concept's rise minus the control words', under each question)."""
    s = s[s["mean"].notna()]
    s = s.assign(passages=pd.Categorical(s.passages, PASSAGE_KINDS, ordered=True),
                 method=method_column(s),
                 question=pd.Categorical(np.where(s.measure.str.startswith("include"), "Include", "Exclude"),
                                         list(QUESTIONS), ordered=True))
    passages = dict(linetype="passages", linetypes=dict(zip(PASSAGE_KINDS, ["solid", "dashed"])))

    def figure(d, color, colors, ylab, chance, path):
        if d.empty:
            return
        d = d.assign(series=d[color].astype(str) + "|" + d.passages.astype(str))
        save(strength_curves(d, sorted(d.k.unique()), color, colors, ylab, chance, group="series", **passages), path)

    for measure, (name, ylab, chance) in DIFFERENT_CURVES.items():
        figure(s[s.measure == measure], "method", METHODS, ylab, chance, OUT / "different" / f"{name}.pdf")
    for kind in s.vectors.unique():
        figure(s[(s.vectors == kind) & s.measure.isin(["include_specific", "exclude_specific"])], "question",
               QUESTIONS, "Rise in Log-Odds\n(Concept − Control Words)", 0,
               OUT / "different" / f"include_exclude_{kind}.pdf")


# ---------------------------------------------------------------- across layers
def layer_figure(d, path, color, colors, panel, title, ylab, chance=(), base=None, free=False, height=5.6, log=False,
                 span=None, label=False):
    """One by-layer figure (a 12-inch-wide PDF): a line per `color` level across injection layers, a panel per `panel`
    level, the legend inside the top right of the last panel. base: the no-injection level, drawn dotted; label:
    name the reference lines; span: values (panel, mean) a panel's axis must reach, e.g. zero."""
    if d.empty:
        return
    relabel = lambda x: pd.Categorical(x.astype(str).map(title_case), [title_case(c) for c in x.cat.categories], ordered=True)
    d = d.assign(**{color: relabel(d[color]), panel: relabel(d[panel])})
    colors = {title_case(k): v for k, v in colors.items()}
    if base is not None:
        base = base.assign(**{panel: relabel(base[panel])})
    p = ggplot(d, aes("layer", "mean", color=color, fill=color))
    for y in chance:
        p += geom_hline(yintercept=y, linetype="dashed", color=INK_3, size=0.5)
    notes = [dict(panel=q, y=y, label="Chance") for q in d[panel].cat.categories for y in chance] if label else []
    if base is not None and len(base):  # the no-injection level: the same at every layer, so one line across the panel
        level = base.groupby(panel, observed=True)["mean"].mean().reset_index()
        p += geom_hline(level, aes(yintercept="mean"), linetype="dotted", color=INK_2, size=0.8)
        for r in level.itertuples(index=False) if label else []:
            same = [n for n in notes if n["panel"] == r[0] and abs(n["y"] - r[1]) < 1e-6]  # it sits on the chance line
            for n in same:
                n["label"] = "Chance = No Injection"
            notes += [] if same else [dict(panel=r[0], y=r[1], label="No Injection")]
    if notes:  # each reference line named just underneath it, at the left edge of the panel
        notes = pd.DataFrame(notes).rename(columns={"panel": panel})
        notes[panel] = pd.Categorical(notes[panel], d[panel].cat.categories, ordered=True)
        reach = max(d.hi.max(), notes.y.max()) - min(d.lo.min(), notes.y.min())
        p += geom_text(notes.assign(x=-np.inf, y=notes.y - 0.012 * reach, label=" " + notes.label),  # at the panel's edge
                       aes("x", "y", label="label"), inherit_aes=False, ha="left", va="top", size=12, family=FONT,
                       fontweight="bold", color=INK)
    shown = [c for c in colors if c in set(d[color])]
    p = (
        p + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.15, color=None)
        + geom_line(size=0.9)
        + geom_point(size=2)
        + facet_wrap(panel, nrow=1, scales="free_y" if free else "fixed")
        + scale_x_continuous(breaks=sorted(d.layer.unique()))
        + scale_color_manual(values=colors, breaks=shown)
        + scale_fill_manual(values=colors, breaks=shown)
        + labs(x="Injection Layer", y=title_case(ylab), color="", fill="", title=title)
        + poster_theme(height)
    )
    if d[panel].nunique() == 1:  # a single panel needs no heading
        p += theme(strip_text=element_blank(), strip_background=element_blank())
    if d[color].nunique() == 1:  # nor a single line a legend
        p += theme(legend_position="none")
    if log:
        p += scale_y_log10()
    if span is not None:
        p += geom_blank(span.assign(**{panel: relabel(span[panel])}), aes("layer", "mean"), inherit_aes=False)
    save(p, path)


def best_k(d, measure):
    """For each injection layer and vector kind, the strength k > 0 at which `measure` is highest → (vectors, layer, k).
    Only strengths that every vector kind has at that layer compete, so neither kind gets a stronger push."""
    m = d[(d.measure == measure) & (d.k > 0)]
    m = m[m.groupby(["layer", "k"]).vectors.transform("nunique") == m.vectors.nunique()]
    return m.loc[m.groupby(["vectors", "layer"])["mean"].idxmax(), ["vectors", "layer", "k"]]


def by_layer(s):
    """Each evaluation across injection layers at its best strength, mean-diff against random vectors:
    detection/, identification/ and include_exclude/by_layer_best.pdf, and include_exclude/control_automatic_best.pdf
    (Jacoby's two estimates). The best k is chosen per layer and vector kind on the figure's headline measure
    (detection % correct, identification % correct, include's rise over the control words), and every measure in the
    figure is shown at that k. Dashed = chance, dotted = no injection."""
    d = main_runs(s)
    d = d.assign(method=pd.Categorical(d.vectors.map(METHOD_NAMES), list(METHOD_COLORS), ordered=True))
    zero = d[(d.k == 0) & (d.vectors == "concept")]
    at = lambda measure: d.merge(best_k(d, measure), on=["vectors", "layer", "k"])
    name = "by_layer_best.pdf"

    def panels(x, names):
        x = x[x.measure.isin(names)]
        return x.assign(panel=pd.Categorical(x.measure.map(names), list(names.values()), ordered=True))

    kinds = {"detect": "% correct", "detect_pair": "bias-free (% of passages)"}
    layer_figure(panels(at("detect"), kinds), OUT / "detection" / name, "method", METHOD_COLORS, "panel",
                 title="Gemma3-27B: Detection Across Layers (Best k)", ylab="detection", chance=[50], label=True,
                 base=panels(zero[zero.measure == "detect"], kinds))

    one = {"identify": "% correct among the four options"}
    layer_figure(panels(at("identify"), one), OUT / "identification" / name, "method", METHOD_COLORS, "panel",
                 title="Gemma3-27B: Identification Across Layers (Best k)", ylab="identification (% correct)",
                 chance=[25], label=True, base=panels(zero, one))

    best = at("include_specific")
    jac = best[best.measure.isin(["include_specific", "exclude_specific"])]
    jac = jac.assign(question=pd.Categorical(jac.measure.str.replace("_specific", "").map(QUESTION_NAMES),
                                             list(QUESTION_COLORS), ordered=True))
    layer_figure(jac, OUT / "include_exclude" / name, "question", QUESTION_COLORS, "method", chance=[0],
                 title="Gemma3-27B: Include vs Exclude Across Layers (Best k)", ylab="Δ log-odds vs control words")

    ca = {"control": "control C = I − E (percentage points)", "automatic": "automatic A = E / (1 − C) (%)"}
    layer_figure(panels(best, ca), OUT / "include_exclude" / "control_automatic_best.pdf", "method", METHOD_COLORS,
                 "panel", title="Gemma3-27B: Jacoby's Control and Automatic Estimates Across Layers (Best k)",
                 ylab="", free=True)


# ---------------------------------------------------------------- calibration
def pretty_alpha(b):
    """Tick labels for log2 strengths: 1/4, 1, 4, ..."""
    return [f"{2.0 ** x:g}" if x >= 0 else f"1/{2.0 ** -x:g}" for x in b]


def dose_response(kind):
    """calibration/dose_response_<kind>.pdf: each concept's rise in say-a-word log-odds against raw strength on the
    calibration passages, one panel per layer, with the mean over concepts and each concept's α* (where it first
    crosses the target)."""
    curve, alpha = pd.read_csv(calibration_curve_path(kind)), pd.read_csv(alpha_path(kind))
    d = curve.assign(x=np.log2(curve.alpha), panel=layer_panels(curve.layer))
    mean = d.groupby(["panel", "x"], observed=True).delta.mean().reset_index()
    star = alpha.dropna(subset=["alpha_star"])
    star = star.assign(x=np.log2(star.alpha_star), y=TARGET, panel=layer_panels(star.layer))
    keys = ["Each Concept", "Mean Over Concepts", "α* of Each Concept"]
    colors = dict(zip(keys, ["#9fc2ec", INK, "#eda100"]))  # not orange, which means random vectors elsewhere
    ticks = np.arange(np.floor(d.x.min()), d.x.max(), 4)  # 1/16, 1, 16 (none at the right edge, where the next panel starts)
    p = (
        ggplot(d, aes("x", "delta"))
        + geom_hline(yintercept=TARGET, linetype="dashed", color=INK_3, size=0.5)
        + geom_line(aes(group="concept", color=f"'{keys[0]}'"), alpha=0.6, size=0.5)
        + geom_line(mean, aes(color=f"'{keys[1]}'"), size=1.8)
        + geom_point(star, aes(y="y", color=f"'{keys[2]}'"), size=2.4, alpha=0.9)
        + facet_wrap("panel", nrow=-(-alpha.layer.nunique() // 4))
        + scale_x_continuous(breaks=ticks, labels=pretty_alpha(ticks))
        + coord_cartesian(ylim=(-1, 15))  # curves outside run off the top and bottom
        + scale_color_manual(values=colors, breaks=keys)
        + guides(color=guide_legend(override_aes={"alpha": 1, "size": [1.2, 1.8, 2.4],
                                                  "linetype": ["solid", "solid", "none"], "shape": ["", "", "o"]}))
        + labs(x="Strength α (× Raw Vector)", y="Rise in Log-Odds", color="")
        + poster_theme(**POSTER_GRID)
        + theme(**GRID_THEME)
    )
    save(p, OUT / "calibration" / f"dose_response_{kind}.pdf")


def dose_response_layer(kind, layer=16):
    """calibration/dose_response_layer<layer>_<kind>.pdf: every concept's dose-response at one layer against the raw
    strength, with its α* marked where it crosses the target."""
    curve, alpha = pd.read_csv(calibration_curve_path(kind)), pd.read_csv(alpha_path(kind))
    d = curve[curve.layer == layer].assign(x=lambda c: np.log2(c.alpha))
    if d.empty:
        return
    star = alpha[(alpha.layer == layer) & alpha.alpha_star.notna()].assign(x=lambda a: np.log2(a.alpha_star), y=TARGET)
    mean = d.groupby("x").delta.mean().reset_index()
    lo, hi = d.x.min(), d.x.max()
    ticks = np.arange(np.floor(lo), hi + 1, 1 if hi - lo < 7 else 2)
    top = d.delta.max()
    note = lambda y, label, color: annotate("text", x=lo, y=y, label=label, ha="left", va="bottom", size=13,
                                            family=FONT, fontweight="bold", color=color)
    p = (
        ggplot(d, aes("x", "delta"))
        + geom_line(aes(group="concept"), color="#2a78d6", alpha=0.3, size=0.5)
        + geom_hline(yintercept=TARGET, linetype="dashed", color=INK_2, size=0.6)
        + geom_line(mean, color=INK, size=1.4)
        + geom_point(star, aes(y="y"), color="#eb6834", size=2.4, alpha=0.9)
        + note(TARGET + 0.05 * top, "Target: the Odds Double", INK_2)
        + note(TARGET + 0.17 * top, "α* of Each Concept", "#eb6834")
        + scale_x_continuous(breaks=ticks, labels=pretty_alpha(ticks))
        + labs(x="Strength α (× Raw Vector)", y="Rise in Log-Odds", title=f"Calibration at Layer {layer}")
        + poster_theme(height=5.2, width=6)
    )
    save(p, OUT / "calibration" / f"dose_response_layer{layer}_{kind}.pdf")


def dose_response_by_layer(s, kind):
    """calibration/dose_response_by_layer_<kind>.pdf: the rise in log-odds of saying the concept ("say a word") on the
    held-out test passages, against strength in units of each concept's own α*, one line per layer (error bars = 95%
    bootstrap CI), at the strengths every layer has. Calibration set α* so that this rise is the target at k = 1 on
    the separate calibration passages."""
    n = main_runs(s)
    n = n[(n.vectors == kind) & (n.measure == "priming") & (n.k > 0)]
    done = n.pivot_table(index="k", columns="layer", values="mean").notna().all(axis=1)
    n = n[n.k.isin(done.index[done])]  # only the strengths every layer has
    if n.empty:
        return
    layers = sorted(n.layer.unique())
    palette = {str(l): to_hex(colormaps["viridis"](0.92 * i / max(len(layers) - 1, 1))) for i, l in enumerate(layers)}
    n = n.assign(x=np.log2(n.k), Layer=pd.Categorical(n.layer.astype(str), list(palette), ordered=True))
    ticks = sorted(n.x.unique())
    dodge = position_dodge(width=0.18)  # so the layers' error bars do not sit on top of each other
    p = (
        ggplot(n, aes("x", "mean", color="Layer", group="Layer"))
        + geom_hline(yintercept=TARGET, linetype="dashed", color=INK_2, size=0.6)
        + geom_vline(xintercept=0, linetype="dashed", color=INK_2, size=0.6)
        + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.12, size=0.6, position=dodge)
        + geom_line(size=1.0, position=dodge)
        + geom_point(size=1.6, position=dodge)
        + scale_x_continuous(breaks=ticks, labels=pretty_alpha(ticks))
        + scale_color_manual(values=palette)
        + guides(color=guide_legend(ncol=2))
        + labs(x="Strength (× Each Concept's α*)", y="Rise in Log-Odds", title="Dose-Response After Calibration")
        + poster_theme(height=5.2, legend=(0, 1), width=6)
    )
    save(p, OUT / "calibration" / f"dose_response_by_layer_{kind}.pdf")


def calibration_by_layer(kind):
    """Calibration across injection layers → calibration/by_layer_alpha_<kind>.pdf (the calibrated strength, as a
    multiplier of the vector and as a share of the residual stream's norm; geometric mean over concepts with a 95%
    bootstrap CI, log scale) and by_layer_response_<kind>.pdf (how many concepts calibrate, and the largest rise in
    log-odds any strength on the grid reaches)."""
    alpha = pd.read_csv(alpha_path(kind))

    def summary(column, name, geometric=False):
        rows = []
        for layer, g in alpha.groupby("layer"):
            x = g[column].dropna().to_numpy()
            m, lo, hi = boot(np.log(x) if geometric else x) if len(x) else (np.nan,) * 3
            rows.append(dict(layer=layer, mean=m, lo=lo, hi=hi))
        d = pd.DataFrame(rows)
        if geometric:
            d[["mean", "lo", "hi"]] = np.exp(d[["mean", "lo", "hi"]])
        return d.assign(panel=name)

    def figure(parts, name, title, span=None, **kw):
        d = pd.concat(parts).dropna(subset=["mean"])
        order = list(dict.fromkeys(d.panel))
        d = d.assign(method=pd.Categorical([METHOD_NAMES[kind]] * len(d), list(METHOD_COLORS), ordered=True),
                     panel=pd.Categorical(d.panel, order, ordered=True))
        if span is not None:
            kw["span"] = span.assign(panel=pd.Categorical(span.panel, order, ordered=True))
        layer_figure(d, OUT / "calibration" / f"{name}_{kind}.pdf", "method", METHOD_COLORS, "panel", title=title,
                     ylab="", **kw)

    figure([summary("alpha_star", "α* (multiple of the raw vector)", geometric=True),
            summary("inj_norm_ratio", "α* · |vector| / |residual stream|", geometric=True)],
           "by_layer_alpha", "Gemma3-27B: Calibrated Strength α* Across Layers", free=True, log=True)
    count = alpha.groupby("layer").alpha_star.agg(lambda s: s.notna().sum()).rename("mean").reset_index()
    counted = f"concepts calibrated (of {len(CONCEPTS)})"
    figure([count.assign(lo=count["mean"], hi=count["mean"], panel=counted),
            summary("max_delta", "largest rise in log-odds on the grid")],
           "by_layer_response", "Gemma3-27B: Calibration Across Layers", free=True,
           span=pd.DataFrame(dict(layer=count.layer.iloc[0], mean=[0, len(CONCEPTS)], panel=counted)))


# ---------------------------------------------------------------- all
def plot():
    s = pd.read_csv(OUT / "summary.csv", dtype={"labels": str})
    for path in sorted(OUT.glob("alpha_*.csv")):
        kind = path.stem.removeprefix("alpha_")
        dose_response(kind)
        dose_response_layer(kind)
        dose_response_by_layer(s, kind)
        calibration_by_layer(kind)
    for measure in STRENGTH_CURVES:
        strength_figure(s, measure)
    for kind in s.vectors.unique():
        include_exclude_figure(s, kind)
        include_exclude_figure(s, kind, specific=True)
    by_layer(s)
    if (OUT / "different" / "summary.csv").exists():
        different_figures(pd.read_csv(OUT / "different" / "summary.csv", dtype={"labels": str}))


if __name__ == "__main__":
    plot()
