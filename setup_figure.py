"""A diagram of the two-passage paradigm of psychophysics.py → results/psychophysics/setup.png (portrait: the
questions under the prompt) and setup_wide.png (landscape: the questions beside it).

Laid out after the "injected thoughts" figure of Lindsey 2026, with poster-sized text: the prompt with the injected
copy marked, and the five questions that are scored from that one injected context.

  python setup_figure.py
"""
import textwrap

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from analysis import OUT
from figures import FONT
from scoring import PREAMBLE, QUESTIONS
from stimuli import TEST

BG, PANEL, BUBBLE, CARD_BG, INK, MUTED = "#F3EADB", "#E9DECB", "#E2D6C1", "#FBF7F0", "#141413", "#5e5d59"
INJECT, INJECT_BG = "#c2410c", "#F8D9C3"  # the injected copy
BUBBLE_GAP = 0.15  # between two question bubbles of one card
POSTER_WIDTH = 36  # inches, for setup_wide.png
W = 15.6  # width of the prompt panel in inches; all coordinates below are inches from the bottom-left corner
PASSAGE = TEST[1]  # the library-hours passage


def text(ax, x, y, s, size=11, weight="normal", style="normal", color=INK, ha="left", va="top", width=None):
    """Text at (x, y); `width` wraps it to that many characters. Returns the number of lines."""
    if width:
        s = "\n".join(textwrap.fill(line, width) if line else "" for line in s.split("\n"))
    ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, color=color, ha=ha, va=va, family=FONT,
            linespacing=1.35)
    return s.count("\n") + 1


def box(ax, x, y, w, h, fill, dashed=False, edge=None, radius=0.12, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge or (INK if dashed else "none"), linewidth=lw if (dashed or edge) else 0,
                                linestyle=(0, (3, 2.5)) if dashed else "solid"))


def user(ax, x, y, r=0.2):
    """A person icon centred at (x, y)."""
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=INK, lw=1.4))
    ax.add_patch(Circle((x, y + r * 0.22), r * 0.33, facecolor="none", edgecolor=INK, lw=1.3))
    ax.plot([x - r * 0.62, x - r * 0.4, x + r * 0.4, x + r * 0.62], [y - r * 0.72, y - r * 0.3, y - r * 0.3, y - r * 0.72],
            color=INK, lw=1.3, solid_capstyle="round")


def robot(ax, x, y, s=0.2):
    """A model icon centred at (x, y)."""
    box(ax, x - s, y - s * 0.85, 2 * s, 1.7 * s, "none", edge=INK, radius=0.05, lw=1.4)
    ax.plot([x, x], [y + s * 0.85, y + s * 1.25], color=INK, lw=1.3)
    for dx in (-0.42, 0.42):
        ax.add_patch(Circle((x + dx * s, y + s * 0.25), s * 0.11, color=INK))
    box(ax, x - s * 0.5, y - s * 0.55, s, s * 0.4, "none", edge=INK, radius=0.03, lw=1.1)


def syringe(ax, x, y, s=0.26):
    """A syringe coming from the upper right, its needle tip at (x, y)."""
    d = s / 2 ** 0.5
    ax.plot([x, x + d], [y, y + d], color=INK, lw=1.0)  # needle
    bx, by = x + d, y + d
    barrel = [(bx - 0.05, by + 0.05), (bx + 0.05, by - 0.05), (bx + 0.05 + d, by - 0.05 + d), (bx - 0.05 + d, by + 0.05 + d)]
    ax.add_patch(Polygon(barrel, closed=True, facecolor=CARD_BG, edgecolor=INK, lw=1.2))
    ax.plot([bx + d, bx + 1.6 * d], [by + d, by + 1.6 * d], color=INK, lw=1.2)  # plunger
    ax.plot([bx + 1.6 * d - 0.07, bx + 1.6 * d + 0.07], [by + 1.6 * d + 0.07, by + 1.6 * d - 0.07], color=INK, lw=1.4)


def justified(ax, x, mid, w, words, size):
    """`words` ((word, weight) pairs) as a fully justified block `w` inches wide, vertically centred on `mid`; the
    last line is set ragged. Matplotlib cannot justify, so each word is placed on its own."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    def measure(s, weight):
        t = ax.text(0, 0, s, fontsize=size, fontweight=weight, family=FONT)
        width = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return width
    space = measure("x x", "normal") - measure("xx", "normal")
    sized = [(s, weight, measure(s, weight)) for s, weight in words]
    rows, row = [], []
    for word in sized:
        if row and sum(c[2] for c in row) + space * len(row) + word[2] > w:
            rows.append(row)
            row = []
        row.append(word)
    rows.append(row)
    step = size / 72 * 1.35
    y = mid + step * len(rows) / 2
    for i, row in enumerate(rows):
        gap = (w - sum(c[2] for c in row)) / (len(row) - 1) if i < len(rows) - 1 and len(row) > 1 else space
        cx = x
        for s, weight, ww in row:
            ax.text(cx, y, s, fontsize=size, fontweight=weight, color=INK, ha="left", va="top", family=FONT)
            cx += ww + gap
        y -= step


def lines(s, size, inches, em=0.47):
    """`s` wrapped to fit `inches` at font size `size` (Palatino averages about `em` per character), and its height."""
    s = "\n".join(textwrap.fill(line, int(inches / (em * size / 72))) if line else "" for line in s.split("\n"))
    return s, (s.count("\n") + 1) * size / 72 * 1.35 if s else 0.0


def draw(wide=False):
    TITLE, BODY, NOTE, CARD = 48, 20.5, 17.5, 18.5  # font sizes, for a poster
    L, R, pad = 0.3, W - 0.3, 0.26  # panel edges; padding inside a bubble
    bx = L + 0.95  # left edge of the chat bubbles
    bw = R - 0.25 - bx
    setup, h_setup = lines(PREAMBLE, BODY, bw - 2 * pad, em=0.44)  # long lines: fill the bubble
    pw = 10.1  # width of the passage column; the notes sit to its right
    passage, h_passage = lines(PASSAGE, BODY, pw - 0.3)
    row = BODY / 72 * 1.35  # one line of body text
    notes = [lines(s, NOTE, R - 0.25 - (bx + pw + 1.05) - 0.15)[0] for s in (
        "“bread” vector injected over this copy only (strength k · α*); it reaches the question through the KV cache.",
        "No injection.")]
    q = lambda name, **kw: QUESTIONS[name].format(**kw)
    cards = [
        ("Calibration", q("say"), "P(“bread”)", "Sets the injection strength."),
        ("Detection", q("detect", p1="1", p2="2", pkind="number"), "P(“1”) vs P(“2”)", "Can the model tell which copy was injected?"),
        ("Identification", q("identify", options="A) Dust\nB) Bread\nC) Poetry\nD) Phones", okind="letter"),
         "P(“B”) among A–D", "Can the model tell which concept was injected?"),
        ("Include vs Exclude", (("Include", q("include")), ("Exclude", q("exclude"))), "P(“bread”)",
         "Can the model report or withhold the concept? Adapts Jacoby's (1991) process dissociation."),
    ]
    gap = 0.18
    QL, QW, c_note, c_q, c_r = R + 0.35, 10.8, 3.2, 4.2, 2.7  # wide: the questions' left edge, width and three columns
    n = len(cards)
    cw = QW if wide else (R - L - (n - 1) * gap) / n
    h_label = CARD / 72 * 1.45  # a bubble's label, when a card has several

    def blocks(qs):
        """A card's question bubbles as (label, text, height): one unlabelled bubble, or one labelled bubble per
        (label, question); and their total height, counted as for a single bubble."""
        qs = [(None, qs)] if isinstance(qs, str) else qs
        b = [(label, *lines(s, CARD, (c_q if wide else cw - 0.36) - 2 * 0.14, em=0.47 if wide else 0.5)) for label, s in qs]
        return b, sum(h + (h_label if label else 0) for label, _, h in b) + (0.3 + BUBBLE_GAP) * (len(b) - 1)
    cards = [(n, *blocks(qu), r, *lines(note, NOTE - 1, c_note if wide else cw - 0.36)) for n, qu, r, note in cards]
    h_q, h_n = max(c[2] for c in cards), max(c[5] for c in cards)
    h_name = (CARD + 2) / 72 * 1.7

    def bubbles(x, top, w, qs, total=None):
        """The question bubbles stacked down from `top`; the last one stretched so that they span `total`."""
        start = top
        for i, (label, s, h) in enumerate(qs):
            if label:
                text(ax, x + 0.02, top, label, size=CARD, weight="bold")
                top -= h_label
            if total is not None and i == len(qs) - 1:
                h = total - 0.3 - (start - top)
            box(ax, x, top - h - 0.3, w, h + 0.3, BUBBLE, dashed=True, radius=0.09)
            text(ax, x + 0.14, top - 0.15, s, size=CARD)
            top -= h + 0.3 + BUBBLE_GAP

    # heights, top to bottom
    h_title = TITLE / 72 * 1.9
    h_trial = pad + row * 1.4 + 2 * (row * 1.4 + h_passage + 0.35) + row + pad
    h_prompt = 0.25 + (h_setup + 2 * pad) + 0.5 + (row + 2 * 0.17) + 0.25 + h_trial + 0.25
    h_card = 0.25 + CARD / 72 * 1.9 + (h_q + 0.3) + 0.25 + 0.62 + (0.25 + h_n if h_n else 0) + 0.2
    H = 0.3 + h_title + h_prompt + 0.35 + h_card + 0.3
    width = W
    if wide:  # one row per question beside the prompt, the rows stretched to the prompt's height
        rows = [max(h_name + c[5], c[2] + 0.3, 0.62) + 0.44 for c in cards]
        stretch = max(0, h_prompt - sum(rows) - (n - 1) * gap) / n
        rows = [r + stretch for r in rows]
        h_prompt = max(h_prompt, sum(rows) + (n - 1) * gap)  # both columns the same height
        H = 0.3 + h_title + h_prompt + 0.3
        width = QL + QW + 0.3

    fig = plt.figure(figsize=(width, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, H))
    ax.axis("off")

    # ---------------------------------------------------------------- the prompt
    y = H - 0.3
    text(ax, L + 0.1, y, "Two-Passage Experimental Paradigm", size=TITLE, weight="bold")
    if wide:  # the model, in the space right of the title
        about = [("Setup.", "bold")] + [(w, "normal") for w in (
            "We test Gemma3-27B, a 62-layer transformer shown by Macar et al. (2026) to succeed at the "
            "thought-injection paradigm of Lindsey (2025). Each of 50 concepts is injected over 30 passages, "
            "once into passage 1 and once into passage 2.").split()]
        justified(ax, QL, y - h_title / 2, QW, about, size=BODY)  # a block as wide as the question cards
    top = y - h_title
    box(ax, L, top - h_prompt, R - L, h_prompt, PANEL, radius=0.18)
    y = top - 0.25
    user(ax, L + 0.47, y - 0.45, r=0.27)  # setup
    box(ax, bx, y - h_setup - 2 * pad, bw, h_setup + 2 * pad, BUBBLE, dashed=True)
    text(ax, bx + pad, y - pad, setup, size=BODY)
    y -= h_setup + 2 * pad + 0.5
    text(ax, bx + 0.05, y + 0.36, "(Prefill)", size=NOTE - 1, style="italic", color=MUTED)  # prefill
    robot(ax, L + 0.47, y - (row + 0.34) / 2, s=0.26)
    box(ax, bx, y - row - 0.34, bw, row + 0.34, BUBBLE, dashed=True)
    text(ax, bx + pad, y - 0.17, "Ok.", size=BODY, style="italic")
    y -= row + 0.34 + 0.25
    user(ax, L + 0.47, y - 0.45, r=0.27)  # trial
    box(ax, bx, y - h_trial, bw, h_trial, BUBBLE, dashed=True)
    y -= pad
    text(ax, bx + pad, y, "Trial 1:", size=BODY)
    y -= row * 1.4
    for label, note, injected in zip(("Passage 1:", "Passage 2:"), notes, (True, False)):
        text(ax, bx + pad, y, label, size=BODY)
        y -= row * 1.4
        if injected:
            box(ax, bx + pad - 0.14, y - h_passage - 0.08, pw, h_passage + 0.2, INJECT_BG, edge=INJECT, radius=0.07, lw=1.6)
            syringe(ax, bx + pad - 0.14 + pw, y - h_passage / 2 - 0.1, s=0.3)
        text(ax, bx + pad, y, passage, size=BODY)
        text(ax, bx + pw + 1.05, y, note, size=NOTE, color=INJECT if injected else MUTED,
             weight="bold" if injected else "normal", style="normal" if injected else "italic")
        y -= h_passage + 0.35
    text(ax, bx + pad, y, "<question>", size=BODY, weight="bold")

    # ---------------------------------------------------------------- the questions
    if wide:
        y = top
        for (name, question, hq, readout, note, hn), h in zip(cards, rows):
            box(ax, QL, y - h, QW, h, PANEL, radius=0.18)
            mid = y - h / 2
            text(ax, QL + 0.22, mid + (h_name + hn) / 2, name, size=CARD + 2, weight="bold")
            text(ax, QL + 0.22, mid + (h_name + hn) / 2 - h_name, note, size=NOTE - 1, style="italic", color=MUTED)
            qx = QL + 0.22 + c_note + 0.2
            bubbles(qx, mid + (hq + 0.3) / 2, c_q, question)
            rx = qx + c_q + 0.2
            box(ax, rx, mid - 0.31, c_r, 0.62, CARD_BG, edge=INK, radius=0.09, lw=1.3)
            text(ax, rx + c_r / 2, mid, readout, size=CARD + 0.5, weight="bold", ha="center", va="center")
            y -= h + gap
        cards = []  # drawn
    top -= h_prompt + 0.35
    for i, (name, question, _, readout, note, _) in enumerate(cards):
        x0 = L + i * (cw + gap)
        box(ax, x0, top - h_card, cw, h_card, PANEL, radius=0.18)
        y = top - 0.25
        text(ax, x0 + 0.18, y, name, size=CARD + 2, weight="bold")
        y -= CARD / 72 * 1.9
        bubbles(x0 + 0.18, y, cw - 0.36, question, total=h_q + 0.3)
        y -= h_q + 0.3 + 0.25
        box(ax, x0 + 0.18, y - 0.62, cw - 0.36, 0.62, CARD_BG, edge=INK, radius=0.09, lw=1.3)
        text(ax, x0 + cw / 2, y - 0.31, readout, size=CARD + 0.5, weight="bold", ha="center", va="center")
        text(ax, x0 + 0.18, y - 0.62 - 0.25, note, size=NOTE - 1, style="italic", color=MUTED)

    path = OUT / ("setup_wide.png" if wide else "setup.png")
    # the wide version is for a poster: POSTER_WIDTH inches at 300 dpi, everything scaled up together
    scale = POSTER_WIDTH / width if wide else 1
    fig.savefig(path, dpi=300 * scale, facecolor=BG)
    Image.open(path).save(path, dpi=(300, 300))  # so the file's physical size is the scaled one
    print(f"wrote {path}")


if __name__ == "__main__":
    draw()
    draw(wide=True)
