"""Pakistan mobile brands: one data set, seven chart types. Easy-English on-screen text, conversational Urdu voice-over,
beat-level sync (every sentence is a beat; charts move exactly while that sentence is spoken), suspense music.

Data: StatCounter Global Stats, Pakistan, mobile vendor share (share of mobile WEB TRAFFIC), monthly Apr 2010 - Oct 2026.
"""
import asyncio
import csv
import hashlib
import os
import random
import subprocess
import sys
import wave

import edge_tts
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FFMPEG, FPS, ease, font, lerp, to_wav, wav_len, wrap  # noqa
from music_suspense import make_music  # noqa

VW, VH = 1080, 1920
BG = (13, 27, 42)
PANEL = (27, 38, 59)
GRID = (44, 58, 82)
GOLD = (242, 183, 5)
TXT = (241, 245, 249)
MUTED = (148, 163, 184)
VOICE = "ur-PK-AsadNeural"

# ------------------------------------------------------------------ data
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "statcounter_pk_mobile_vendor_monthly.csv"),
                                encoding="utf-8")))
rows = [r for r in rows if sum(float(v) for k, v in r.items() if k != "Date") > 0]
rows.sort(key=lambda r: r["Date"])
MONTHS = [r["Date"] for r in rows]
N = len(MONTHS)
MI = {d: i for i, d in enumerate(MONTHS)}
COLS = [k for k in rows[0] if k != "Date"]
S = {c: np.array([float(r[c]) for r in rows]) for c in COLS}
KNOWN = [c for c in COLS if c not in ("Unknown", "Other", "Infinex")]
CHINA = ["Huawei", "Honor", "Oppo", "Vivo", "Realme", "OnePlus", "Xiaomi", "Infinix", "Infinex", "Tecno", "Itel"]
CH = sum(S[c] for c in CHINA)
KM = np.stack([S[c] for c in KNOWN])  # known brands x months
srt = -np.sort(-KM, axis=0)
TOP1, TOP3 = srt[0], srt[:3].sum(axis=0)
YEARS = sorted({d[:4] for d in MONTHS})
YA = {y: {c: float(np.mean([S[c][i] for i, d in enumerate(MONTHS) if d[:4] == y])) for c in COLS} for y in YEARS}
YRANK = {}
for y in YEARS:
    order = sorted(KNOWN, key=lambda c: -YA[y][c])
    YRANK[y] = {c: (order.index(c) + 1 if YA[y][c] >= 0.3 else 99) for c in KNOWN}  # no rank without real share
N5 = {y: sum(1 for c in KNOWN if YA[y][c] >= 5) for y in YEARS}

COL = {
    "Nokia": (59, 130, 246), "Samsung": (34, 211, 238), "Apple": (190, 190, 200), "Sony": (139, 92, 246),
    "HTC": (132, 204, 22), "RIM": (100, 116, 139), "Huawei": (239, 68, 68), "Oppo": (16, 185, 129),
    "Vivo": (99, 102, 241), "Infinix": (245, 158, 11), "Tecno": (236, 72, 153), "Xiaomi": (249, 115, 22),
    "Realme": (250, 204, 21), "QMobile": (20, 184, 166), "Motorola": (168, 85, 247), "LG": (225, 29, 72),
    "Lenovo": (148, 163, 184), "Mobicel": (120, 113, 108), "Itel": (217, 119, 6), "Google": (96, 165, 250),
}
AMB = (245, 158, 11)
SLATE = (100, 116, 139)
DARKG = (58, 68, 90)
GREEN = (46, 194, 126)
RED = (239, 84, 84)
MNAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def val(c, d):
    return float(S[c][MI[d]])


def check():
    def eq(x, y, tol=0.06):
        assert abs(x - y) <= tol, (x, y)
    eq(val("Nokia", "2010-04"), 80.75)
    eq(val("Nokia", "2026-10"), 0.2)
    eq(val("Nokia", "2011-09"), 87.06)
    eq(val("Samsung", "2019-08"), 41.42)
    eq(val("Infinix", "2023-12"), 16.89)
    eq(val("Samsung", "2023-12"), 15.7, 0.1)
    eq(TOP1[MI["2011-09"]], 87.06)
    eq(TOP1[MI["2026-10"]], 16.96)
    eq(TOP3[MI["2011-09"]], 95.4, 0.1)
    eq(TOP3[MI["2026-10"]], 45.4, 0.1)
    assert N5["2011"] == 2 and N5["2022"] == 8, (N5["2011"], N5["2022"])
    assert all(YRANK[str(y)]["Nokia"] == 1 for y in range(2010, 2016))
    assert all(YRANK[str(y)]["Samsung"] == 1 for y in range(2016, 2024))
    assert all(YRANK[str(y)]["Infinix"] == 1 for y in range(2024, 2027))
    eq(CH[MI["2010-04"]], 0.0)
    eq(CH[MI["2019-01"]], 23.1, 0.1)
    eq(CH[MI["2023-01"]], 70.1, 0.1)
    eq(CH[MI["2026-10"]], 68.5, 0.1)
    eq(val("Samsung", "2026-10") + val("Apple", "2026-10"), 25.4, 0.1)
    eq(val("Xiaomi", "2023-01"), 5.56)
    eq(val("Xiaomi", "2026-10"), 10.62)
    for c, d in [("Xiaomi", 5.1), ("Apple", 3.5), ("Infinix", 2.4), ("Tecno", 1.4), ("Huawei", -4.6), ("Oppo", -3.4),
                 ("Samsung", -2.6), ("Vivo", -2.2)]:
        eq(val(c, "2026-10") - val(c, "2023-01"), d, 0.06)
    peaks = {"Nokia": "2011", "Apple": "2010", "QMobile": "2017", "Samsung": "2019", "Huawei": "2020", "Oppo": "2021",
             "Vivo": "2023", "Infinix": "2024", "Tecno": "2024", "Xiaomi": "2026", "Realme": "2025"}
    for c, y in peaks.items():
        assert max(YEARS, key=lambda yy: YA[yy][c]) == y, (c, y)
    print("data checks ok")


# ------------------------------------------------------------------ drawing helpers
def mix(c, a, bg=BG):
    return tuple(int(lerp(bg[i], c[i], max(0.0, min(1.0, a)))) for i in range(3))


def E(dr, xy, text, size, fill, anchor="lm", bold=False):
    dr.text(xy, text, font=font("seguisb.ttf" if bold else "segoeui.ttf", size), fill=fill, anchor=anchor)


def smooth(f):
    f = max(0.0, min(1.0, f))
    return 0.5 * f + 0.5 * f * f * (3 - 2 * f)


def cursor(cues, targets, s, m0, hold=0.85):
    """Chart cursor (float month index): during beat j it travels to targets[j]; reaches it at `hold` of the beat."""
    prev = m0
    for (a, b), tgt in zip(cues, targets):
        if tgt is None:
            continue
        if s < a:
            return prev
        f = (s - a) / max(0.05, (b - a) * hold)
        if f < 1.0:
            return lerp(prev, tgt, smooth(f))
        prev = tgt
    return prev


def beat_alpha(cues, j, s, delay=0.0, ramp=0.5):
    return ease((s - cues[j][0] - delay) / ramp)


def interp(arr, m):
    i = int(np.floor(m))
    j = min(i + 1, len(arr) - 1)
    return float(lerp(arr[i], arr[j], m - i))


def pill(dr, cx, cy, text, a=1.0, size=32, color=GOLD):
    f = font("seguisb.ttf", size)
    w = dr.textlength(text, font=f)
    dr.rounded_rectangle([cx - w / 2 - 28, cy - 34, cx + w / 2 + 28, cy + 34], radius=34, fill=mix(PANEL, a),
                         outline=mix(color, a), width=3)
    dr.text((cx, cy), text, font=f, fill=mix(color, a), anchor="mm")


def label_pill(dr, x, y, text, color, a=1.0, anchor="lm", size=26):
    f = font("seguisb.ttf", size)
    w = dr.textlength(text, font=f)
    x0 = x if anchor == "lm" else (x - w if anchor == "rm" else x - w / 2)
    dr.rounded_rectangle([x0 - 10, y - 20, x0 + w + 10, y + 20], radius=12, fill=mix(PANEL, a))
    dr.text((x0, y), text, font=f, fill=mix(color, a), anchor="lm")


def spread(items, min_gap, lo, hi):
    """items: list of (y, key). Push labels apart so none overlap. Returns dict key -> y."""
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    return {k: y for y, (_, k) in zip(ys, items)}


def year_ticks(dr, X, y_bottom, every=3, start=2011):
    for y in range(start, 2027, every):
        i = MI.get(f"{y}-01")
        if i is None:
            continue
        x = X(i)
        dr.line([x, y_bottom, x, y_bottom + 10], fill=MUTED, width=2)
        E(dr, (x, y_bottom + 34), str(y), 24, MUTED, anchor="mm")


# ------------------------------------------------------------------ bodies
def draw_race(dr, m):
    vals = {b: interp(S[b], m) for b in RACE_BRANDS}
    yi = MONTHS[int(round(m))]
    dr.text((880, 1290), yi[:4], font=font("seguisb.ttf", 150), fill=(36, 52, 76), anchor="mm")
    dr.text((880, 1395), MNAME[int(yi[5:7]) - 1], font=font("segoeui.ttf", 46), fill=(52, 70, 98), anchor="mm")
    axis = max(max(vals.values()), 8.0)
    x0, maxlen, pitch, ytop, sg = 300, 520, 112, 350, 0.025
    order = []
    for b in RACE_BRANDS:
        rk = sum(1 / (1 + np.exp(-np.clip((vals[o] - vals[b] + 0.004 * (RACE_BRANDS.index(o) - RACE_BRANDS.index(b)))
                                          / sg, -30, 30))) for o in RACE_BRANDS if o != b)
        order.append((float(rk), b))
    order.sort()
    spaced, prev = [], -1.0
    for rk, b in order:
        p = max(rk, prev + 0.9)
        spaced.append((p, b))
        prev = p
    for rk, b in sorted(spaced, reverse=True):
        v = vals[b]
        a = max(0.0, min(1.0, v / 0.5)) * max(0.0, min(1.0, 10.3 - rk))
        if a <= 0.01:
            continue
        y = ytop + rk * pitch
        c = COL.get(b, MUTED)
        x1 = x0 + v / axis * maxlen
        E(dr, (x0 - 48, y), b, 36, mix(TXT, a), anchor="rm", bold=(rk < 0.5))
        dr.rounded_rectangle([x0, y - 30, max(x0 + 8, x1), y + 30], radius=14, fill=mix(c, a))
        dr.ellipse([x1 - 38, y - 38, x1 + 38, y + 38], fill=mix(c, a), outline=mix(BG, a), width=5)
        dr.text((x1, y), b[0], font=font("arialbd.ttf", 40), fill=mix((255, 255, 255), a), anchor="mm")
        E(dr, (x1 + 52, y), f"{v:.1f}%", 38, mix(TXT, a), anchor="lm", bold=True)


RACE_BRANDS = [b for b in KNOWN if S[b].max() >= 1.5]
RACE_T = [MI["2010-04"], MI["2011-09"], MI["2015-08"], MI["2023-12"], MI["2026-10"]]
RACE_CALL = [None, "Sep 2011  |  Nokia 87.1% (peak)", "Aug 2015  |  Samsung passes Nokia",
             "Dec 2023  |  Infinix is #1 (16.9%)",
             "Oct 2026  |  Samsung 17.0%  |  Infinix 16.1%"]


def body_race(dr, s, d, cues):
    m = cursor(cues, RACE_T[:], s, 0.0, hold=0.85)
    # the first beat is only an intro: start from Apr 2010
    draw_race(dr, m)
    shown = None
    for j, c in enumerate(RACE_CALL):
        if c and s >= cues[j][0] + 0.75 * (cues[j][1] - cues[j][0]):
            shown = (j, c)
    if shown:
        j, c = shown
        a = ease((s - (cues[j][0] + 0.75 * (cues[j][1] - cues[j][0]))) / 0.4)
        pill(dr, VW / 2, 266, c, a, size=30)


LINE_SER = ["Nokia", "Samsung", "Huawei", "Oppo", "Vivo", "Infinix", "Apple"]
LINE_T = [None, MI["2021-01"], N - 1]
LINE_ANN = [("2011-09", "Nokia", "Nokia peak 87.1%", "r", 0), ("2015-08", "Samsung", "Samsung passes Nokia", "l", -6),
            ("2019-08", "Samsung", "Samsung peak 41.4%", "t", 0), ("2020-04", "Huawei", "Huawei peak 17.2%", "r", -30),
            ("2023-12", "Infinix", "Infinix #1 (Dec 2023)", "b", 0)]


def body_line(dr, s, d, cues):
    x0, x1, y0, y1, ymax = 130, 790, 290, 1330, 90.0
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / ymax  # noqa
    for v in (0, 20, 40, 60, 80):
        dr.line([x0, Y(v), x1, Y(v)], fill=GRID, width=2)
        E(dr, (x0 - 14, Y(v)), f"{v}%", 24, MUTED, anchor="rm")
    year_ticks(dr, X, y1)
    m = cursor(cues, LINE_T, s, 0.0)
    heads = []
    for b in LINE_SER:
        arr = S[b]
        st = int(np.argmax(arr >= 0.3))
        if m <= st:
            continue
        pts = [(X(i), Y(arr[i])) for i in range(st, int(np.floor(m)) + 1)]
        hx, hy = X(m), Y(interp(arr, m))
        pts.append((hx, hy))
        c = COL[b]
        dr.line(pts, fill=c, width=(5 if b == "Apple" else 8), joint="curve")
        dr.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=c, outline=BG, width=3)
        heads.append((hy, b))
    ly = spread([(y, b) for y, b in heads if interp(S[b], m) >= 0.8], 34, y0, y1)
    for b, y in ly.items():
        v = interp(S[b], m)
        E(dr, (min(X(m), x1) + 22, y), f"{b} {v:.1f}%", 26, COL[b], anchor="lm", bold=True)
    for d_, b, text, side, dy in LINE_ANN:
        i = MI[d_]
        if m >= i + 1:
            a = ease((m - i - 1) / 5)
            px, py = X(i), Y(S[b][i])
            dr.ellipse([px - 11, py - 11, px + 11, py + 11], outline=mix(TXT, a), width=4)
            if side == "r":
                label_pill(dr, px + 22, py + dy, text, GOLD, a)
            elif side == "l":
                label_pill(dr, px - 22, py + dy, text, GOLD, a, anchor="rm")
            elif side == "t":
                label_pill(dr, px, py - 44, text, GOLD, a, anchor="mm")
            else:
                label_pill(dr, px, py + 50, text, GOLD, a, anchor="mm")


LAYERS = [("Nokia", S["Nokia"], COL["Nokia"]), ("Samsung", S["Samsung"], COL["Samsung"]),
          ("Apple", S["Apple"], COL["Apple"]), ("Chinese-company brands", CH, AMB)]
OTH = np.clip(100 - (S["Nokia"] + S["Samsung"] + S["Apple"] + CH + S["Unknown"]), 0, None)
LAYERS += [("Other brands", OTH, SLATE), ("Unknown", S["Unknown"], DARKG)]
AREA_T = [MI["2011-06"], MI["2019-01"], MI["2023-01"], N - 1]


def body_area(dr, s, d, cues):
    x0, x1, y0, y1 = 100, 980, 300, 1230
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / 100.0  # noqa
    m = cursor(cues, AREA_T, s, 0.0)
    if m > 0.05:
        i_end = int(np.floor(m))
        xs = [X(i) for i in range(0, i_end + 1)] + [X(m)]
        base = np.zeros(i_end + 2)
        for name, arr, c in LAYERS:
            part = [arr[i] for i in range(0, i_end + 1)] + [interp(arr, m)]
            top = base + np.array(part)
            poly = [(x, Y(v)) for x, v in zip(xs, top)] + [(x, Y(v)) for x, v in zip(xs[::-1], base[::-1])]
            dr.polygon(poly, fill=c)
            base = top
    for v in (0, 25, 50, 75, 100):
        dr.line([x0 - 8, Y(v), x0, Y(v)], fill=MUTED, width=2)
        E(dr, (x0 - 14, Y(v)), f"{v}%", 24, MUTED, anchor="rm")
    dr.rectangle([x0, y0, x1, y1], outline=GRID, width=2)
    year_ticks(dr, X, y1)
    if m > 0.05:
        cx = X(m)
        dr.line([cx, y0, cx, y1], fill=TXT, width=3)
        yi = MONTHS[int(round(m))]
        pill(dr, min(max(cx, 260), 840), y0 - 30, f"{MNAME[int(yi[5:7]) - 1]} {yi[:4]}", 1.0, size=28)
    # live read-out
    mm = max(m, 0.0)
    for k, (name, arr, c) in enumerate(LAYERS):
        cx0 = 70 + (k % 2) * 500
        cy0 = 1345 + (k // 2) * 62
        dr.rounded_rectangle([cx0, cy0 - 14, cx0 + 28, cy0 + 14], radius=6, fill=c)
        E(dr, (cx0 + 44, cy0), f"{name}  {interp(arr, mm):.0f}%", 28, TXT, anchor="lm", bold=False)


CON_T = [None, MI["2011-09"], N - 1, N - 1, N - 1]


def body_conc(dr, s, d, cues):
    x0, x1, y0, y1 = 130, 900, 300, 830
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / 100.0  # noqa
    for v in (0, 25, 50, 75, 100):
        dr.line([x0, Y(v), x1, Y(v)], fill=GRID, width=2)
        E(dr, (x0 - 14, Y(v)), f"{v}%", 24, MUTED, anchor="rm")
    year_ticks(dr, X, y1)
    # beat 1 reveals the #1 line, beat 2 the top-3 line, beats 3+ the columns
    t1 = cues[0]
    t2 = cues[1]
    m1 = 0 if s < t1[0] else (N - 1) * smooth((s - t1[0]) / ((t1[1] - t1[0]) * 0.9))
    m2 = 0 if s < t2[0] else (N - 1) * smooth((s - t2[0]) / ((t2[1] - t2[0]) * 0.9))
    for arr, m, c, name in [(TOP3, m2, (251, 191, 36), "Top 3 brands combined"), (TOP1, m1, (34, 211, 238), "#1 brand")]:
        if m > 0.05:
            pts = [(X(i), Y(arr[i])) for i in range(0, int(np.floor(m)) + 1)] + [(X(m), Y(interp(arr, m)))]
            dr.line(pts, fill=c, width=8, joint="curve")
            hx, hy = pts[-1]
            dr.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=c, outline=BG, width=3)
            tx = 520 if name.startswith("Top") else x0
            E(dr, (tx, y0 - 34), f"{name}: {interp(arr, m):.0f}%", 28, c, anchor="lm", bold=True)
    # column chart: brands with 5%+ share, per year
    cy0, cy1 = 980, 1400
    cx0, cx1 = 130, 1000
    E(dr, (cx0, cy0 - 60), "Brands with 5%+ share, by year (count)", 30, TXT, anchor="lm", bold=True)
    cw = (cx1 - cx0) / len(YEARS)
    for v in (0, 2, 4, 6, 8):
        yy = cy1 - (cy1 - cy0) * v / 9.0
        dr.line([cx0, yy, cx1, yy], fill=GRID, width=1)
        E(dr, (cx0 - 14, yy), str(v), 22, MUTED, anchor="rm")
    ca = beat_alpha(cues, 2, s, 0.0, 2.5)
    for k, y in enumerate(YEARS):
        a = max(0.0, min(1.0, ca * len(YEARS) - k * 0.55))
        h = (cy1 - cy0) * N5[y] / 9.0 * ease(a)
        xx = cx0 + k * cw
        dr.rounded_rectangle([xx + 6, cy1 - h, xx + cw - 6, cy1], radius=6, fill=mix(GREEN if N5[y] >= 6 else (59, 130, 246), a))
        if a > 0.5:
            E(dr, (xx + cw / 2, cy1 - h - 16), str(N5[y]), 24, mix(TXT, a), anchor="mm", bold=True)
        E(dr, (xx + cw / 2, cy1 + 24), "'" + y[2:], 20, MUTED, anchor="mm")


HEAT_ROWS = ["Nokia", "Apple", "QMobile", "Samsung", "Huawei", "Oppo", "Vivo", "Infinix", "Tecno", "Realme", "Xiaomi"]
HEAT_PEAK = {c: max(YEARS, key=lambda y: YA[y][c]) for c in HEAT_ROWS}
HEAT_T = [None, 2021.5, 2026.9]  # year cursor targets (as years), mapped in body


def heat_color(v):
    t = min(1.0, (v / 84.0) ** 0.5)
    stops = [(0.0, (22, 36, 58)), (0.25, (30, 80, 150)), (0.55, (34, 190, 170)), (1.0, (252, 224, 70))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(lerp(c0[i], c1[i], f)) for i in range(3))
    return stops[-1][1]


def body_heat(dr, s, d, cues):
    gx0, gy0, cw, ch = 262, 330, 44, 88
    ycur = 2008.9
    prev = 2008.9
    for (a, b), tgt in zip(cues, HEAT_T):
        if tgt is None:
            continue
        if s < a:
            break
        f = (s - a) / max(0.05, (b - a) * 0.85)
        if f < 1.0:
            ycur = lerp(prev, tgt, smooth(f))
            break
        prev = tgt
        ycur = tgt
    for k, y in enumerate(YEARS):
        if k % 2 == 0 or True:
            E(dr, (gx0 + k * cw + cw / 2, gy0 - 24), "'" + y[2:], 21, MUTED, anchor="mm")
    for r, c in enumerate(HEAT_ROWS):
        cy = gy0 + r * ch
        E(dr, (gx0 - 16, cy + ch / 2), c, 30, COL.get(c, TXT), anchor="rm", bold=True)
        for k, y in enumerate(YEARS):
            a = max(0.0, min(1.0, ycur - (2010 + k) + 1.2))
            if a <= 0:
                continue
            v = YA[y][c]
            col = mix(heat_color(v), a)
            x = gx0 + k * cw
            dr.rectangle([x + 1, cy + 1, x + cw - 1, cy + ch - 1], fill=col)
            if v >= 5 and a > 0.6:
                lum = sum(heat_color(v)) / 3
                E(dr, (x + cw / 2, cy + ch / 2), f"{v:.0f}", 21, (20, 24, 34) if lum > 120 else TXT, anchor="mm", bold=True)
    if ycur >= 2026.5:
        pa = beat_alpha(cues, 2, s, 1.2, 0.8)
        for r, c in enumerate(HEAT_ROWS):
            k = YEARS.index(HEAT_PEAK[c])
            x, cy = gx0 + k * cw, gy0 + r * ch
            dr.rectangle([x, cy, x + cw, cy + ch], outline=mix(TXT, pa), width=4)
    ly = gy0 + len(HEAT_ROWS) * ch + 40
    for k in range(200):
        v = 84.0 * k / 199
        dr.rectangle([gx0 + k * 3.4, ly, gx0 + k * 3.4 + 4, ly + 28], fill=heat_color(v))
    for v in (0, 20, 40, 60, 80):
        E(dr, (gx0 + v / 84.0 * 680, ly + 52), f"{v}%", 22, MUTED, anchor="mm")
    E(dr, (60, ly + 14), "Average share", 22, MUTED, anchor="lm")
    E(dr, (VW / 2, ly + 96), "White box = the year each brand was at its best.  2010 = Apr-Dec,  2026 = Jan-Oct.", 22, MUTED,
      anchor="mm")


BUMP_SER = ["Nokia", "Samsung", "Apple", "Huawei", "QMobile", "Oppo", "Vivo", "Infinix", "Tecno", "Xiaomi"]
BUMP_T = [None, 2015.9, 2023.9, 2026.9]


def body_bump(dr, s, d, cues):
    x0, x1, y0, y1 = 150, 930, 330, 1330
    X = lambda k: x0 + (x1 - x0) * k / (len(YEARS) - 1)  # noqa
    Y = lambda r: y0 + (y1 - y0) * (r - 1) / 9.0  # noqa
    for r in range(1, 11):
        dr.line([x0, Y(r), x1, Y(r)], fill=GRID, width=1)
        E(dr, (x0 - 30, Y(r)), f"#{r}", 26, MUTED, anchor="rm", bold=(r == 1))
    for k, y in enumerate(YEARS):
        if k % 2 == 0:
            E(dr, (X(k), y1 + 50), y, 24, MUTED, anchor="mm")
    kc = 0.0
    prev = 0.0
    for (a, b), tgt in zip(cues, BUMP_T):
        if tgt is None:
            continue
        if s < a:
            break
        f = (s - a) / max(0.05, (b - a) * 0.85)
        tk = tgt - 2010
        if f < 1.0:
            kc = lerp(prev, tk, smooth(f))
            break
        prev = tk
        kc = tk
    for c in BUMP_SER:
        col = COL[c]
        seg = []
        last = None
        for k, y in enumerate(YEARS):
            if k > kc + 0.001:
                break
            r = YRANK[y][c]
            if r <= 10:
                seg.append((X(k), Y(r)))
                last = (k, r)
            else:
                if len(seg) > 1:
                    dr.line(seg, fill=col, width=9, joint="curve")
                seg = []
        # partial segment to the cursor
        k0 = int(np.floor(kc))
        if k0 + 1 < len(YEARS) and seg and last and last[0] == k0:
            r2 = YRANK[YEARS[k0 + 1]][c]
            if r2 <= 10:
                f = kc - k0
                seg.append((lerp(X(k0), X(k0 + 1), f), lerp(Y(last[1]), Y(r2), f)))
        if len(seg) > 1:
            dr.line(seg, fill=col, width=9, joint="curve")
        for k, y in enumerate(YEARS):
            if k > kc + 0.001:
                break
            r = YRANK[y][c]
            if r <= 10:
                dr.ellipse([X(k) - 12, Y(r) - 12, X(k) + 12, Y(r) + 12], fill=col, outline=BG, width=3)
        if last and last[0] == int(np.floor(kc + 0.001)):
            E(dr, (X(last[0]) + 20, Y(last[1])), c, 24, col, anchor="lm", bold=True)
    # crown markers for the yearly #1
    for k, y in enumerate(YEARS):
        if k > kc + 0.001:
            break
        for c in BUMP_SER:
            if YRANK[y][c] == 1:
                dr.ellipse([X(k) - 18, Y(1) - 18, X(k) + 18, Y(1) + 18], outline=TXT, width=3)


DIV = ["Xiaomi", "Apple", "Infinix", "Tecno", "Realme", "Samsung", "Vivo", "Oppo", "Huawei"]
DIV_BEAT = {"Xiaomi": 1, "Apple": 1, "Infinix": 1, "Tecno": 1, "Realme": 1, "Samsung": 2, "Vivo": 2, "Oppo": 2,
            "Huawei": 2}


def body_div(dr, s, d, cues):
    cx, scale = 680, 54.0
    y0, step = 340, 118
    dr.line([cx, y0 - 40, cx, y0 + step * len(DIV) - 40], fill=MUTED, width=3)
    E(dr, (cx - 14, y0 - 56), "Lost share", 24, RED, anchor="rm", bold=True)
    E(dr, (cx + 14, y0 - 56), "Gained share", 24, GREEN, anchor="lm", bold=True)
    for k, c in enumerate(DIV):
        y = y0 + k * step + 20
        v = val(c, "2026-10") - val(c, "2023-01")
        j = DIV_BEAT[c]
        a = ease((s - cues[j][0] - 0.15 * (k % 4)) / 0.7)
        if a <= 0.01:
            continue
        E(dr, (250, y - 12), c, 34, mix(COL.get(c, TXT), a), anchor="rm", bold=True)
        E(dr, (250, y + 22), f"{val(c, '2023-01'):.1f}% > {val(c, '2026-10'):.1f}%", 22, mix(MUTED, a), anchor="rm")
        w = v * scale * a
        col = GREEN if v >= 0 else RED
        dr.rounded_rectangle([min(cx, cx + w), y - 30, max(cx, cx + w), y + 30], radius=10, fill=mix(col, a))
        txt = f"{v:+.1f}"
        if v >= 0:
            E(dr, (cx + w + 14, y), txt, 34, mix(TXT, a), anchor="lm", bold=True)
        else:
            E(dr, (cx + w - 14, y), txt, 34, mix(TXT, a), anchor="rm", bold=True)
    E(dr, (VW / 2, y0 + step * len(DIV) + 6), "Change in share, Jan 2023 to Oct 2026 (percentage points)", 24, MUTED,
      anchor="mm")


def body_hook(dr, s, d, cues):
    a1 = ease((s - cues[0][0]) / 0.6)
    n1 = 80.75 * ease((s - cues[0][0] - 0.2) / 1.6)
    a2 = ease((s - cues[1][0]) / 0.5)
    n2 = 0.2 * ease((s - cues[1][0] - 0.1) / 1.0)
    E(dr, (VW / 2, 110), "Mobile brands in Pakistan", 62, TXT, anchor="mm", bold=True)
    E(dr, (VW / 2, 180), "Share of mobile web traffic", 34, MUTED, anchor="mm")
    for i, (lab, num, col, a) in enumerate([("Apr 2010", n1, COL["Nokia"], a1), ("Oct 2026", n2, MUTED, a2)]):
        top = 300 + i * 440
        dr.rounded_rectangle([90, top, VW - 90, top + 380], radius=30, fill=mix(PANEL, a))
        E(dr, (VW / 2, top + 64), f"Nokia  |  {lab}", 44, mix(TXT, a), anchor="mm", bold=True)
        dr.text((VW / 2, top + 210), f"{num:.1f}%", font=font("seguisb.ttf", 190), fill=mix(col, a), anchor="mm")
        E(dr, (VW / 2, top + 330), "of all mobile web traffic", 32, mix(MUTED, a), anchor="mm")
    names = ["Bar race", "Line", "Stacked area", "Column", "Heatmap", "Bump", "Gain / loss"]
    a3 = ease((s - cues[2][0]) / 0.8)
    for k, nme in enumerate(names):
        x = 90 + (k % 4) * 225
        y = 1250 + (k // 4) * 84
        aa = max(0.0, min(1.0, a3 * 3 - k * 0.3))
        f = font("seguisb.ttf", 28)
        w = dr.textlength(nme, font=f)
        dr.rounded_rectangle([x, y - 30, x + w + 40, y + 30], radius=30, fill=mix(PANEL, aa), outline=mix(GOLD, aa), width=2)
        dr.text((x + 20, y), nme, font=f, fill=mix(GOLD, aa), anchor="lm")
    E(dr, (VW / 2, 1150), "7 charts, 1 story", 34, mix(TXT, a3), anchor="mm", bold=True)


def body_summary(dr, s, d, cues):
    E(dr, (VW / 2, 110), "The big picture", 62, TXT, anchor="mm", bold=True)
    cards = [
        ("87% > 17%", "Share of the #1 brand\n(Sep 2011 vs Oct 2026)", (34, 211, 238)),
        ("68.5%", "Chinese-company brands\ncombined (Oct 2026)", AMB),
        ("Xiaomi  Apple  Infinix", "Biggest gainers since Jan 2023\n(+5.1, +3.5, +2.4 points)", GREEN),
    ]
    for i, (big, small, col) in enumerate(cards):
        a = ease((s - cues[i][0]) / 0.6)
        top = 260 + i * 330
        dr.rounded_rectangle([70, top, VW - 70, top + 290], radius=28, fill=mix(PANEL, a))
        dr.rounded_rectangle([70, top, 92, top + 290], radius=10, fill=mix(col, a))
        size = 100 if len(big) < 12 else 62
        dr.text((VW / 2 + 20, top + 105), big, font=font("seguisb.ttf", size), fill=mix(col, a), anchor="mm")
        for k, line in enumerate(small.split("\n")):
            E(dr, (VW / 2 + 20, top + 200 + k * 40), line, 30, mix(MUTED, a), anchor="mm")
    a4 = ease((s - cues[3][0]) / 0.6)
    dr.rounded_rectangle([70, 1270, VW - 70, 1360], radius=22, fill=mix(PANEL, a4))
    E(dr, (VW / 2, 1315), "This is share of web traffic, not phone sales.", 30, mix(GOLD, a4), anchor="mm", bold=True)
    a5 = ease((s - cues[3][0] - 0.5 * (cues[3][1] - cues[3][0])) / 0.6)
    E(dr, (VW / 2, 1450), "Which brand do you use?", 56, mix(TXT, a5), anchor="mm", bold=True)
    E(dr, (VW / 2, 1520), "Tell us in the comments", 36, mix(GOLD, a5), anchor="mm")


# ------------------------------------------------------------------ script (beats: Urdu voice-over, easy-English caption)
SCENES = [
    dict(name="hook", tag=None, body=body_hook, hits=[(1, 0.2)], beats=[
        ("دو ہزار دس میں پاکستان کے موبائل انٹرنیٹ کا تقریباً اکیاسی فیصد حصہ نوکیا کا تھا۔",
         "In 2010, Nokia had about 81% of mobile web traffic in Pakistan."),
        ("اور آج؟ صرف صفر اعشاریہ دو فیصد۔",
         "Today, it has just 0.2%."),
        ("یہ کیسے ہوا؟ آئیں، سات گراف میں دیکھتے ہیں۔",
         "How did this happen? Let's see it in 7 charts."),
    ]),
    dict(name="race", tag="CHART 1 OF 7", title="Bar chart race", question="Who is #1, month by month?", body=body_race,
         hits=[(1, 2.0), (2, 2.4), (3, 2.4)], beats=[
        ("پہلا گراف ایک ریس ہے۔ جس برانڈ کا شیئر زیادہ، وہ اوپر۔",
         "Each bar is a brand. A bigger share means a higher place."),
        ("نوکیا بادشاہ ہے۔ ستمبر دو ہزار گیارہ میں اس کا شیئر ستاسی فیصد!",
         "Nokia is the king. It peaks at 87% in September 2011."),
        ("پھر اگست دو ہزار پندرہ میں سام سنگ اسے پیچھے چھوڑ دیتا ہے۔",
         "In August 2015, Samsung passes Nokia."),
        ("پھر چینی برانڈز آتے ہیں، اور دسمبر دو ہزار تئیس میں انفینکس نمبر ون بن جاتا ہے۔",
         "Chinese brands rise. Infinix becomes #1 in December 2023."),
        ("آج سام سنگ اور انفینکس تقریباً برابر ہیں۔",
         "Today, Samsung and Infinix are almost equal."),
    ]),
    dict(name="line", tag="CHART 2 OF 7", title="Line chart", question="How did each brand's share change?", body=body_line,
         hits=[(1, 3.5)], beats=[
        ("دوسرا گراف، لائن چارٹ۔ اس میں ہر برانڈ کا پورا سفر نظر آتا ہے۔",
         "A line chart shows each brand's whole journey."),
        ("نوکیا کی اونچی چوٹی، پھر گراوٹ۔ سام سنگ دو ہزار انیس میں چوٹی پر۔",
         "Nokia: a high peak, then a long fall. Samsung peaks in 2019."),
        ("اور آج بہت سی لکیریں ایک ہی علاقے میں ہیں۔ کوئی بہت آگے نہیں۔",
         "Today, many lines sit close together. No one is far ahead."),
    ]),
    dict(name="area", tag="CHART 3 OF 7", title="Stacked area chart", question="Who owns the whole market?", body=body_area,
         hits=[(2, 0.6)], beats=[
        ("تیسرا گراف۔ پوری مارکیٹ سو فیصد، ہر رنگ ایک گروپ۔",
         "The whole market is 100%. Each color is one group. At first, it is mostly Nokia."),
        ("دو ہزار انیس تک چینی کمپنیوں کے برانڈز تیئیس فیصد پر پہنچ جاتے ہیں۔",
         "By 2019, Chinese-company brands reach 23%."),
        ("اور دو ہزار تئیس میں ستر فیصد!",
         "By 2023, they hold about 70%."),
        ("سام سنگ اور ایپل مل کر آج بھی صرف ایک چوتھائی کے قریب ہیں۔",
         "Samsung and Apple together are still only about 25%."),
    ]),
    dict(name="conc", tag="CHART 4 OF 7", title="Line + column chart", question="Is the market getting more crowded?",
         body=body_conc, hits=[(1, 1.5)], beats=[
        ("کیا اب بھی کسی ایک کا قبضہ ہے؟ نمبر ون کا شیئر ستاسی سے گر کر سترہ فیصد رہ گیا۔",
         "Does one brand still control the market? The #1 share fell from 87% to 17%."),
        ("ٹاپ تین برانڈز کا ملا کر شیئر پچانوے فیصد سے پینتالیس فیصد پر آ گیا۔",
         "The top 3 brands together fell from 95% to 45%."),
        ("دو ہزار گیارہ میں صرف دو برانڈ پانچ فیصد سے اوپر تھے۔ دو ہزار بائیس میں آٹھ۔",
         "In 2011, only 2 brands had 5%+. In 2022, there were 8. One king became an open fight."),
    ]),
    dict(name="heat", tag="CHART 5 OF 7", title="Heatmap", question="When was each brand at its best?", body=body_heat,
         hits=[(2, 2.5)], beats=[
        ("پانچواں گراف، ہیٹ میپ۔ روشن رنگ مطلب بڑا شیئر۔",
         "Each row is a brand. Each column is a year. Brighter = bigger share."),
        ("بائیں طرف نوکیا اور ایپل ہیں۔ بیچ میں سام سنگ، ہواوے اور اوپو۔",
         "Left: Nokia and Apple. Middle: Samsung, Huawei and Oppo."),
        ("اور دائیں طرف ویوو، انفینکس، ٹیکنو اور شاؤمی۔ ہر چوٹی پچھلی کے بعد آتی ہے، جیسے سیڑھی۔",
         "Right: Vivo, Infinix, Tecno, Xiaomi. Each peak comes after the last, like steps."),
    ]),
    dict(name="bump", tag="CHART 6 OF 7", title="Bump chart", question="Who ranked first each year?", body=body_bump,
         hits=[(2, 1.2)], beats=[
        ("چھٹا گراف، بمپ چارٹ۔ یہ صرف رینک دکھاتا ہے۔",
         "This chart shows only rank: who is #1, #2, #3."),
        ("نوکیا دو ہزار دس سے دو ہزار پندرہ تک ہر سال نمبر ون رہا۔",
         "Nokia was #1 every year from 2010 to 2015."),
        ("سام سنگ دو ہزار سولہ سے دو ہزار تئیس تک، پورے آٹھ سال۔",
         "Samsung was #1 for eight years, from 2016 to 2023."),
        ("اور دو ہزار چوبیس سے اب تک انفینکس۔",
         "Since 2024, Infinix is #1."),
    ]),
    dict(name="div", tag="CHART 7 OF 7", title="Gain and loss chart", question="Who is winning right now?", body=body_div,
         hits=[(1, 0.6)], beats=[
        ("آخری گراف۔ دو ہزار تئیس سے اب تک کون جیتا، کون ہارا؟",
         "Who won and who lost since January 2023?"),
        ("شاؤمی تقریباً دگنا ہو گیا، ساڑھے پانچ سے ساڑھے دس فیصد۔ ایپل اور انفینکس بھی بڑھے۔",
         "Xiaomi almost doubled, from 5.6% to 10.6%. Apple, Infinix and Tecno also grew."),
        ("ہارنے والوں میں ہواوے سب سے آگے ہے، پھر اوپو، سام سنگ اور ویوو۔",
         "Huawei lost the most. Then Oppo, Samsung and Vivo."),
    ]),
    dict(name="summary", tag=None, body=body_summary, hits=[(0, 0.3)], beats=[
        ("تو کہانی تین باتوں میں۔ ایک بادشاہ کی جگہ اب کھلا مقابلہ ہے۔",
         "One: a king became an open fight."),
        ("چینی کمپنیوں کے برانڈز کے پاس ساڑھے اڑسٹھ فیصد ٹریفک ہے۔",
         "Two: Chinese-company brands hold 68.5% of traffic."),
        ("اور شاؤمی، ایپل اور انفینکس سب سے تیزی سے بڑھے ہیں۔",
         "Three: Xiaomi, Apple and Infinix grew the most."),
        ("یہ ٹریفک کا شیئر ہے، فروخت کا نہیں۔ آپ کے پاس کون سا برانڈ ہے؟",
         "Remember: this is web traffic, not sales. Which brand do you use? Tell us in the comments."),
    ]),
]


# ------------------------------------------------------------------ audio, frames, mix
def prosody(i, text):
    rnd = random.Random(1000 + i)
    rate = rnd.choice([3, 5, 6, 8, 10])
    pitch = rnd.choice([-2, 0, 1, 2, 3])
    if text.strip().endswith("؟") or "؟" in text:
        pitch += 3
        rate -= 2
    return f"{rate:+d}%", f"{pitch:+d}Hz"


def pause_after(i, last):
    rnd = random.Random(2000 + i)
    return 0.45 if last else rnd.choice([0.22, 0.28, 0.34, 0.4])


def synth_audio(work):
    os.makedirs(work, exist_ok=True)
    wavs = []
    bi = 0
    for si, sc in enumerate(SCENES):
        sc["cues"] = []
        t = 0.0
        for j, (ur_text, en) in enumerate(sc["beats"]):
            rate, pitch = prosody(bi, ur_text)
            key = hashlib.md5(f"{VOICE}|{rate}|{pitch}|{ur_text}".encode("utf-8")).hexdigest()[:12]
            mp3 = f"{work}/{key}.mp3"
            wav = f"{work}/b{bi}.wav"
            if not os.path.exists(mp3):
                asyncio.run(edge_tts.Communicate(ur_text, VOICE, rate=rate, pitch=pitch).save(mp3))
            to_wav(mp3, wav, pause_after(bi, j == len(sc["beats"]) - 1))
            dur = wav_len(wav)
            sc["cues"].append((t, t + dur))
            t += dur
            wavs.append(wav)
            bi += 1
        sc["dur"] = t
    full = f"{work}/full.wav"
    with wave.open(wavs[0]) as w0:
        params = w0.getparams()
    with wave.open(full, "wb") as o:
        o.setparams(params)
        for p in wavs:
            with wave.open(p) as w:
                o.writeframes(w.readframes(w.getnframes()))
    return full


def chrome(dr, sc, j):
    if sc.get("tag"):
        E(dr, (60, 92), sc["tag"], 26, GOLD, anchor="lm", bold=True)
        E(dr, (60, 148), sc["title"], 60, TXT, anchor="lm", bold=True)
        E(dr, (60, 208), sc["question"], 32, MUTED, anchor="lm")
    en = sc["beats"][j][1]
    fnt = font("segoeui.ttf", 40)
    lines = wrap(dr, en, fnt, 900, False)
    h = len(lines) * 58 + 34
    y1 = 1790
    dr.rounded_rectangle([50, y1 - h, VW - 50, y1], radius=20, fill=PANEL)
    for i, ln in enumerate(lines):
        dr.text((VW / 2, y1 - h + 17 + 29 + i * 58), ln, font=fnt, fill=TXT, anchor="mm")
    E(dr, (VW / 2, 1846), "Source: StatCounter Global Stats, Pakistan, mobile vendor share, monthly, Apr 2010 - Oct 2026.", 21,
      MUTED, anchor="mm")
    E(dr, (VW / 2, 1878), "Share of mobile web traffic, not sales. 'Unknown' brands are not ranked.", 21, MUTED, anchor="mm")


def overlay(dr, p):
    dr.rectangle([0, VH - 8, int(VW * p), VH], fill=GOLD)
    dr.rounded_rectangle([28, 24, 214, 62], radius=9, fill=PANEL)
    dr.text((121, 43), "SAMPLE PREVIEW", font=font("arial.ttf", 17), fill=MUTED, anchor="mm")


def render(full, out_video, work):
    total = sum(sc["dur"] for sc in SCENES)
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", full,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", "-shortest", out_video]
    log = open(f"{work}/ffmpeg.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    done = 0.0
    for sc in SCENES:
        n = max(1, round(sc["dur"] * FPS))
        cues = sc["cues"]
        for k in range(n):
            s = k / FPS
            img = Image.new("RGB", (VW, VH), BG)
            dr = ImageDraw.Draw(img)
            sc["body"](dr, s, sc["dur"], cues)
            j = max([i for i, (a, b) in enumerate(cues) if a <= s] or [0])
            chrome(dr, sc, j)
            overlay(dr, (done + s) / total)
            proc.stdin.write(img.tobytes())
        done += sc["dur"]
    proc.stdin.close()
    proc.wait()
    log.close()
    return total


def mix_music(narr_video, out, work, total):
    starts, acc = [], 0.0
    for sc in SCENES:
        starts.append(acc)
        acc += sc["dur"]
    hits = []
    for si, sc in enumerate(SCENES):
        for j, off in sc.get("hits", []):
            hits.append(starts[si] + sc["cues"][j][0] + off)
    hits = sorted(h for h in hits if 3.5 < h < total - 1)
    music = f"{work}/music.wav"
    make_music(total, hits, music)
    flt = ("[0:a]pan=stereo|c0=c0|c1=c0,asplit=2[nar][sc];[1:a]volume=0.6[mus];"
           "[mus][sc]sidechaincompress=threshold=0.02:ratio=9:attack=15:release=500:makeup=1[duck];"
           "[nar][duck]amix=inputs=2:duration=first:normalize=0,pan=stereo|c0=c0|c1=c1[a]")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", narr_video, "-i", music, "-filter_complex", flt, "-map",
                    "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out], check=True)
    return hits


if __name__ == "__main__":
    check()
    work = sys.argv[2]
    full = synth_audio(work)
    total = sum(sc["dur"] for sc in SCENES)
    for sc in SCENES:
        print(sc["name"], round(sc["dur"], 1), [round(b - a, 1) for a, b in sc["cues"]])
    print("TOTAL", round(total, 1))
    if len(sys.argv) > 3 and sys.argv[3] == "--audio-only":
        sys.exit(0)
    out = sys.argv[1]
    narr = f"{work}/narration_only.mp4"
    render(full, narr, work)
    hits = mix_music(narr, out, work, total)
    print("hits", [round(h, 1) for h in hits])
    print("done", out)
