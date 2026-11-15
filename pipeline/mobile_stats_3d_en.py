"""Pakistan mobile brands, 7 chart types in a 3D look, very easy English voice-over and on-screen text.

Same real data as mobile_stats_charts.py (StatCounter Global Stats, Pakistan, mobile vendor share, monthly Apr 2010 - Oct 2026).
Every sentence is a beat: the chart moves exactly while that sentence is spoken. Suspense music is generated from scratch.
"""
import asyncio
import hashlib
import os
import random
import subprocess
import sys
import wave
from functools import lru_cache

import edge_tts
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mobile_stats_charts as M  # noqa  (data, analysis, cursor helpers)
from common import FFMPEG, FPS, ease, lerp, to_wav, wav_len, wrap  # noqa
from music_suspense import make_music  # noqa

VW, VH = 1080, 1920
BGC = (10, 14, 38)
TXT = (245, 248, 255)
MUTED = (160, 172, 205)
GOLD = (255, 196, 20)
GREEN = (52, 211, 130)
RED = (250, 92, 92)
VOICE = "en-US-AndrewMultilingualNeural"
FONT_DIR = "C:/Windows/Fonts/"
N, MI, S, MONTHS = M.N, M.MI, M.S, M.MONTHS
cursor, smooth, interp = M.cursor, M.smooth, M.interp

CANVAS = None  # current frame image (set by the render loop)

# ------------------------------------------------------------------ fonts: catchy but readable, all installed on Windows
_FONTS = {"title": ("brlnsdb.ttf", None), "num": ("impact.ttf", None), "lab": ("bahnschrift.ttf", b"SemiBold"),
          "labb": ("bahnschrift.ttf", b"Bold"), "cap": ("gothicb.ttf", None), "small": ("bahnschrift.ttf", b"Regular")}
_FALLBACK = {"title": ("bahnschrift.ttf", b"Bold"), "cap": ("seguisb.ttf", None)}
_fc = {}


def F(kind, size):
    key = (kind, size)
    if key not in _fc:
        fn, var = _FONTS[kind]
        if not os.path.exists(FONT_DIR + fn):  # e.g. GitHub Windows runners lack the Office fonts
            fn, var = _FALLBACK.get(kind, ("arialbd.ttf", None))
        f = ImageFont.truetype(FONT_DIR + fn, size)
        if var:
            f.set_variation_by_name(var)
        _fc[key] = f
    return _fc[key]


def mix(c, a):
    return tuple(int(lerp(BGC[i], c[i], max(0.0, min(1.0, a)))) for i in range(3))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def tint(c, f):
    return tuple(int(v + (255 - v) * f) for v in c)


def TX(dr, xy, text, kind, size, fill, anchor="lm", shadow=True, stroke=0, stroke_fill=(8, 10, 28)):
    f = F(kind, size)
    k = min(1.0, sum(abs(fill[i] - BGC[i]) for i in range(3)) / 200.0)
    if k < 0.04:
        return
    if shadow:
        dr.text((xy[0] + 3, xy[1] + 4), text, font=f, fill=mix((3, 5, 16), k), anchor=anchor, stroke_width=stroke,
                stroke_fill=mix((3, 5, 16), k))
    dr.text(xy, text, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)


# ------------------------------------------------------------------ 3D building blocks
def make_bg():
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    t = yy / VH
    top, bot = np.array([26, 20, 70], np.float32), np.array([8, 12, 32], np.float32)
    img = top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None]
    for (cx, cy, rad, col, k) in [(160, 260, 760, (110, 70, 220), 0.42), (960, 1560, 900, (20, 150, 180), 0.28),
                                  (900, 380, 520, (230, 80, 140), 0.16)]:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
        g = np.clip(1 - d, 0, 1) ** 2 * k
        img = img * (1 - g[..., None]) + np.array(col, np.float32)[None, None, :] * g[..., None]
    dots = ((xx % 44 < 2) & (yy % 44 < 2)).astype(np.float32) * 14
    img = img + dots[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


BGIMG = make_bg()


@lru_cache(maxsize=None)
def sphere_sprite(c, r):
    ss = 3
    n = 2 * r * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = (n - 1) / 2
    nx, ny = (xx - cx) / (r * ss), (yy - cx) / (r * ss)
    rr = nx * nx + ny * ny
    mask = rr <= 1.0
    nz = np.sqrt(np.clip(1 - rr, 0, 1))
    light = np.array([-0.45, -0.55, 0.70], np.float32)
    light /= np.linalg.norm(light)
    diff = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    spec = diff ** 26
    rim = np.clip(1 - nz, 0, 1) ** 2
    col = np.array(c, np.float32)[None, None, :]
    rgb = col * (0.28 + 0.85 * diff[..., None]) + 255 * spec[..., None] * 0.6 - 45 * rim[..., None]
    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (mask * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA").resize((2 * r, 2 * r), Image.LANCZOS)


def sphere(cx, cy, r, c, a=1.0, shadow=True):
    if a <= 0.02:
        return
    sp = sphere_sprite(tuple(c), int(r))
    if shadow:
        ImageDraw.Draw(CANVAS).ellipse([cx - r * 0.8 + 8, cy + r * 0.55, cx + r * 0.8 + 8, cy + r * 1.0 + 8], fill=(3, 5, 16))
    if a < 0.98:
        sp = sp.copy()
        sp.putalpha(sp.getchannel("A").point(lambda v: int(v * a)))
    CANVAS.paste(sp, (int(cx - r), int(cy - r)), sp)


def card(dr, x0, y0, x1, y1, fill=(28, 34, 84), depth=14, radius=28, edge=(92, 108, 190), a=1.0):
    if a < 0.03:
        return
    dr.rounded_rectangle([x0 + 6, y0 + depth, x1 + 6, y1 + depth], radius=radius, fill=mix((4, 6, 20), a))
    dr.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=mix(fill, a), outline=mix(edge, a), width=3)
    dr.line([x0 + radius, y0 + 5, x1 - radius, y0 + 5], fill=mix(tint(fill, 0.35), a), width=3)


def bar3d_h(dr, x0, y0, x1, y1, c, d=16, a=1.0):
    """Horizontal 3D bar: front face + top face + end face + drop shadow."""
    if x1 < x0 + 2:
        x1 = x0 + 2
    sh = mix((3, 5, 14), a)
    dr.polygon([(x0 + 10, y1 + 14), (x0 + d + 10, y1 - d + 14), (x1 + d + 10, y1 - d + 14), (x1 + 10, y1 + 14)], fill=sh)
    dr.polygon([(x0, y0), (x1, y0), (x1 + d, y0 - d), (x0 + d, y0 - d)], fill=mix(tint(c, 0.42), a))
    dr.polygon([(x1, y0), (x1 + d, y0 - d), (x1 + d, y1 - d), (x1, y1)], fill=mix(shade(c, 0.52), a))
    nb = 6
    for i in range(nb):
        f = 1.18 - 0.42 * i / (nb - 1)
        ya, yb = y0 + (y1 - y0) * i / nb, y0 + (y1 - y0) * (i + 1) / nb
        dr.rectangle([x0, ya, x1, yb + 1], fill=mix(shade(c, f), a))
    dr.rectangle([x0, y0 + 3, x1, y0 + 8], fill=mix(tint(c, 0.55), a))


def bar3d_v(dr, x0, y0, x1, y1, c, d=12, a=1.0):
    """Vertical 3D column from baseline y1 up to y0."""
    if y1 < y0 + 2:
        y0 = y1 - 2
    dr.polygon([(x0 + 8, y1 + 8), (x1 + 8 + d, y1 + 8), (x1 + d + 8, y1 - d + 8), (x0 + d + 8, y1 - d + 8)], fill=mix((3, 5, 14), a))
    dr.polygon([(x0, y0), (x1, y0), (x1 + d, y0 - d), (x0 + d, y0 - d)], fill=mix(tint(c, 0.45), a))
    dr.polygon([(x1, y0), (x1 + d, y0 - d), (x1 + d, y1 - d), (x1, y1)], fill=mix(shade(c, 0.5), a))
    nb = 5
    for i in range(nb):
        f = 1.15 - 0.4 * i / (nb - 1)
        xa, xb = x0 + (x1 - x0) * i / nb, x0 + (x1 - x0) * (i + 1) / nb
        dr.rectangle([xa, y0, xb + 1, y1], fill=mix(shade(c, f), a))


def tube(dr, pts, c, w=9, off=(7, 9), a=1.0):
    """A line that looks like a 3D tube: dark extrusion behind, bright body, thin highlight."""
    if len(pts) < 2:
        return
    dr.line([(x + off[0], y + off[1]) for x, y in pts], fill=mix(shade(c, 0.32), a), width=w + 2, joint="curve")
    dr.line(pts, fill=mix(c, a), width=w, joint="curve")
    dr.line([(x - 1, y - 2) for x, y in pts], fill=mix(tint(c, 0.55), a), width=max(2, w // 3), joint="curve")


def pill(dr, cx, cy, text, a=1.0, size=34, color=GOLD, kind="labb"):
    if a < 0.03:
        return
    f = F(kind, size)
    w = dr.textlength(text, font=f)
    card(dr, cx - w / 2 - 30, cy - 36, cx + w / 2 + 30, cy + 36, fill=(40, 34, 90), depth=8, radius=36, edge=color, a=a)
    dr.text((cx, cy), text, font=f, fill=mix(color, a), anchor="mm")


def tag_pill(dr, x, y, text, a=1.0, fill=(40, 34, 90), color=GOLD, size=26, anchor="lm"):
    if a < 0.03:
        return
    f = F("labb", size)
    w = dr.textlength(text, font=f)
    x0 = x if anchor == "lm" else (x - w if anchor == "rm" else x - w / 2)
    card(dr, x0 - 12, y - 22, x0 + w + 12, y + 22, fill=fill, depth=6, radius=14, edge=color, a=a)
    dr.text((x0, y), text, font=f, fill=mix(color, a), anchor="lm")


def spread(items, min_gap, hi):
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    if ys and ys[-1] > hi:
        over = ys[-1] - hi
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
        dr.text((x, y_bottom + 34), str(y), font=F("lab", 26), fill=MUTED, anchor="mm")


def grid_lines(dr, x0, x1, Y, ticks, fmt="{}%"):
    for v in ticks:
        dr.line([x0, Y(v), x1, Y(v)], fill=(58, 70, 126), width=2)
        dr.text((x0 - 14, Y(v)), fmt.format(v), font=F("lab", 26), fill=MUTED, anchor="rm")


# ------------------------------------------------------------------ bodies
def draw_race(dr, m):
    vals = {b: interp(S[b], m) for b in M.RACE_BRANDS}
    yi = MONTHS[int(round(m))]
    dr.text((800, 1480), yi[:4], font=F("num", 170), fill=(44, 56, 112), anchor="mm")
    dr.text((800, 1590), M.MNAME[int(yi[5:7]) - 1], font=F("title", 56), fill=(60, 74, 140), anchor="mm")
    axis = max(max(vals.values()), 8.0)
    x0, maxlen, pitch, ytop, sg = 300, 500, 112, 392, 0.025
    order = []
    br = M.RACE_BRANDS
    for b in br:
        rk = sum(1 / (1 + np.exp(-np.clip((vals[o] - vals[b] + 0.004 * (br.index(o) - br.index(b))) / sg, -30, 30)))
                 for o in br if o != b)
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
        c = M.COL.get(b, MUTED)
        x1 = x0 + v / axis * maxlen
        TX(dr, (x0 - 40, y), b, "lab", 38, mix(TXT, a), anchor="rm", shadow=False)
        bar3d_h(dr, x0, y - 26, max(x0 + 10, x1), y + 26, c, d=16, a=a)
        sphere(x1, y, 36, c, a)
        dr.text((x1, y), b[0], font=F("title", 36), fill=mix((255, 255, 255), a), anchor="mm", stroke_width=2,
                stroke_fill=mix(shade(c, 0.3), a))
        TX(dr, (x1 + 54, y), f"{v:.1f}%", "num", 46, mix(TXT, a), anchor="lm")


RACE_CALL = [None, "Sep 2011: Nokia reaches 87%", "Aug 2015: Samsung passes Nokia", "Dec 2023: Infinix is #1",
             "Oct 2026: Samsung 17%, Infinix 16%"]


def body_race(dr, s, d, cues):
    m = cursor(cues, M.RACE_T, s, 0.0, hold=0.85)
    draw_race(dr, m)
    shown = None
    for j, c in enumerate(RACE_CALL):
        if c and s >= cues[j][0] + 0.7 * (cues[j][1] - cues[j][0]):
            shown = (j, c)
    if shown:
        j, c = shown
        a = ease((s - (cues[j][0] + 0.7 * (cues[j][1] - cues[j][0]))) / 0.4)
        pill(dr, VW / 2, 292, c, a, size=34)


LINE_SER = ["Nokia", "Samsung", "Huawei", "Oppo", "Vivo", "Infinix", "Apple"]
LINE_T = [None, MI["2021-01"], N - 1]
LINE_ANN = [("2011-09", "Nokia", "Nokia peak: 87%", "r", 0), ("2015-08", "Samsung", "Samsung passes Nokia", "l", -6),
            ("2019-08", "Samsung", "Samsung peak: 41%", "t", 0), ("2020-04", "Huawei", "Huawei peak: 17%", "r", -30),
            ("2023-12", "Infinix", "Infinix #1 (Dec 2023)", "b", 0)]


def body_line(dr, s, d, cues):
    x0, x1, y0, y1, ymax = 140, 770, 330, 1330, 90.0
    card(dr, 60, 270, 1020, 1400, fill=(22, 28, 70), depth=16, radius=30)
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / ymax  # noqa
    grid_lines(dr, x0, x1, Y, (0, 20, 40, 60, 80))
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
        tube(dr, pts, M.COL[b], w=(6 if b == "Apple" else 9))
        heads.append((hy, b, hx))
    for hy, b, hx in heads:
        sphere(hx, hy, 15, M.COL[b], 1.0, shadow=False)
    ly = spread([(y, b) for y, b, _ in heads if interp(S[b], m) >= 0.8], 36, y1)
    for b, y in ly.items():
        TX(dr, (X(m) + 26, y), f"{b} {interp(S[b], m):.0f}%", "lab", 28, M.COL[b], anchor="lm")
    for d_, b, text, side, dy in LINE_ANN:
        i = MI[d_]
        if m >= i + 1:
            a = ease((m - i - 1) / 5)
            px, py = X(i), Y(S[b][i])
            sphere(px, py, 12, (255, 255, 255), a, shadow=False)
            if side == "r":
                tag_pill(dr, px + 24, py + dy, text, a)
            elif side == "l":
                tag_pill(dr, px - 24, py + dy, text, a, anchor="rm")
            elif side == "t":
                tag_pill(dr, px, py - 48, text, a, anchor="mm")
            else:
                tag_pill(dr, px, py + 54, text, a, anchor="mm")


LAYERS = [("Nokia", S["Nokia"], M.COL["Nokia"]), ("Samsung", S["Samsung"], M.COL["Samsung"]),
          ("Apple", S["Apple"], M.COL["Apple"]), ("Chinese brands", M.CH, M.AMB), ("Other brands", M.OTH, M.SLATE),
          ("Unknown", S["Unknown"], (74, 84, 118))]
AREA_T = [MI["2011-06"], MI["2019-01"], MI["2023-01"], N - 1]


def body_area(dr, s, d, cues):
    x0, x1, y0, y1 = 110, 930, 440, 1260
    dx, dy = 34, 30
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / 100.0  # noqa
    m = cursor(cues, AREA_T, s, 0.0)
    card(dr, 50, 282, 1030, 1530, fill=(20, 26, 66), depth=16, radius=30)
    for v in (0, 25, 50, 75, 100):
        dr.text((x0 - 14, Y(v)), f"{v}%", font=F("lab", 26), fill=MUTED, anchor="rm")
    if m > 0.05:
        i_end = int(np.floor(m))
        xs = [X(i) for i in range(0, i_end + 1)] + [X(m)]
        xc = xs[-1]
        # 3D slab: top face, right end face (striped by layer), then the front face
        dr.polygon([(x0, y0), (xc, y0), (xc + dx, y0 - dy), (x0 + dx, y0 - dy)], fill=(96, 108, 150))
        base = 0.0
        for name, arr, c in LAYERS:
            v = interp(arr, m)
            dr.polygon([(xc, Y(base + v)), (xc + dx, Y(base + v) - dy), (xc + dx, Y(base) - dy), (xc, Y(base))],
                       fill=shade(c, 0.55))
            base += v
        base_arr = np.zeros(i_end + 2)
        for name, arr, c in LAYERS:
            part = [arr[i] for i in range(0, i_end + 1)] + [interp(arr, m)]
            top = base_arr + np.array(part)
            poly = [(x, Y(v)) for x, v in zip(xs, top)] + [(x, Y(v)) for x, v in zip(xs[::-1], base_arr[::-1])]
            dr.polygon(poly, fill=c)
            dr.line([(x, Y(v)) for x, v in zip(xs, top)], fill=tint(c, 0.45), width=3)
            base_arr = top
        dr.line([xc, y0, xc, y1], fill=(255, 255, 255), width=3)
    dr.rectangle([x0, y0, x1, y1], outline=(70, 84, 150), width=2)
    year_ticks(dr, X, y1)
    if m > 0.05:
        yi = MONTHS[int(round(m))]
        pill(dr, min(max(X(m), 230), 840), y0 - 100, f"{M.MNAME[int(yi[5:7]) - 1]} {yi[:4]}", 1.0, size=30)
    for k, (name, arr, c) in enumerate(LAYERS):
        cx0 = 80 + (k % 2) * 480
        cy0 = 1366 + (k // 2) * 56
        sphere(cx0 + 14, cy0, 15, c, 1.0, shadow=False)
        TX(dr, (cx0 + 42, cy0), f"{name}  {interp(arr, max(m, 0.0)):.0f}%", "lab", 30, TXT, anchor="lm", shadow=False)


def body_conc(dr, s, d, cues):
    x0, x1, y0, y1 = 140, 960, 340, 840
    X = lambda m: x0 + (x1 - x0) * m / (N - 1)  # noqa
    Y = lambda v: y1 - (y1 - y0) * v / 100.0  # noqa
    card(dr, 50, 262, 1030, 880, fill=(22, 28, 70), depth=14, radius=28)
    grid_lines(dr, x0, x1, Y, (0, 25, 50, 75, 100))
    year_ticks(dr, X, y1)
    t1, t2 = cues[0], cues[1]
    m1 = 0 if s < t1[0] else (N - 1) * smooth((s - t1[0]) / ((t1[1] - t1[0]) * 0.9))
    m2 = 0 if s < t2[0] else (N - 1) * smooth((s - t2[0]) / ((t2[1] - t2[0]) * 0.9))
    for arr, m, c, name, tx in [(M.TOP3, m2, (251, 191, 36), "Top 3 brands", 540), (M.TOP1, m1, (34, 211, 238), "#1 brand", 140)]:
        if m > 0.05:
            pts = [(X(i), Y(arr[i])) for i in range(0, int(np.floor(m)) + 1)] + [(X(m), Y(interp(arr, m)))]
            tube(dr, pts, c, w=9)
            sphere(pts[-1][0], pts[-1][1], 15, c, 1.0, shadow=False)
            TX(dr, (tx, 304), f"{name}: {interp(arr, m):.0f}%", "labb", 30, c, anchor="lm")
    card(dr, 50, 920, 1030, 1500, fill=(22, 28, 70), depth=14, radius=28)
    TX(dr, (90, 975), "Brands with 5% or more (each year)", "labb", 32, TXT, anchor="lm")
    cy0, cy1, cx0, cx1 = 1060, 1420, 140, 990
    cw = (cx1 - cx0) / len(M.YEARS)
    for v in (0, 2, 4, 6, 8):
        yy = cy1 - (cy1 - cy0) * v / 9.0
        dr.line([cx0, yy, cx1, yy], fill=(58, 70, 126), width=1)
        dr.text((cx0 - 14, yy), str(v), font=F("lab", 24), fill=MUTED, anchor="rm")
    ca = M.beat_alpha(cues, 2, s, 0.0, 2.5)
    for k, y in enumerate(M.YEARS):
        a = max(0.0, min(1.0, ca * len(M.YEARS) - k * 0.55))
        if a <= 0.01:
            continue
        h = (cy1 - cy0) * M.N5[y] / 9.0 * ease(a)
        xx = cx0 + k * cw
        bar3d_v(dr, xx + 5, cy1 - h, xx + cw - 12, cy1, GREEN if M.N5[y] >= 6 else (80, 140, 255), d=10, a=a)
        if a > 0.5:
            TX(dr, (xx + cw / 2 + 2, cy1 - h - 24), str(M.N5[y]), "num", 26, mix(TXT, a), anchor="mm")
    for k, y in enumerate(M.YEARS):
        dr.text((cx0 + k * cw + cw / 2, cy1 + 28), "'" + y[2:], font=F("lab", 22), fill=MUTED, anchor="mm")


HEAT_ROWS = M.HEAT_ROWS
HEAT_T = [None, 2021.5, 2026.9]


def body_heat(dr, s, d, cues):
    gx0, gy0, cw, ch = 262, 340, 44, 88
    card(dr, 40, 262, 1040, 1500, fill=(20, 26, 66), depth=14, radius=28)
    ycur, prev = 2008.9, 2008.9
    for (a, b), tgt in zip(cues, HEAT_T):
        if tgt is None:
            continue
        if s < a:
            break
        f = (s - a) / max(0.05, (b - a) * 0.85)
        if f < 1.0:
            ycur = lerp(prev, tgt, smooth(f))
            break
        prev = ycur = tgt
    for k, y in enumerate(M.YEARS):
        dr.text((gx0 + k * cw + cw / 2, gy0 - 24), "'" + y[2:], font=F("lab", 22), fill=MUTED, anchor="mm")
    for r, c in enumerate(HEAT_ROWS):
        cy = gy0 + r * ch
        TX(dr, (gx0 - 14, cy + ch / 2), c, "labb", 30, M.COL.get(c, TXT), anchor="rm", shadow=False)
        for k, y in enumerate(M.YEARS):
            a = max(0.0, min(1.0, ycur - (2010 + k) + 1.2))
            if a <= 0:
                continue
            v = M.YA[y][c]
            col = M.heat_color(v)
            e = 3 + 18 * min(1.0, v / 40.0)
            x, yt, w, h = gx0 + k * cw + 2, cy + 4, cw - 8, ch - 10
            ex = e * 0.6
            dr.polygon([(x, yt), (x + w, yt), (x + w + ex, yt - e), (x + ex, yt - e)], fill=mix(tint(col, 0.4), a))
            dr.polygon([(x + w, yt), (x + w + ex, yt - e), (x + w + ex, yt + h - e), (x + w, yt + h)], fill=mix(shade(col, 0.5), a))
            dr.rectangle([x, yt, x + w, yt + h], fill=mix(col, a))
            if v >= 5 and a > 0.6:
                lum = sum(col) / 3
                dr.text((x + w / 2, yt + h / 2), f"{v:.0f}", font=F("num", 24), fill=(18, 22, 34) if lum > 120 else TXT,
                        anchor="mm")
    if ycur >= 2026.5:
        pa = M.beat_alpha(cues, 2, s, 1.2, 0.8)
        for r, c in enumerate(HEAT_ROWS):
            k = M.YEARS.index(M.HEAT_PEAK[c])
            x, yt = gx0 + k * cw + 2, gy0 + r * ch + 4
            dr.rectangle([x - 2, yt - 2, x + cw - 6, yt + ch - 6], outline=mix((255, 255, 255), pa), width=4)
    ly = gy0 + len(HEAT_ROWS) * ch + 36
    for k in range(200):
        dr.rectangle([gx0 + k * 3.4, ly, gx0 + k * 3.4 + 4, ly + 28], fill=M.heat_color(84.0 * k / 199))
    for v in (0, 20, 40, 60, 80):
        dr.text((gx0 + v / 84.0 * 680, ly + 52), f"{v}%", font=F("lab", 22), fill=MUTED, anchor="mm")
    dr.text((VW / 2, ly + 92), "Brighter and taller = bigger share.  White box = best year.", font=F("small", 24),
            fill=MUTED, anchor="mm")


BUMP_SER = ["Nokia", "Samsung", "Apple", "Huawei", "QMobile", "Oppo", "Vivo", "Infinix", "Tecno", "Xiaomi"]
BUMP_T = [None, 2015.9, 2023.9, 2026.9]


def body_bump(dr, s, d, cues):
    x0, x1, y0, y1 = 170, 900, 350, 1330
    card(dr, 50, 262, 1030, 1470, fill=(20, 26, 66), depth=14, radius=28)
    X = lambda k: x0 + (x1 - x0) * k / (len(M.YEARS) - 1)  # noqa
    Y = lambda r: y0 + (y1 - y0) * (r - 1) / 9.0  # noqa
    for r in range(1, 11):
        dr.line([x0, Y(r), x1, Y(r)], fill=(50, 62, 116), width=1)
        dr.text((x0 - 28, Y(r)), f"#{r}", font=F("num" if r == 1 else "lab", 30 if r == 1 else 26),
                fill=GOLD if r == 1 else MUTED, anchor="rm")
    for k, y in enumerate(M.YEARS):
        if k % 2 == 0:
            dr.text((X(k), y1 + 50), y, font=F("lab", 26), fill=MUTED, anchor="mm")
    kc, prev = 0.0, 0.0
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
        prev = kc = tk
    for c in BUMP_SER:
        col = M.COL[c]
        seg, last = [], None
        for k, y in enumerate(M.YEARS):
            if k > kc + 0.001:
                break
            r = M.YRANK[y][c]
            if r <= 10:
                seg.append((X(k), Y(r)))
                last = (k, r)
            else:
                tube(dr, seg, col, w=10, off=(6, 8))
                seg = []
        k0 = int(np.floor(kc))
        if k0 + 1 < len(M.YEARS) and seg and last and last[0] == k0:
            r2 = M.YRANK[M.YEARS[k0 + 1]][c]
            if r2 <= 10:
                f = kc - k0
                seg.append((lerp(X(k0), X(k0 + 1), f), lerp(Y(last[1]), Y(r2), f)))
        tube(dr, seg, col, w=10, off=(6, 8))
    for c in BUMP_SER:
        col = M.COL[c]
        for k, y in enumerate(M.YEARS):
            if k > kc + 0.001:
                break
            r = M.YRANK[y][c]
            if r <= 10:
                if r == 1:
                    sphere(X(k), Y(r), 21, GOLD, 1.0, shadow=False)
                    dr.text((X(k), Y(r)), "1", font=F("num", 24), fill=(40, 24, 0), anchor="mm")
                else:
                    sphere(X(k), Y(r), 13, col, 1.0, shadow=False)
        kk = int(np.floor(kc + 0.001))
        r = M.YRANK[M.YEARS[min(kk, len(M.YEARS) - 1)]][c]
        if r <= 10 and kk < len(M.YEARS):
            TX(dr, (X(kk) + 28, Y(r)), c, "labb", 26, col, anchor="lm")


DIV = M.DIV
DIV_BEAT = M.DIV_BEAT


def body_div(dr, s, d, cues):
    cx, scale = 660, 44.0
    y0, step = 360, 118
    card(dr, 40, 262, 1040, 1520, fill=(20, 26, 66), depth=14, radius=28)
    dr.line([cx, y0 - 40, cx, y0 + step * len(DIV) - 40], fill=MUTED, width=3)
    TX(dr, (cx - 16, y0 - 64), "LOST", "num", 30, RED, anchor="rm")
    TX(dr, (cx + 16, y0 - 64), "GAINED", "num", 30, GREEN, anchor="lm")
    for k, c in enumerate(DIV):
        y = y0 + k * step + 20
        v = M.val(c, "2026-10") - M.val(c, "2023-01")
        a = ease((s - cues[DIV_BEAT[c]][0] - 0.15 * (k % 4)) / 0.7)
        if a <= 0.01:
            continue
        TX(dr, (250, y - 12), c, "labb", 36, mix(M.COL.get(c, TXT), a), anchor="rm", shadow=False)
        dr.text((250, y + 24), f"{M.val(c, '2023-01'):.1f}% to {M.val(c, '2026-10'):.1f}%", font=F("small", 24),
                fill=mix(MUTED, a), anchor="rm")
        w = v * scale * a
        col = GREEN if v >= 0 else RED
        bar3d_h(dr, min(cx, cx + w), y - 28, max(cx, cx + w), y + 28, col, d=15, a=a)
        if v >= 0:
            TX(dr, (cx + w + 32, y), f"+{v:.1f}", "num", 40, mix(TXT, a), anchor="lm")
        else:
            TX(dr, (cx + w - 14, y), f"{v:.1f}", "num", 40, mix(TXT, a), anchor="rm")
    dr.text((VW / 2, y0 + step * len(DIV) + 4), "Change in share, Jan 2023 to Oct 2026 (% points)", font=F("small", 26),
            fill=MUTED, anchor="mm")


def body_hook(dr, s, d, cues):
    a1 = ease((s - cues[0][0]) / 0.6)
    n1 = 80.75 * ease((s - cues[0][0] - 0.2) / 1.6)
    a2 = ease((s - cues[1][0]) / 0.5)
    n2 = 0.2 * ease((s - cues[1][0] - 0.1) / 1.0)
    TX(dr, (VW / 2, 120), "MOBILE BRANDS", "title", 78, TXT, anchor="mm", stroke=2)
    TX(dr, (VW / 2, 200), "in Pakistan", "title", 50, GOLD, anchor="mm")
    for i, (lab, num, col, a) in enumerate([("April 2010", n1, M.COL["Nokia"], a1), ("October 2026", n2, (170, 180, 210), a2)]):
        top = 290 + i * 450
        card(dr, 90, top, VW - 90, top + 390, fill=(30, 36, 92), depth=18, radius=34, a=a)
        TX(dr, (VW / 2, top + 62), f"Nokia  |  {lab}", "title", 46, mix(TXT, a), anchor="mm", shadow=False)
        TX(dr, (VW / 2, top + 215), f"{num:.1f}%", "num", 230, mix(col, a), anchor="mm", stroke=3)
        TX(dr, (VW / 2, top + 340), "of mobile internet use", "lab", 34, mix(MUTED, a), anchor="mm", shadow=False)
    names = ["Bar race", "Line", "Area", "Columns", "Heatmap", "Rank", "Winners"]
    a3 = ease((s - cues[2][0]) / 0.8)
    TX(dr, (VW / 2, 1196), "7 charts, 1 story", "title", 44, mix(TXT, a3), anchor="mm")
    for k, nme in enumerate(names):
        x = 80 + (k % 4) * 235
        y = 1270 + (k // 4) * 92
        aa = max(0.0, min(1.0, a3 * 3 - k * 0.3))
        tag_pill(dr, x, y, nme, aa, size=30)


def body_summary(dr, s, d, cues):
    TX(dr, (VW / 2, 120), "THE BIG PICTURE", "title", 70, TXT, anchor="mm", stroke=2)
    cards = [("87% to 17%", "Share of the #1 brand\n(Sep 2011 vs Oct 2026)", (34, 211, 238)),
             ("68.5%", "Chinese brands together\n(Oct 2026)", M.AMB),
             ("Xiaomi  Apple  Infinix", "Grew the most since Jan 2023\n(+5.1, +3.5, +2.4 points)", GREEN)]
    for i, (big, small, col) in enumerate(cards):
        a = ease((s - cues[i][0]) / 0.6)
        top = 230 + i * 330
        if a < 0.03:
            continue
        card(dr, 70, top, VW - 70, top + 290, fill=(30, 36, 92), depth=16, radius=30, a=a)
        dr.rounded_rectangle([70, top, 100, top + 290], radius=14, fill=mix(col, a))
        TX(dr, (VW / 2 + 24, top + 105), big, "num", 110 if len(big) < 12 else 66, mix(col, a), anchor="mm", stroke=2)
        for k, line in enumerate(small.split("\n")):
            dr.text((VW / 2 + 24, top + 200 + k * 40), line, font=F("lab", 30), fill=mix(MUTED, a), anchor="mm")
    a4 = ease((s - cues[3][0]) / 0.6)
    card(dr, 70, 1250, VW - 70, 1350, fill=(60, 40, 20), depth=10, radius=24, edge=GOLD, a=a4)
    if a4 > 0.03:
        dr.text((VW / 2, 1300), "This is internet use, not phone sales.", font=F("labb", 34), fill=mix(GOLD, a4), anchor="mm")
    a5 = ease((s - cues[3][0] - 0.5 * (cues[3][1] - cues[3][0])) / 0.6)
    TX(dr, (VW / 2, 1450), "Which brand do you use?", "title", 62, mix(TXT, a5), anchor="mm", stroke=2)
    TX(dr, (VW / 2, 1530), "Tell us in the comments", "lab", 40, mix(GOLD, a5), anchor="mm")


# ------------------------------------------------------------------ script: very easy English, one idea per sentence
SCENES = [
    dict(name="hook", tag=None, body=body_hook, hits=[(1, 0.2)], beats=[
        "In 2010, Nokia phones made up 81% of mobile internet use in Pakistan.",
        "Today? Just 0.2%.",
        "What happened? Let's see it in 7 simple charts.",
    ]),
    dict(name="race", tag="CHART 1 OF 7", title="Bar Race", question="Who is number 1?", body=body_race,
         hits=[(1, 2.0), (2, 2.4), (3, 2.4)], beats=[
        "Chart one is a race. A longer bar means a bigger share.",
        "Nokia is the king. In September 2011, it reaches 87%.",
        "In August 2015, Samsung passes Nokia.",
        "Then Chinese brands rise. In December 2023, Infinix becomes number one.",
        "Today, Samsung and Infinix are almost equal.",
    ]),
    dict(name="line", tag="CHART 2 OF 7", title="Line Chart", question="How did each brand change?", body=body_line,
         hits=[(1, 3.5)], beats=[
        "Chart two is a line chart. It shows each brand's full journey.",
        "Nokia goes very high, then falls. Samsung peaks in 2019.",
        "Today, many lines are close together. No brand is far ahead.",
    ]),
    dict(name="area", tag="CHART 3 OF 7", title="Area Chart", question="Who owns the whole market?", body=body_area,
         hits=[(2, 0.6)], beats=[
        "Chart three. The whole market is 100%. Each color is one group.",
        "By 2019, Chinese brands have 23%.",
        "By 2023, they have 70%!",
        "Samsung and Apple together have only about one quarter.",
    ]),
    dict(name="conc", tag="CHART 4 OF 7", title="Crowd Chart", question="Is the market more crowded now?",
         body=body_conc, hits=[(1, 1.5)], beats=[
        "Chart four. The number one brand fell from 87% to 17%.",
        "The top three brands together fell from 95% to 45%.",
        "In 2011, only 2 brands had 5% or more. In 2022, eight brands did.",
    ]),
    dict(name="heat", tag="CHART 5 OF 7", title="Heatmap", question="When was each brand at its best?", body=body_heat,
         hits=[(2, 2.5)], beats=[
        "Chart five is a heatmap. Brighter color means bigger share.",
        "On the left, Nokia and Apple. In the middle, Samsung, Huawei and Oppo.",
        "On the right, Vivo, Infinix, Tecno and Xiaomi. Each one peaks after the one before.",
    ]),
    dict(name="bump", tag="CHART 6 OF 7", title="Rank Chart", question="Who was number 1 each year?", body=body_bump,
         hits=[(2, 1.2)], beats=[
        "Chart six shows rank only. Who is first, second, third?",
        "Nokia was number one from 2010 to 2015.",
        "Samsung was number one for eight years, from 2016 to 2023.",
        "Since 2024, it is Infinix.",
    ]),
    dict(name="div", tag="CHART 7 OF 7", title="Winners and Losers", question="Who gained since 2023?", body=body_div,
         hits=[(1, 0.6)], beats=[
        "Chart seven. Who won and who lost since January 2023?",
        "Xiaomi almost doubled, from 5.6% to 10.6%. Apple and Infinix also grew.",
        "Huawei lost the most. Then Oppo, Samsung and Vivo.",
    ]),
    dict(name="summary", tag=None, body=body_summary, hits=[(0, 0.3)], beats=[
        "So, here is the story. One king became many players.",
        "Chinese brands now have 68.5% of mobile internet use.",
        "And Xiaomi, Apple and Infinix grew the most.",
        "Remember, this is internet use, not phone sales. Which brand do you use?",
    ]),
]


# ------------------------------------------------------------------ audio, frames, mix
def prosody(i, text):
    rnd = random.Random(500 + i)
    rate = rnd.choice([-3, -1, 0, 2, 4])
    pitch = rnd.choice([-1, 0, 1, 2])
    if "?" in text:
        pitch += 2
    if text.rstrip().endswith("!"):
        rate += 2
        pitch += 2
    return f"{rate:+d}%", f"{pitch:+d}Hz"


def pause_after(i, last):
    rnd = random.Random(900 + i)
    return 0.5 if last else rnd.choice([0.25, 0.32, 0.4, 0.45])


def synth_audio(work):
    os.makedirs(work, exist_ok=True)
    wavs, bi = [], 0
    for sc in SCENES:
        sc["cues"], t = [], 0.0
        for j, text in enumerate(sc["beats"]):
            rate, pitch = prosody(bi, text)
            key = hashlib.md5(f"{VOICE}|{rate}|{pitch}|{text}".encode("utf-8")).hexdigest()[:12]
            mp3, wav = f"{work}/{key}.mp3", f"{work}/b{bi}.wav"
            if not os.path.exists(mp3):
                asyncio.run(edge_tts.Communicate(text, VOICE, rate=rate, pitch=pitch).save(mp3))
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
        tag_pill(dr, 60, 96, sc["tag"], 1.0, size=28)
        TX(dr, (60, 168), sc["title"], "title", 74, TXT, anchor="lm", stroke=2)
        dr.text((62, 232), sc["question"], font=F("lab", 36), fill=(190, 200, 235), anchor="lm")
    text = sc["beats"][j]
    fnt = F("cap", 40)
    lines = wrap(dr, text, fnt, 880, False)
    h = len(lines) * 58 + 36
    y1 = 1792
    card(dr, 50, y1 - h, VW - 50, y1, fill=(30, 36, 92), depth=12, radius=26)
    for i, ln in enumerate(lines):
        dr.text((VW / 2, y1 - h + 18 + 29 + i * 58), ln, font=fnt, fill=TXT, anchor="mm")
    dr.text((VW / 2, 1848), "Data: StatCounter, Pakistan, mobile phones, Apr 2010 to Oct 2026.", font=F("small", 22),
            fill=MUTED, anchor="mm")
    dr.text((VW / 2, 1880), "This shows mobile internet use, not phone sales.", font=F("small", 22), fill=MUTED,
            anchor="mm")


def overlay(dr, p):
    dr.rectangle([0, VH - 10, VW, VH], fill=(30, 36, 92))
    dr.rectangle([0, VH - 10, int(VW * p), VH], fill=GOLD)
    tag_pill(dr, 28, 40, "SAMPLE PREVIEW", 1.0, fill=(30, 36, 92), color=MUTED, size=18)


def frame(sc, s):
    global CANVAS
    CANVAS = BGIMG.copy()
    dr = ImageDraw.Draw(CANVAS)
    cues = sc["cues"]
    sc["body"](dr, s, sc["dur"], cues)
    j = max([i for i, (a, b) in enumerate(cues) if a <= s] or [0])
    chrome(dr, sc, j)
    return dr


def render(full, out_video, work):
    total = sum(sc["dur"] for sc in SCENES)
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", full,
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", "-shortest", out_video]
    log = open(f"{work}/ffmpeg.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    done = 0.0
    for sc in SCENES:
        for k in range(max(1, round(sc["dur"] * FPS))):
            s = k / FPS
            dr = frame(sc, s)
            overlay(dr, (done + s) / total)
            proc.stdin.write(CANVAS.tobytes())
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
    hits = sorted(h for h in (starts[i] + sc["cues"][j][0] + off for i, sc in enumerate(SCENES)
                              for j, off in sc.get("hits", [])) if 3.5 < h < total - 1)
    music = f"{work}/music.wav"
    make_music(total, hits, music)
    flt = ("[0:a]pan=stereo|c0=c0|c1=c0,asplit=2[nar][sc];[1:a]volume=0.55[mus];"
           "[mus][sc]sidechaincompress=threshold=0.02:ratio=9:attack=15:release=500:makeup=1[duck];"
           "[nar][duck]amix=inputs=2:duration=first:normalize=0,pan=stereo|c0=c0|c1=c1[a]")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", narr_video, "-i", music, "-filter_complex", flt, "-map",
                    "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out], check=True)
    return hits


if __name__ == "__main__":
    M.check()
    work = sys.argv[2]
    full = synth_audio(work)
    total = sum(sc["dur"] for sc in SCENES)
    for sc in SCENES:
        print(sc["name"], round(sc["dur"], 1), [round(b - a, 1) for a, b in sc["cues"]])
    print("TOTAL", round(total, 1))
    if len(sys.argv) > 3 and sys.argv[3] == "--audio-only":
        sys.exit(0)
    narr = f"{work}/narration_only.mp4"
    render(full, narr, work)
    hits = mix_music(narr, sys.argv[1], work, total)
    print("hits", [round(h, 1) for h in hits])
    print("done", sys.argv[1])
