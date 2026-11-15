"""Pakistan mobile brands: ONE chart, a smooth 3D bar race with official logos. No voice-over: on-screen text + suspense music.

    python bar_race_smooth.py OUT.mp4 WORKDIR [--fps 60]

Smoothness:
  * one continuous time -> month curve (monotone cubic): the race never stops, it only speeds up and slows down gently
  * bar lengths use Catmull-Rom interpolation between months (no kinks at month boundaries)
  * rows follow their rank with a critically damped spring (smooth overtakes, no jumps), computed once per frame
  * bars are drawn per frame as anti-aliased 3D capsules with sub-pixel position; all other round shapes are supersampled
  * rendered in parallel (all CPU cores) at 60 fps
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mobile_stats_3d_en as T  # noqa  (fonts, background, text helper)
import mobile_stats_charts as M  # noqa  (real data)
from common import FFMPEG, wrap  # noqa
from logos import logo  # noqa
from music_suspense import make_music  # noqa

VW, VH = 1080, 1920
S, MI, MONTHS, N = M.S, M.MI, M.MONTHS, M.N
BR = M.RACE_BRANDS
B = len(BR)
F, TXT, MUTED, GOLD = T.F, T.TXT, T.MUTED, T.GOLD
CARD = (30, 36, 92)
FPS = int(os.environ.get("BARRACE_FPS", "60"))

# ------------------------------------------------------------------ the one continuous time -> month curve
SEG = [("2010-12", 0.9), ("2011-09", 0.9), ("2012-02", 1.0), ("2013-06", 2.0), ("2015-04", 1.8), ("2015-12", 0.8),
       ("2019-06", 2.4), ("2020-04", 1.3), ("2020-10", 1.0), ("2023-09", 2.4), ("2024-06", 0.8), ("2026-06", 2.1),
       ("2026-10", 0.9)]
SPEED_SCALE = 1.17  # whole race a bit faster so the video fits in 2 minutes
TS = [0.0]
MS = [0.0]
for _lab, _sp in SEG:
    MS.append(float(MI[_lab]))
    TS.append(TS[-1] + (MS[-1] - MS[-2]) / (_sp * SPEED_SCALE))
T_MOTION = TS[-1]
TAIL = 10.0
TOTAL = T_MOTION + TAIL
assert TOTAL <= 119.5, TOTAL  # hard limit: never longer than 2 minutes


def _tangents():
    n = len(TS)
    h = np.diff(TS)
    d = np.diff(MS) / h
    m = np.zeros(n)
    for k in range(1, n - 1):
        w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
        m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0] = d[0]
    m[-1] = 0.0  # glide to a soft stop on the last month
    return h, m


_H, _MT = _tangents()


def month_at(t):
    if t >= TS[-1]:
        return MS[-1]
    k = max(0, min(len(TS) - 2, int(np.searchsorted(TS, t, side="right")) - 1))
    s = (t - TS[k]) / _H[k]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * MS[k] + (s3 - 2 * s2 + s) * _H[k] * _MT[k] + (-2 * s3 + 3 * s2) * MS[k + 1]
            + (s3 - s2) * _H[k] * _MT[k + 1])


_TG = np.arange(0.0, T_MOTION + 0.005, 0.01)
_MG = np.array([month_at(t) for t in _TG])


def t_of(label):
    return float(_TG[min(len(_TG) - 1, int(np.searchsorted(_MG, MI[label] - 1e-6)))])


def catmull(arr, m):
    i = int(math.floor(m))
    f = m - i
    p0, p1, p2, p3 = arr[max(i - 1, 0)], arr[min(i, N - 1)], arr[min(i + 1, N - 1)], arr[min(i + 2, N - 1)]
    v = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    return max(float(v), 0.0)


# ------------------------------------------------------------------ on-screen text (replaces the voice-over)
CAPS = [
    (0.0, "In 2010, almost every phone in Pakistan was a Nokia."),
    (t_of("2010-11"), "Each bar is a phone brand. A longer bar means more people use it."),
    (t_of("2011-09"), "September 2011: Nokia reaches 87%. That is the top."),
    (t_of("2012-04"), "After that, Nokia only goes down."),
    (t_of("2013-08"), "Samsung grows, month after month."),
    (t_of("2015-08"), "August 2015: Samsung passes Nokia. A new king!"),
    (t_of("2016-04"), "Samsung stays number one for more than eight years."),
    (t_of("2019-08"), "In 2019, Samsung reaches 41%. Its best time."),
    (t_of("2020-04"), "April 2020: Huawei reaches 17%. Its best month."),
    (t_of("2021-03"), "Oppo and Vivo join the fight."),
    (t_of("2022-11"), "Samsung keeps falling. Infinix and Tecno rise."),
    (t_of("2023-12"), "December 2023: Infinix becomes number one!"),
    (t_of("2024-05"), "Xiaomi grows fast. Apple starts to grow too."),
    (t_of("2025-10"), "Now Samsung and Infinix fight for number one."),
    (t_of("2026-09"), "October 2026: Samsung 17%, Infinix 16%. A very close race."),
    (T_MOTION + 3.4, "7 of the top 10 are Chinese brands. Nokia has just 0.2%."),
    (T_MOTION + 6.6, "Which brand do you use? Tell us in the comments."),
]
EVENTS = [(t_of("2011-09"), t_of("2011-09") + 3.6, "NOKIA PEAK: 87%"),
          (t_of("2015-08"), t_of("2015-08") + 4.2, "SAMSUNG PASSES NOKIA!"),
          (t_of("2023-12"), t_of("2023-12") + 4.2, "INFINIX IS NUMBER 1!"),
          (t_of("2026-09"), T_MOTION + 3.4, "SAMSUNG 17%  vs  INFINIX 16%"),
          (T_MOTION + 3.4, T_MOTION + 6.4, "7 OF THE TOP 10 ARE CHINESE")]


def check():
    v = lambda b, d: float(S[b][MI[d]])  # noqa
    assert 80.5 < v("Nokia", "2010-04") < 81.0 and 87.0 < v("Nokia", "2011-09") < 87.3
    assert S["Nokia"].max() == S["Nokia"][MI["2011-09"]]
    assert all(v("Nokia", d) < v("Nokia", "2011-09") for d in MONTHS[MI["2011-10"]:])
    assert v("Nokia", "2013-06") < v("Nokia", "2012-04") < v("Nokia", "2011-09")
    assert v("Samsung", "2015-08") > v("Nokia", "2015-08") and v("Samsung", "2015-07") < v("Nokia", "2015-07")
    assert v("Samsung", "2013-08") < v("Samsung", "2015-03")
    for d in MONTHS[MI["2015-08"]:MI["2023-11"] + 1]:
        assert max(BR, key=lambda b: S[b][MI[d]]) == "Samsung", d
    assert 41.0 < v("Samsung", "2019-08") < 41.6 and S["Samsung"].max() == S["Samsung"][MI["2019-08"]]
    assert 17.0 < v("Huawei", "2020-04") < 17.4 and S["Huawei"].max() == S["Huawei"][MI["2020-04"]]
    assert v("Samsung", "2022-11") < v("Samsung", "2019-08")
    assert v("Infinix", "2023-06") > v("Infinix", "2021-03") and v("Tecno", "2023-06") > v("Tecno", "2021-03")
    order = sorted(BR, key=lambda b: -v(b, "2023-12"))
    assert order[0] == "Infinix" and sorted(BR, key=lambda b: -v(b, "2023-11"))[0] == "Samsung"
    assert v("Xiaomi", "2025-10") > v("Xiaomi", "2024-05") > v("Xiaomi", "2023-01")
    assert v("Apple", "2026-10") > v("Apple", "2025-10") > v("Apple", "2024-05")
    for d in MONTHS[MI["2025-10"]:]:
        assert abs(v("Samsung", d) - v("Infinix", d)) < 4, d
    assert abs(v("Samsung", "2026-10") - 17.0) < 0.1 and abs(v("Infinix", "2026-10") - 16.1) < 0.1
    top10 = sorted(BR, key=lambda b: -v(b, "2026-10"))[:10]
    assert sum(1 for b in top10 if b in M.CHINA) == 7 and 0.15 < v("Nokia", "2026-10") < 0.25
    assert v("Vivo", "2021-03") > 5 and v("Oppo", "2021-03") > 5
    print("data checks ok")


# ------------------------------------------------------------------ spring-smoothed rank positions (computed once)
P = AX = None


def simulate():
    nf = int(round(TOTAL * FPS)) + 1
    vals0 = np.array([catmull(S[b], 0.0) for b in BR])
    order = list(np.argsort(-vals0, kind="stable"))
    pos = np.zeros(B)
    for r, i in enumerate(order):
        pos[i] = r
    vel = np.zeros(B)
    ax = max(float(vals0.max()), 8.0)
    P_ = np.zeros((nf, B), np.float32)
    AX_ = np.zeros(nf, np.float32)
    sub, w = 4, 10.0
    dt = 1.0 / (FPS * sub)
    for f in range(nf):
        m = month_at(f / FPS)
        vals = np.array([catmull(S[b], m) for b in BR])
        changed = True
        while changed:  # order with a little hysteresis, so near-equal brands do not flicker
            changed = False
            for i in range(B - 1):
                a_, b_ = order[i], order[i + 1]
                if vals[b_] > vals[a_] + 0.06:
                    order[i], order[i + 1] = b_, a_
                    changed = True
        tgt = np.zeros(B)
        for r, i in enumerate(order):
            tgt[i] = r
        for _ in range(sub):
            acc = w * w * (tgt - pos) - 2 * w * vel
            vel += acc * dt
            pos += vel * dt
        P_[f] = pos
        ax += (max(float(vals.max()), 8.0) - ax) * (1 - math.exp(-1.0 / (FPS * 0.45)))
        AX_[f] = ax
    return P_, AX_


# ------------------------------------------------------------------ anti-aliased building blocks
def soft_card(w, h, radius, fill, edge=None, ew=3, sh_off=(4, 11), sh_blur=9, sh_alpha=165, gloss=True):
    pad, ss = 26, 3
    W, H = w + 2 * pad, h + 2 * pad
    im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    if sh_alpha:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([(pad + sh_off[0]) * ss, (pad + sh_off[1]) * ss, (pad + sh_off[0] + w) * ss,
                                              (pad + sh_off[1] + h) * ss], radius=radius * ss, fill=(2, 4, 14, sh_alpha))
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(sh_blur * ss)))
    d = ImageDraw.Draw(im)
    fa = fill if len(fill) == 4 else tuple(fill) + (255,)
    d.rounded_rectangle([pad * ss, pad * ss, (pad + w) * ss, (pad + h) * ss], radius=radius * ss, fill=fa,
                        outline=(tuple(edge) + (255,)) if edge else None, width=ew * ss)
    if gloss:
        d.line([(pad + radius) * ss, (pad + 4) * ss, (pad + w - radius) * ss, (pad + 4) * ss],
               fill=tuple(T.tint(fill[:3], 0.32)) + (200,), width=3 * ss)
    return im.resize((W, H), Image.LANCZOS), pad


def paste_card(canvas, x, y, spr):
    im, pad = spr
    canvas.paste(im, (int(x - pad), int(y - pad)), im)


@lru_cache(maxsize=None)
def bake_cards():
    return {
        "date": soft_card(500, 104, 36, (46, 36, 100), edge=GOLD),
        "band": soft_card(VW - 68, 112, 36, (96, 72, 10, 150), edge=GOLD, sh_alpha=0, gloss=False),
        "timeline": soft_card(VW - 100, 112, 30, (24, 30, 76)),
        "caption": soft_card(VW - 100, 150, 34, CARD),
    }


def paste_sub(canvas, spr, x, y, a=1.0):
    """Paste an RGBA sprite at a fractional position (bilinear) with opacity a."""
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    if fx > 1e-3 or fy > 1e-3:
        spr = spr.transform(spr.size, Image.AFFINE, (1, 0, -fx, 0, 1, -fy), resample=Image.BILINEAR)
    if a < 0.99:
        spr = spr.copy()
        spr.putalpha(spr.getchannel("A").point([int(v * a) for v in range(256)]))
    canvas.paste(spr, (ix, iy), spr)


BW, BH = 176, 70
BADGE_PAD = 16


@lru_cache(maxsize=None)
def badge(brand):
    ss, pad, inset = 3, BADGE_PAD, 4
    w, h = BW, BH
    W, H = w + 2 * pad, h + 2 * pad
    im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([(pad + 3) * ss, (pad + 8) * ss, (pad + w + 3) * ss, (pad + h + 8) * ss],
                                         radius=h * ss // 2, fill=(2, 4, 14, 175))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5 * ss)))
    d = ImageDraw.Draw(im)
    col = tuple(M.COL.get(brand, (120, 130, 150)))
    d.rounded_rectangle([pad * ss, pad * ss, (pad + w) * ss, (pad + h) * ss], radius=h * ss // 2, fill=col + (255,))
    d.rounded_rectangle([(pad + inset) * ss, (pad + inset) * ss, (pad + w - inset) * ss, (pad + h - inset) * ss],
                        radius=(h // 2 - inset) * ss, fill=(255, 255, 255, 255))
    lg = logo(brand)
    if lg is not None:
        bb = lg.getbbox()
        if bb:
            lg = lg.crop(bb)
        mw, mh = (w - 2 * 26) * ss, (h - 2 * 12) * ss
        sc = min(mw / lg.width, mh / lg.height)
        lg = lg.resize((max(1, int(lg.width * sc)), max(1, int(lg.height * sc))), Image.LANCZOS)
        im.alpha_composite(lg, (int((W * ss - lg.width) / 2), int((H * ss - lg.height) / 2)))
    else:
        d.text((W * ss // 2, H * ss // 2), brand.upper(), font=F("num", 30 * ss), fill=col + (255,), anchor="mm")
    return im.resize((W, H), Image.LANCZOS)


@lru_cache(maxsize=None)
def chip(r=22):
    ss = 4
    im = Image.new("RGBA", ((2 * r + 6) * ss,) * 2, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([3 * ss, 3 * ss, (2 * r + 3) * ss, (2 * r + 3) * ss], fill=(40, 48, 104, 255), outline=(96, 112, 190, 255),
              width=3 * ss)
    return im.resize((2 * r + 6, 2 * r + 6), Image.LANCZOS)


@lru_cache(maxsize=None)
def crown_sprite():
    ss = 5
    w, h = 64, 44
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = [(8, 36), (8, 14), (20, 24), (32, 8), (44, 24), (56, 14), (56, 36)]
    d.polygon([(x * ss + 3 * ss, y * ss + 4 * ss) for x, y in pts], fill=(3, 5, 16, 150))
    d.polygon([(x * ss, y * ss) for x, y in pts], fill=GOLD + (255,), outline=(150, 90, 0, 255), width=ss)
    d.rectangle([8 * ss, 30 * ss, 56 * ss, 36 * ss], fill=(255, 226, 110, 255))
    for px, py in ((8, 12), (32, 6), (56, 12)):
        d.ellipse([(px - 4) * ss, (py - 4) * ss, (px + 4) * ss, (py + 4) * ss], fill=(255, 255, 255, 255))
    return im.resize((w, h), Image.LANCZOS)


@lru_cache(maxsize=None)
def pill_sprite(text, size=40):
    f = F("labb", size)
    w = int(ImageDraw.Draw(Image.new("RGB", (4, 4))).textlength(text, font=f)) + 70
    spr = soft_card(w, 76, 38, (46, 36, 100), edge=GOLD, sh_off=(3, 8), sh_blur=7)
    im, pad = spr
    ImageDraw.Draw(im).text((im.width / 2, im.height / 2 - 1), text, font=f, fill=GOLD, anchor="mm")
    return im


@lru_cache(maxsize=None)
def dot(r, lv):
    n = 2 * r + 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
    a = np.clip(1 - d, 0, 1) ** 1.5 * (0.18, 0.3, 0.45)[lv]
    arr = np.dstack([np.full((n, n), 190, np.uint8), np.full((n, n), 205, np.uint8), np.full((n, n), 255, np.uint8),
                     (a * 255).astype(np.uint8)])
    return Image.fromarray(arr)


PARTS = []
for _i in range(40):
    _r = np.random.default_rng(100 + _i)
    PARTS.append((_r.random(), _r.random(), 0.4 + _r.random() * 0.9, 3 + int(_r.random() * 5), int(_r.random() * 3)))


def particles(canvas, t):
    for fx, fy, sp, r, lv in PARTS:
        x = (fx * VW + 40 * math.sin(t * 0.35 * sp + fy * 6)) % VW
        y = (fy * VH - t * 20 * sp) % VH
        canvas.paste(dot(r, lv), (int(x - r), int(y - r)), dot(r, lv))


# ------------------------------------------------------------------ 3D capsule bar (anti-aliased, sub-pixel)
BAR_H, BAR_PAD = 52, 26
LIGHT = np.array([-0.38, -0.62, 0.69], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def bar_sprite(L, c, fy, shine, a):
    H, pad = BAR_H, BAR_PAD
    W, Hh = int(math.ceil(L)) + 2 * pad + 10, H + 2 * pad
    yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
    yy += 0.5
    xx += 0.5
    r = H / 2.0
    cy = pad + r + fy
    xa, xb = pad + r, pad + L - r
    px = np.clip(xx, xa, xb)
    dx, dy = xx - px, yy - cy
    d = np.sqrt(dx * dx + dy * dy) - r
    alpha = np.clip(0.5 - d, 0.0, 1.0)
    nx, ny = dx / r, dy / r
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    spec = diff ** 28
    g = 0.90 + 0.10 * np.clip((xx - pad) / max(L, 1.0), 0, 1)
    base = np.array(c, np.float32)
    col = base[None, None, :] * (0.30 + 0.92 * diff * g)[..., None] + 255.0 * (0.55 * spec)[..., None]
    bx = pad + shine * (L + 80) - 40
    col += 255.0 * (np.exp(-((xx - bx) / 22.0) ** 2) * 0.32 * np.clip(diff * 1.2, 0, 1))[..., None]
    col *= (1.0 - 0.22 * np.clip(ny, 0, 1) ** 2)[..., None]
    col = np.clip(col, 0, 255)
    sx, sy = xx - 5, yy - 10
    ds = np.sqrt((sx - np.clip(sx, xa, xb)) ** 2 + (sy - cy) ** 2) - r
    sh = np.clip(1 - ds / 18.0, 0, 1) ** 2 * 0.55
    oa = alpha + sh * (1 - alpha)
    shc = np.array([2, 4, 14], np.float32)
    rgb = (col * alpha[..., None] + shc * (sh * (1 - alpha))[..., None]) / np.maximum(oa, 1e-4)[..., None]
    out = np.dstack([np.clip(rgb, 0, 255), np.clip(oa * a * 255, 0, 255)]).astype(np.uint8)
    return Image.fromarray(out)


# ------------------------------------------------------------------ the frame
X0, MAXLEN, PITCH, YTOP = 304, 500, 106, 408
TL_Y = 1546
MEDAL = {1: (255, 196, 20), 2: (200, 210, 226), 3: (214, 132, 64)}
YEAR_X = {}


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def ease_back(x):
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def lerpc(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def static_layer(canvas):
    cards = bake_cards()
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (VW / 2, 84), "TOP 10 MOBILE BRANDS", "title", 72, TXT, anchor="mm", stroke=3)
    T.TX(dr, (VW / 2, 156), "IN PAKISTAN", "title", 52, GOLD, anchor="mm")
    paste_card(canvas, 290, 218, cards["date"])
    paste_card(canvas, 34, YTOP - 56, cards["band"])
    paste_card(canvas, 50, TL_Y - 56, cards["timeline"])
    paste_card(canvas, 50, 1650, cards["caption"])
    dr = ImageDraw.Draw(canvas)
    for yr in (2010, 2014, 2018, 2022, 2026):
        k = MI.get(f"{yr}-01", 0 if yr == 2010 else N - 1)
        dr.text((90 + 900 * k / (N - 1), TL_Y + 36), str(yr), font=F("lab", 24), fill=MUTED, anchor="mm")
    dr.text((VW / 2, 1848), "Data: StatCounter, Pakistan, mobile phones, Apr 2010 to Oct 2026. Logos belong to their owners.",
            font=F("small", 20), fill=MUTED, anchor="mm")
    dr.text((VW / 2, 1880), "This shows mobile internet use, not phone sales.", font=F("small", 22), fill=MUTED, anchor="mm")
    return dr


def timeline_dyn(canvas, m):
    ss = 3
    x0, x1 = 90, 990
    bx, by, bw, bh = 60, TL_Y - 30, 960, 60
    im = Image.new("RGBA", (bw * ss, bh * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    X = lambda k: (x0 + (x1 - x0) * k / (N - 1) - bx) * ss  # noqa
    cy = (bh / 2) * ss
    d.line([X(0), cy, X(N - 1), cy], fill=(58, 70, 126, 255), width=8 * ss)
    for xx in (X(0), X(N - 1)):
        d.ellipse([xx - 4 * ss, cy - 4 * ss, xx + 4 * ss, cy + 4 * ss], fill=(58, 70, 126, 255))
    d.line([X(0), cy, X(m), cy], fill=GOLD + (255,), width=8 * ss)
    d.ellipse([X(0) - 4 * ss, cy - 4 * ss, X(0) + 4 * ss, cy + 4 * ss], fill=GOLD + (255,))
    for lab, col in (("2011-09", M.COL["Nokia"]), ("2015-08", M.COL["Samsung"]), ("2023-12", M.COL["Infinix"])):
        k = MI[lab]
        c = col if m >= k else (90, 100, 150)
        d.ellipse([X(k) - 9 * ss, cy - 9 * ss, X(k) + 9 * ss, cy + 9 * ss], fill=tuple(c) + (255,), outline=(255, 255, 255, 255),
                  width=2 * ss)
    mx = X(m)
    d.ellipse([mx - 26 * ss, cy - 26 * ss, mx + 26 * ss, cy + 26 * ss], fill=GOLD + (60,))
    d.ellipse([mx - 16 * ss, cy - 16 * ss, mx + 16 * ss, cy + 16 * ss], fill=GOLD + (255,), outline=(255, 244, 190, 255),
              width=3 * ss)
    d.ellipse([mx - 9 * ss, cy - 11 * ss, mx - 1 * ss, cy - 4 * ss], fill=(255, 255, 255, 190))
    im = im.resize((bw, bh), Image.LANCZOS)
    canvas.paste(im, (bx, by), im)


def alpha_window(t, t0, t1, fade=0.5):
    return max(0.0, min(ease((t - t0) / fade), ease((t1 - t) / fade)))


def render_frame(k):
    t = k / FPS
    m = month_at(t)
    canvas = T.BGIMG.copy()
    particles(canvas, t)
    static_layer(canvas)
    dr = ImageDraw.Draw(canvas)
    yi = MONTHS[min(N - 1, int(round(m)))]
    T.TX(dr, (VW / 2, 272), f"{M.MNAME[int(yi[5:7]) - 1].upper()}  {yi[:4]}", "num", 80, GOLD, anchor="mm", stroke=2)

    pk, ax = P[k], float(AX[k])
    vals = [catmull(S[b], m) for b in BR]
    for i in sorted(range(B), key=lambda j: -pk[j]):
        b, v, p = BR[i], vals[i], float(pk[i])
        a = max(0.0, min(1.0, (v - 0.15) / 0.4)) * max(0.0, min(1.0, 9.9 - p))
        if a < 0.02:
            continue
        y = YTOP + p * PITCH
        c = M.COL.get(b, MUTED)
        L = min(max(BAR_H * 1.05, v / ax * MAXLEN), MAXLEN * 1.05)
        top = y - BAR_H / 2 - BAR_PAD
        iy = int(math.floor(top))
        shine = (t * 0.42 + i * 0.19) % 1.6 - 0.3
        spr = bar_sprite(L, c, top - iy, shine, a)
        canvas.paste(spr, (X0 - BAR_PAD, iy), spr)
        paste_sub(canvas, badge(b), 100 - BADGE_PAD, y - 10 - BH / 2 - BADGE_PAD, a)
        dr.text((100 + BW / 2, y + 38), b, font=F("lab", 21), fill=T.mix(MUTED, a), anchor="mm")
        T.TX(dr, (X0 + L + 24, y), f"{v:.1f}%", "num", 46, T.mix(TXT, a), anchor="lm")
        rank = int(round(p)) + 1
        if rank in MEDAL:
            sp = T.sphere_sprite(MEDAL[rank], 26)
            paste_sub(canvas, sp, 58 - 26, y - 26, a)
            dr.text((58, y + 1), str(rank), font=F("num", 30), fill=T.mix((60, 36, 0), a), anchor="mm")
        else:
            paste_sub(canvas, chip(), 58 - 25, y - 25, a)
            dr.text((58, y + 1), str(min(rank, 10)), font=F("num", 26), fill=T.mix(MUTED, a), anchor="mm")
        lead = max(0.0, 1.0 - p / 0.5)
        if lead > 0.03:
            paste_sub(canvas, crown_sprite(), 100 + 8, y - 10 - BH / 2 - 34, lead * a)

    timeline_dyn(canvas, m)

    for t0, t1, text in EVENTS:
        if t0 <= t < t1:
            a = alpha_window(t, t0, t1, 0.3)
            sc = 0.6 + 0.4 * ease_back((t - t0) / 0.45)
            spr = pill_sprite(text)
            sp2 = spr.resize((max(2, int(spr.width * sc)), max(2, int(spr.height * sc))), Image.BICUBIC)
            if a < 0.99:
                sp2.putalpha(sp2.getchannel("A").point([int(v * a) for v in range(256)]))
            canvas.paste(sp2, (int(VW / 2 - sp2.width / 2), int(1450 - sp2.height / 2)), sp2)

    dr = ImageDraw.Draw(canvas)
    for j, (c0, text) in enumerate(CAPS):
        c1 = CAPS[j + 1][0] if j + 1 < len(CAPS) else TOTAL
        if c0 <= t < c1:
            a = alpha_window(t, c0, c1, 0.5) if j else min(ease(t / 0.8), ease((c1 - t) / 0.5))
            fnt = F("cap", 36)
            lines = wrap(dr, text, fnt, 880, False)
            for q, ln in enumerate(lines):
                yy = 1725 + (q - (len(lines) - 1) / 2) * 52
                dr.text((VW / 2, yy), ln, font=fnt, fill=lerpc(CARD, TXT, a), anchor="mm")
    prog = t / TOTAL
    dr.rectangle([0, VH - 10, VW, VH], fill=CARD)
    dr.rectangle([0, VH - 10, int(VW * prog), VH], fill=GOLD)
    return canvas.tobytes()


def init_worker(fps):
    global FPS, P, AX
    FPS = fps
    P, AX = simulate()


def render(out_video, work, fps):
    global P, AX
    init_worker(fps)
    n = int(round(TOTAL * FPS))
    music = f"{work}/music_smooth.wav"
    from music_race import make_race_music
    bounds = [0.0] + [round(e[0], 3) for e in EVENTS[:4]] + [TOTAL]
    bpms = [102, 111, 116, 120, 116]
    bars = [max(3, round((bounds[i + 1] - bounds[i]) * bpms[i] / 240.0)) for i in range(5)]
    print('music bounds', bounds, 'bars', bars, flush=True)
    make_race_music(bounds, bars, music)
    hits = bounds[1:5]
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", music,
           "-af", f"volume=0.5,alimiter=limit=0.8",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", out_video]
    log = open(f"{work}/ffmpeg_smooth.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, (os.cpu_count() or 2) - 0), initializer=init_worker, initargs=(fps,)) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            proc.stdin.write(b)
            if i % 600 == 0:
                print("frame", i, "/", n, flush=True)
    proc.stdin.close()
    proc.wait()
    log.close()
    return hits


if __name__ == "__main__":
    check()
    print("motion", round(T_MOTION, 1), "s, total", round(TOTAL, 1), "s")
    for j, (c0, text) in enumerate(CAPS):
        c1 = CAPS[j + 1][0] if j + 1 < len(CAPS) else TOTAL
        print(f"  caption {c1 - c0:5.1f}s  {text}")
    fps = FPS
    if "--fps" in sys.argv:
        fps = int(sys.argv[sys.argv.index("--fps") + 1])
    hits = render(sys.argv[1], sys.argv[2], fps)
    print("hits", hits)
    print("done", sys.argv[1])
