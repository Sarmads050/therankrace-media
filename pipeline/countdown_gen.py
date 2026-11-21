"""Top-10 COUNTDOWN Reel (#10 -> #1), driven by the same JSON config as bar_race_gen.py.

    RACE_CFG=data/race/gdp.json python countdown_gen.py OUT.mp4 WORKDIR [--fps 60]

Why this format (research + our own Buffer numbers, 2026-11): Instagram ranks Reels by watch time / completion first and by
sends (DM shares) second.  The 43 s bar races were watched ~10-18 s on average.  A countdown is shorter (~27 s), readable on a
phone (one BIG card at a time instead of ten tiny bars), has suspense (#1 comes last, unrevealed rows are shown as ???) and
a story per card (up/down since 10 years ago).  Pakistan gets its own moment (the viewers we want to reach).

Layout (inside the Instagram safe zone x 60-900, y 160-1640):
  title            y 205-265
  big reveal card  y 330-905   rank number + flag + name + full number + change chip
  leaderboard      y 930-1590  10 rows, fills from #10 up to #1, slim bars grow in
  source line      y 1610-1635
"""
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bar_race_smooth as R  # noqa
import mobile_stats_3d_en as T  # noqa
from common import FFMPEG, wrap  # noqa
from flags import flag  # noqa

VW, VH = 1080, 1920
F, TXT, MUTED, GOLD, CARD = T.F, T.TXT, T.MUTED, T.GOLD, R.CARD
CX = 480
GREEN = (60, 205, 120)
RED = (255, 100, 100)
R.BAR_H = 12
BAR_H, BAR_PAD = 12, R.BAR_PAD

CFG = json.load(open(os.environ["RACE_CFG"], encoding="utf-8"))
YEARS = CFG["years"]
ENT = CFG["entities"]
KIND = CFG["fmt"]
HL = CFG.get("highlight") or "PAK"
META = CFG.get("meta", {})
SER = {e["iso"]: [float(x or 0.0) for x in CFG["series"][e["iso"]]] for e in ENT}
NAME = {e["iso"]: e["name"] for e in ENT}
NAME.setdefault("PAK", "Pakistan")
NY = len(YEARS)
LAST = NY - 1
Y1 = YEARS[LAST]


def fmt(v):
    """numbers are always written in full"""
    if KIND in ("usd", "usdpc"):
        return f"${v:,.0f}"
    if KIND == "mt":
        return f"{v * 1e6:,.0f} tonnes"
    if KIND == "dec1":
        return f"{v:.1f}"
    if KIND == "dec2":
        return f"{v:.2f}"
    if KIND == "pct":
        return f"{v:.1f}%"
    return f"{v:,.0f}"


# ------------------------------------------------------------------ the ranking (latest year) and its story
def ranking(idx):
    return sorted((i for i in SER if SER[i][idx] > 0), key=lambda i: -SER[i][idx])


NOW = ranking(LAST)
TOP = NOW[:10]
VMAX = SER[TOP[0]][LAST]
IDX0 = max(0, LAST - 10)
THEN = ranking(IDX0)


def change_text(iso, rank):
    """chip text + colour: places gained / lost since 10 years ago, 'NUMBER 1 SINCE ...' for the leader"""
    if rank == 1:
        k = LAST
        while k > 0 and ranking(k - 1)[:1] == [iso]:
            k -= 1
        return (f"NUMBER 1 SINCE {YEARS[k]}", GOLD) if k > 0 else (f"NUMBER 1 EVERY YEAR SINCE {YEARS[0]}", GOLD)
    if iso not in THEN:
        return (f"NEW IN THE TOP 10 (OUT IN {YEARS[IDX0]})", GREEN)
    r0 = THEN.index(iso) + 1
    if r0 > 10:
        return (f"NEW IN THE TOP 10 (OUT IN {YEARS[IDX0]})", GREEN)
    if r0 > rank:
        d = r0 - rank
        return (f"UP {d} PLACE{'S' if d > 1 else ''} SINCE {YEARS[IDX0]}", GREEN)
    if r0 < rank:
        d = rank - r0
        return (f"DOWN {d} PLACE{'S' if d > 1 else ''} SINCE {YEARS[IDX0]}", RED)
    return (f"SAME RANK SINCE {YEARS[IDX0]}", MUTED)


STORY = {iso: change_text(iso, r + 1) for r, iso in enumerate(TOP)}
PK_IN = HL in TOP
PK_RANK = (TOP.index(HL) + 1) if PK_IN else (META.get("hl_rank") or (NOW.index(HL) + 1 if HL in NOW else None))
PK_VAL = SER[HL][LAST] if HL in SER and SER[HL][LAST] > 0 else META.get("hl_value")
PK_CARD = (not PK_IN) and PK_RANK is not None and PK_VAL is not None

# ------------------------------------------------------------------ timeline
# Owner's rule (2026-11-21): relaxing music and the video must be AT LEAST 90 s (max 2 min). Slow cards with a history line
# that draws itself fill the time with real information instead of padding.
INTRO = 4.0
DUR = {r: 7.5 for r in range(10, 3, -1)}
DUR.update({3: 8.5, 2: 9.5, 1: 12.0})
PK_DUR, CTA_DUR = 8.0, 6.0
S = {}
_t = INTRO
for _r in range(10, 0, -1):
    S[_r] = _t
    _t += DUR[_r]
T_PK = _t
_t += PK_DUR if PK_CARD else 0.0
T_CTA = _t
TOTAL = T_CTA + CTA_DUR
assert 90.0 <= TOTAL <= 119.5, TOTAL
COVER_S = S[1] + 2.0  # best cover frame: the #1 card with the whole board filled


def title_parts():
    t1 = CFG["title1"].strip()
    t2 = re.sub(r"\s*\b\d{4}\s*-\s*\d{4}\b\s*$", "", CFG["title2"]).strip()
    return t1, t2


T1, T2 = title_parts()


# ------------------------------------------------------------------ sprites
def _rounded_flag(iso, w, h, radius, border=3):
    fl = flag(iso)
    if fl is None:
        im = Image.new("RGBA", (w, h), (90, 100, 150, 255))
    else:
        sc = min(w / fl.width, h / fl.height)
        im = fl.resize((max(2, int(fl.width * sc)), max(2, int(fl.height * sc))), Image.LANCZOS)
    ss = 3
    W, H = im.width, im.height
    pad = 22
    out = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
    mask = Image.new("L", (W * ss, H * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W * ss - 1, H * ss - 1], radius=radius * ss, fill=255)
    mask = mask.resize((W, H), Image.LANCZOS)
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([pad + 4, pad + 9, pad + W + 4, pad + H + 9], radius=radius, fill=(2, 4, 14, 170))
    from PIL import ImageFilter
    out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)))
    frame = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    out.paste(frame, (pad, pad), mask)
    inner = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    inner.paste(im.convert("RGBA"), (0, 0))
    m2 = Image.new("L", (W * ss, H * ss), 0)
    ImageDraw.Draw(m2).rounded_rectangle([border * ss, border * ss, (W - border) * ss - 1, (H - border) * ss - 1],
                                         radius=max(2, radius - border) * ss, fill=255)
    m2 = m2.resize((W, H), Image.LANCZOS)
    out.paste(inner, (pad, pad), m2)
    return out, pad


_FL = {}


def flag_sprite(iso, w, h, radius):
    k = (iso, w, h)
    if k not in _FL:
        _FL[k] = _rounded_flag(iso, w, h, radius)
    return _FL[k]


def color_of(iso):
    over = {"USA": (70, 120, 245), "CHN": (238, 62, 52), "IND": (255, 150, 40), "PAK": (36, 170, 96), "JPN": (255, 110, 150),
            "DEU": (250, 200, 40), "GBR": (120, 100, 235), "FRA": (60, 150, 255), "ITA": (50, 200, 120), "RUS": (190, 90, 235),
            "BRA": (70, 205, 95), "CAN": (250, 90, 90), "IDN": (255, 85, 85), "NGA": (60, 190, 120), "SAU": (60, 200, 130),
            "ARE": (60, 200, 130), "UKR": (60, 150, 255), "ISR": (80, 160, 255), "TUR": (240, 70, 70), "KOR": (80, 160, 255)}
    if iso in over:
        return over[iso]
    fl = flag(iso)
    if fl is None:
        return (140, 150, 190)
    sm = fl.convert("RGB").resize((24, 16))
    best, bs = (140, 150, 190), -1
    for n, c in sm.getcolors(500):
        mx, mn = max(c), min(c)
        sat = (mx - mn) / max(mx, 1)
        if mx > 90 and sat > 0.35 and n * sat > bs:
            best, bs = c, n * sat
    return best


COL = {i: color_of(i) for i in set(TOP) | {HL}}
_CARDS = {}


def card(key, *a, **k):
    if key not in _CARDS:
        _CARDS[key] = R.soft_card(*a, **k)
    return _CARDS[key]


_BARS = {}


def bar(L, col):
    k = (int(L), col)
    if k not in _BARS:
        if len(_BARS) > 600:
            _BARS.clear()
        _BARS[k] = R.bar_sprite(max(L, BAR_H * 1.05), col, 0.0, 0.35, 1.0)
    return _BARS[k]


def fit(dr, text, kind, size, width, mn=24):
    while size > mn and dr.textlength(text, font=F(kind, size)) > width:
        size -= 2
    return size


def scaled(spr, sc, a=1.0):
    im = spr
    if abs(sc - 1.0) > 0.01:
        im = spr.resize((max(2, int(spr.width * sc)), max(2, int(spr.height * sc))), Image.BICUBIC)
    if a < 0.99:
        im = im.copy()
        im.putalpha(im.getchannel("A").point([int(v * a) for v in range(256)]))
    return im


# ------------------------------------------------------------------ rows (leaderboard)
ROW_Y0, ROW_P = 962, 66


def row_y(r):
    return ROW_Y0 + (r - 1) * ROW_P


def draw_row(canvas, dr, r, t):
    iso = TOP[r - 1]
    y = row_y(r)
    u = t - S[r]
    hi = iso == HL
    if u < 0:  # not revealed yet: a mystery row
        cs = card("row_hidden", 840, 58, 22, (24, 30, 80, 120), sh_alpha=0, gloss=False)
        R.paste_card(canvas, 60, y - 29, cs)
        dr.ellipse([100 - 20, y - 20, 100 + 20, y + 20], fill=(40, 48, 104), outline=(96, 112, 190), width=3)
        dr.text((100, y + 1), str(r), font=F("num", 22), fill=MUTED, anchor="mm")
        dr.text((212, y), "?  ?  ?  ?  ?", font=F("labb", 30), fill=(96, 108, 170), anchor="lm")
        return
    a = R.ease(u / 0.35)
    dx = (1 - a) * 140
    active = S[r] <= t < S[r] + DUR[r]
    cs = card("row_hi" if hi else "row", 840, 58, 22, (24, 62, 60, 215) if hi else (26, 32, 84, 205),
              edge=(GREEN if hi else None), ew=3, sh_alpha=0, gloss=False)
    R.paste_sub(canvas, cs[0], 60 - cs[1] + dx, y - 29 - cs[1], a)
    if active:
        glow = card("row_active", 840, 58, 22, (0, 0, 0, 0), edge=GOLD, ew=4, sh_alpha=0, gloss=False)
        R.paste_sub(canvas, glow[0], 60 - glow[1] + dx, y - 29 - glow[1], a * (0.6 + 0.4 * math.sin(t * 9)))
    if r in R.MEDAL:
        sp = T.sphere_sprite(R.MEDAL[r], 20)
        R.paste_sub(canvas, sp, 100 - 20 + dx, y - 20, a)
        dr.text((100 + dx, y + 1), str(r), font=F("num", 24), fill=T.mix((60, 36, 0), a), anchor="mm")
    else:
        R.paste_sub(canvas, R.chip(18), 100 - 21 + dx, y - 21, a)
        dr.text((100 + dx, y + 1), str(r), font=F("num", 22), fill=T.mix(MUTED, a), anchor="mm")
    fs, pad = flag_sprite(iso, 58, 38, 7)
    R.paste_sub(canvas, fs, 138 - pad + dx, y - 19 - pad, a)
    T.TX(dr, (212 + dx, y - 8), NAME[iso], "labb", 30, T.mix(GOLD if hi else TXT, a), anchor="lm", shadow=False)
    v = SER[iso][LAST]
    shown = v * R.ease(u / 0.9)
    T.TX(dr, (888 + dx, y - 8), fmt(shown if u < 0.9 else v), "num", 30, T.mix(TXT, a), anchor="rm", shadow=False)
    L = (676.0 * v / VMAX) * R.ease(u / 0.7)
    if L > BAR_H * 1.05:
        spr = bar(L, COL[iso])
        R.paste_sub(canvas, spr, 212 - BAR_PAD + dx, y + 20 - BAR_H / 2 - BAR_PAD, a)


# ------------------------------------------------------------------ the big reveal card
BIG_Y, BIG_H = 330, 575


def big_card_bg(canvas, edge):
    cs = card(("big", edge), 840, BIG_H, 44, (34, 30, 92, 238), edge=edge, ew=4)
    R.paste_card(canvas, 60, BIG_Y, cs)


def pill(canvas, text, y, a, sc=1.0):
    sp = scaled(R.pill_sprite(text, 38), sc, a)
    if sp.width > 800:
        sp = sp.resize((800, max(2, int(sp.height * 800 / sp.width))), Image.BICUBIC)
    canvas.paste(sp, (int(CX - sp.width / 2), int(y - sp.height / 2)), sp)


def chip_text(canvas, dr, text, col, y, a):
    f = F("labb", 34)
    w = int(dr.textlength(text, font=f)) + 56
    w = min(w, 800)
    if a < 0.03:
        return
    cs = card(("chip", w), w, 56, 28, (18, 22, 62), edge=col, ew=3, sh_alpha=0, gloss=False)
    R.paste_sub(canvas, cs[0], CX - w / 2 - cs[1], y - 28 - cs[1], a)
    dr.text((CX, y), text, font=f, fill=R.lerpc((20, 24, 70), col, a), anchor="mm")


def history(canvas, dr, iso, a, prog, col):
    """the country's own history line, drawing itself from the first year to the last (phase 2 of a card)"""
    ys = [(YEARS[i], SER[iso][i]) for i in range(NY) if SER[iso][i] > 0]
    if len(ys) < 3:
        return
    x0, x1, yt, yb = 110, 850, 786, 866
    vmin, vmax = min(v for _, v in ys), max(v for _, v in ys)
    span = max(vmax - vmin, vmax * 0.05, 1e-9)
    pts = [(x0 + (x1 - x0) * k / (len(ys) - 1), yb - (yb - yt) * (v - vmin) / span) for k, (_, v) in enumerate(ys)]
    fpos = prog * (len(ys) - 1)
    i = int(fpos)
    head = pts[i] if i >= len(ys) - 1 else (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * (fpos - i), pts[i][1] + (pts[i + 1][1] - pts[i][1]) * (fpos - i))
    vnow = ys[i][1] if i >= len(ys) - 1 else ys[i][1] + (ys[i + 1][1] - ys[i][1]) * (fpos - i)
    ynow = ys[min(i, len(ys) - 1)][0]
    ss = 2
    W, H = 800, 170
    ox, oy = 80, 730
    im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([((x0 - ox) * ss, (yb + 6 - oy) * ss), ((x1 - ox) * ss, (yb + 6 - oy) * ss)], fill=(70, 84, 150, 255), width=2 * ss)
    seg = pts[:i + 1] + [head]
    if len(seg) >= 2:
        poly = [((x - ox) * ss, (y - oy) * ss) for x, y in seg]
        area = poly + [((seg[-1][0] - ox) * ss, (yb + 6 - oy) * ss), ((seg[0][0] - ox) * ss, (yb + 6 - oy) * ss)]
        d.polygon(area, fill=tuple(col) + (46,))
        d.line(poly, fill=GOLD + (255,), width=6 * ss, joint="curve")
    hx, hy = (head[0] - ox) * ss, (head[1] - oy) * ss
    d.ellipse([hx - 16 * ss, hy - 16 * ss, hx + 16 * ss, hy + 16 * ss], fill=GOLD + (70,))
    d.ellipse([hx - 9 * ss, hy - 9 * ss, hx + 9 * ss, hy + 9 * ss], fill=(255, 244, 190, 255), outline=GOLD + (255,), width=3 * ss)
    im = im.resize((W, H), Image.LANCZOS)
    if a < 0.99:
        im.putalpha(im.getchannel("A").point([int(v * a) for v in range(256)]))
    canvas.paste(im, (ox, oy), im)
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (CX, 750), f"{ynow}:  {fmt(vnow)}", "num", fit(dr, f"{ynow}:  {fmt(vnow)}", "num", 40, 780), T.mix(GOLD, a), anchor="mm", shadow=False)
    dr.text((x0, yb + 26), str(ys[0][0]), font=F("lab", 22), fill=T.mix(MUTED, a), anchor="lm")
    dr.text((x1, yb + 26), str(ys[-1][0]), font=F("lab", 22), fill=T.mix(MUTED, a), anchor="rm")


def reveal_card(canvas, dr, iso, rank, value, u, kind="rank", t=0.0, dur=7.5):
    """the BIG card for one country; u = seconds since it appeared"""
    hi = iso == HL
    edge = GREEN if hi else (GOLD if rank == 1 else COL.get(iso, GOLD))
    big_card_bg(canvas, edge)
    a_in = R.ease(u / 0.22)
    # rank number slams in
    k = R.ease_back(u / 0.38)
    sc = 1.3 - 0.3 * k
    rk = f"#{rank}"
    base = fit(dr, rk, "num", 200, 330, 90)  # "#30" must not touch the flag
    fnt = F("num", int(base * sc))
    col = GOLD if rank <= 3 or kind == "pk" else TXT
    if a_in > 0.5:
        dr.text((236 + 4, 520 + 6), rk, font=fnt, fill=T.mix((3, 5, 16), a_in), anchor="mm")
    dr.text((236, 520), rk, font=fnt, fill=T.mix(col, a_in), anchor="mm", stroke_width=3, stroke_fill=T.mix((8, 10, 28), a_in))
    # flag pops in
    fk = R.ease_back((u - 0.08) / 0.45)
    fs, pad = flag_sprite(iso, 400, 270, 22)
    sp = scaled(fs, 0.55 + 0.45 * fk, a_in)
    canvas.paste(sp, (int(640 - sp.width / 2), int(500 - sp.height / 2)), sp)
    if rank == 1 and kind == "rank":
        cr = R.crown_sprite()
        cr = cr.resize((int(cr.width * 1.5), int(cr.height * 1.5)), Image.LANCZOS)
        bob = 6 * math.sin(t * 5)
        canvas.paste(cr, (int(640 - cr.width / 2), int(500 - sp.height / 2 - cr.height + 24 + bob)), cr)
    # name
    nm = NAME[iso].upper() if len(NAME[iso]) < 14 else NAME[iso]
    nsz = fit(dr, nm, "title", 92, 780)
    na = R.ease((u - 0.15) / 0.25)
    T.TX(dr, (CX, 688), nm, "title", nsz, T.mix(GOLD if hi else TXT, na), anchor="mm", stroke=3)
    # full number counts up
    has_hist = iso in SER and sum(1 for x in SER[iso] if x > 0) > 4
    out = R.ease((u - 3.2) / 0.5) if has_hist else 0.0  # phase 2: value + chip fade out, the history line takes over
    vs = value * R.ease((u - 0.2) / 2.2)  # slow, calm count-up
    txt = fmt(vs if u < 2.4 else value)
    vsz = fit(dr, fmt(value), "num", 76, 780)
    if out < 0.97:
        T.TX(dr, (CX, 772), txt, "num", vsz, T.mix(GOLD, R.ease((u - 0.2) / 0.25) * (1 - out)), anchor="mm", stroke=2)
    # story chip
    if kind == "pk":
        chip_text(canvas, dr, f"WHERE IS PAKISTAN? NUMBER {rank}", GREEN, 852, R.ease((u - 0.5) / 0.3) * (1 - out))
    else:
        txt2, c2 = STORY[iso]
        chip_text(canvas, dr, txt2, c2, 852, R.ease((u - 0.65) / 0.3) * (1 - out))
    if has_hist and u > 3.3:
        total_draw = max(2.0, dur - 4.9)
        history(canvas, dr, iso, R.ease((u - 3.4) / 0.5), min(1.0, max(0.0, (u - 4.0) / total_draw)), edge)
    # top band pills
    if kind == "pk" or hi:
        pill(canvas, "PAKISTAN IS HERE!" if kind != "pk" else "YOUR COUNTRY", 368, R.ease((u - 0.1) / 0.3), 0.7 + 0.3 * R.ease_back((u - 0.1) / 0.4))
    elif rank == 1:
        pill(canvas, "THE WINNER!", 368, R.ease((u - 0.1) / 0.3), 0.7 + 0.3 * R.ease_back((u - 0.1) / 0.4))
    # sparkles for the winner
    if rank == 1 and kind == "rank" and u < 3.0:
        rng = np.random.default_rng(7)
        for j in range(34):
            ang = rng.random() * 6.283
            sp_ = 160 + rng.random() * 330
            d = sp_ * R.ease(u / 2.6)
            r_ = 3 + int(rng.random() * 5)
            x, y = 640 + math.cos(ang) * d, 500 + math.sin(ang) * d * 0.8
            dot = R.dot(r_, 2)
            canvas.paste(dot, (int(x - r_), int(y - r_)), dot)


def intro_card(canvas, dr, t):
    big_card_bg(canvas, GOLD)
    a = R.ease(t / 0.3)
    k = R.ease_back(t / 0.5)
    T.TX(dr, (CX, 470), "CAN YOU GUESS", "title", int(70 * (0.7 + 0.3 * k)), T.mix(TXT, a), anchor="mm", stroke=3)
    T.TX(dr, (CX, 580), "THE NUMBER 1?", "title", int(104 * (0.7 + 0.3 * k)), T.mix(GOLD, a), anchor="mm", stroke=3)
    T.TX(dr, (CX, 700), f"Ranking in {Y1}", "labb", 44, T.mix(TXT, R.ease((t - 0.3) / 0.3)), anchor="mm")
    sub = "Is Pakistan in the Top 10? Watch till the end" if PK_RANK else "Watch till the end"
    sub_a = R.ease((t - 0.7) / 0.4)
    T.TX(dr, (CX, 790), sub, "labb", fit(dr, sub, "labb", 38, 780), T.mix(GREEN if PK_RANK else MUTED, sub_a), anchor="mm")


def cta_card(canvas, dr, u):
    big_card_bg(canvas, GOLD)
    a = R.ease(u / 0.3)
    T.TX(dr, (CX, 470), "SAVE THIS REEL", "title", 84, T.mix(GOLD, a), anchor="mm", stroke=3)
    T.TX(dr, (CX, 570), "and send it to a friend", "labb", 50, T.mix(TXT, R.ease((u - 0.2) / 0.3)), anchor="mm")
    T.TX(dr, (CX, 690), "FOLLOW @therankrace", "title", 64, T.mix(TXT, R.ease((u - 0.4) / 0.3)), anchor="mm", stroke=2)
    T.TX(dr, (CX, 790), "A new Top 10 every day", "labb", 40, T.mix(MUTED, R.ease((u - 0.6) / 0.3)), anchor="mm")
    sub = "Which rank surprised you? Comment below"
    T.TX(dr, (CX, 852), sub, "labb", fit(dr, sub, "labb", 36, 780), T.mix(GREEN, R.ease((u - 0.8) / 0.3)), anchor="mm")


# ------------------------------------------------------------------ frame
def source_lines(dr):
    src = CFG["source"].split(" (updated")[0]
    for ssz in (18, 16, 14):
        sfnt = F("small", ssz)
        lines = wrap(dr, src, sfnt, 840, False)
        if len(lines) <= 2:
            break
    return lines[:2], sfnt, ssz


def render_frame(k):
    t = k / FPS
    canvas = T.BGIMG.copy()
    R.particles(canvas, t)
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (CX, 205), T1, "title", fit(dr, T1, "title", 66, 840), TXT, anchor="mm", stroke=3)
    T.TX(dr, (CX, 265), T2, "title", fit(dr, T2, "title", 54, 840), GOLD, anchor="mm")
    # rows: bottom first so that the glow of the top rows stays visible
    for r in range(10, 0, -1):
        draw_row(canvas, dr, r, t)
    # big card
    if t < INTRO:
        intro_card(canvas, dr, t)
    elif t >= T_CTA:
        cta_card(canvas, dr, t - T_CTA)
    elif PK_CARD and t >= T_PK:
        reveal_card(canvas, dr, HL, PK_RANK, PK_VAL, t - T_PK, "pk", t, PK_DUR)
    else:
        r = next(q for q in range(10, 0, -1) if S[q] <= t < S[q] + DUR[q])
        reveal_card(canvas, dr, TOP[r - 1], r, SER[TOP[r - 1]][LAST], t - S[r], "rank", t, DUR[r])
    dr = ImageDraw.Draw(canvas)
    lines, sfnt, ssz = source_lines(dr)
    for q, ln in enumerate(lines):
        dr.text((CX, 1614 + q * (ssz + 5)), ln, font=sfnt, fill=MUTED, anchor="mm")
    prog = t / TOTAL
    dr.rectangle([0, VH - 10, VW, VH], fill=CARD)
    dr.rectangle([0, VH - 10, int(VW * prog), VH], fill=GOLD)
    return canvas.tobytes()


FPS = 60


def init_worker(fps):
    global FPS
    FPS = fps


def render(out_video, work, fps):
    init_worker(fps)
    n = int(round(TOTAL * FPS))
    music = f"{work}/music_countdown.wav"
    from music_chill import make_chill_music  # relaxing ambient music (owner's request), opens up at the #1 reveal
    make_chill_music(TOTAL, S[1], music)
    print("music: chill ambient", round(TOTAL, 1), "s, swell at", round(S[1], 1), flush=True)
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", music,
           "-af", "alimiter=limit=0.9", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", out_video]
    log = open(f"{work}/ffmpeg_countdown.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    ctx = mp.get_context("spawn")
    workers = max(1, min(os.cpu_count() or 2, int(os.environ.get("RENDER_WORKERS", "99"))))  # fewer on a busy laptop
    with ctx.Pool(workers, initializer=init_worker, initargs=(fps,)) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            proc.stdin.write(b)
            if i % 300 == 0:
                print("frame", i, "/", n, flush=True)
    proc.stdin.close()
    proc.wait()
    log.close()
    json.dump({"cover_ms": int(COVER_S * 1000), "total": TOTAL, "format": "countdown"}, open(out_video + ".meta.json", "w"))


def report():
    print(f"countdown {TOTAL:.1f}s  top10 {[NAME[i] for i in TOP]}  Pakistan {'in top 10' if PK_IN else ('#%s' % PK_RANK)}")
    for r in range(10, 0, -1):
        print(f"  #{r:2d} {S[r]:5.1f}s  {NAME[TOP[r - 1]]:18s} {fmt(SER[TOP[r - 1]][LAST]):>26s}  {STORY[TOP[r - 1]][0]}")


if __name__ == "__main__":
    report()
    if len(sys.argv) > 2:
        fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else FPS
        os.makedirs(sys.argv[2], exist_ok=True)
        render(sys.argv[1], sys.argv[2], fps)
