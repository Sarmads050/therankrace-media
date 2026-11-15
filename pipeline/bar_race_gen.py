"""Generic smooth 3D bar race (same look as the Top-10 mobile brands video) driven by a JSON config.

    RACE_CFG=data/race/gdp.json python bar_race_gen.py OUT.mp4 WORKDIR [--fps 60]

Config (made by race_topics.py): years, entities[{iso,name}], series{iso:[value per year]}, seg [[m_end, speed]], captions, pills,
hits (music), title1/title2, fmt, source, highlight.  Everything below is the engine of bar_race_smooth.py made generic:
  * one continuous time -> year curve (monotone cubic), never stops, slows down near the big moments
  * Catmull-Rom between yearly values, critically damped spring rank positions, anti-aliased 3D capsule bars, flags
  * 60 fps, all CPU cores
"""
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bar_race_smooth as R  # noqa  (sprites, cards, particles)
import mobile_stats_3d_en as T  # noqa
from common import FFMPEG, wrap  # noqa
from flags import flag  # noqa
import top10_video as V  # noqa  (flag badge sprites)

VW, VH = 1080, 1920
F, TXT, MUTED, GOLD, CARD = T.F, T.TXT, T.MUTED, T.GOLD, R.CARD
BAR_H, BAR_PAD = R.BAR_H, R.BAR_PAD
FPS = int(os.environ.get("BARRACE_FPS", "60"))
X0, MAXLEN, PITCH, YTOP = 250, 610, 106, 408
TL_Y = 1546

CFG = json.load(open(os.environ["RACE_CFG"], encoding="utf-8"))
YEARS = CFG["years"]
N = len(YEARS)
ENT = CFG["entities"]
B = len(ENT)
SER = [np.array([float(x or 0.0) for x in CFG["series"][e["iso"]]]) for e in ENT]
HIGH = CFG.get("highlight")

# ------------------------------------------------------------------ time curve
TS, MS = [0.0], [0.0]
for _m, _sp in CFG["seg"]:
    MS.append(float(_m))
    TS.append(TS[-1] + (MS[-1] - MS[-2]) / _sp)
T_MOTION = TS[-1]
TAIL = CFG.get("tail", 9.0)
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
    m[-1] = 0.0
    return h, m


_H, _MT = _tangents()


def m_at(t):
    if t >= TS[-1]:
        return MS[-1]
    k = max(0, min(len(TS) - 2, int(np.searchsorted(TS, t, side="right")) - 1))
    s = (t - TS[k]) / _H[k]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * MS[k] + (s3 - 2 * s2 + s) * _H[k] * _MT[k] + (-2 * s3 + 3 * s2) * MS[k + 1]
            + (s3 - s2) * _H[k] * _MT[k + 1])


_TG = np.arange(0.0, T_MOTION + 0.005, 0.01)
_MG = np.array([m_at(t) for t in _TG])


def t_of(year):
    m = year - YEARS[0]
    return float(_TG[min(len(_TG) - 1, int(np.searchsorted(_MG, m - 1e-6)))])


def catmull(arr, m):
    i = int(math.floor(m))
    f = m - i
    p0, p1, p2, p3 = arr[max(i - 1, 0)], arr[min(i, N - 1)], arr[min(i + 1, N - 1)], arr[min(i + 2, N - 1)]
    v = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    return max(float(v), 0.0)


def when(x):
    """caption/pill time spec: number of years -> t_of(year); {'tail': s} -> T_MOTION + s"""
    return T_MOTION + x["tail"] if isinstance(x, dict) else (0.0 if x == YEARS[0] else t_of(x))


CAPS = [(when(c["at"]), c["text"]) for c in CFG["captions"]]
EVENTS = [(when(e["at"]), when(e["at"]) + e["dur"], e["text"]) for e in CFG["pills"]]
HITS = [t_of(y) for y in CFG["hits"]]


def fmt(v):
    k = CFG["fmt"]
    if k == "count":
        return f"{v / 1e9:.2f} B" if v >= 1e9 else f"{v / 1e6:.0f} M"
    if k == "usd":
        return f"${v / 1e12:.1f}T" if v >= 1e12 else (f"${v / 1e9:.0f}B" if v >= 1e10 else f"${v / 1e9:.1f}B")
    return f"{v:,.0f}"


# ------------------------------------------------------------------ colours (from the flag, with overrides)
OVER = {"USA": (70, 120, 245), "CHN": (238, 62, 52), "IND": (255, 150, 40), "PAK": (36, 170, 96), "JPN": (255, 110, 150),
        "DEU": (250, 200, 40), "GBR": (120, 100, 235), "FRA": (60, 150, 255), "ITA": (50, 200, 120), "RUS": (190, 90, 235),
        "BRA": (70, 205, 95), "CAN": (250, 90, 90), "IDN": (255, 85, 85), "NGA": (60, 190, 120), "BGD": (30, 160, 110),
        "ETH": (250, 205, 50), "MEX": (60, 180, 110), "ESP": (255, 165, 40), "KOR": (80, 160, 255), "AUS": (60, 200, 215),
        "NLD": (255, 140, 60), "ARG": (110, 195, 250)}


def color_of(iso):
    if iso in OVER:
        return OVER[iso]
    fl = flag(iso)
    if fl is None:
        return (140, 150, 190)
    sm = fl.convert("RGB").resize((24, 16))
    best, bs = (140, 150, 190), -1
    for c, n in sm.getcolors(500):
        mx, mn = max(c), min(c)
        sat = (mx - mn) / max(mx, 1)
        if mx > 90 and sat > 0.35 and n * sat > bs:
            best, bs = c, n * sat
    return best


COLS = [color_of(e["iso"]) for e in ENT]


# ------------------------------------------------------------------ spring-smoothed rank positions (computed once)
P = AX = None


def simulate():
    nf = int(round(TOTAL * FPS)) + 1
    vals0 = np.array([catmull(s, 0.0) for s in SER])
    order = list(np.argsort(-vals0, kind="stable"))
    pos = np.zeros(B)
    for r, i in enumerate(order):
        pos[i] = r
    vel = np.zeros(B)
    ax = float(vals0.max())
    P_ = np.zeros((nf, B), np.float32)
    AX_ = np.zeros(nf, np.float32)
    sub, w = 4, 9.0
    dt = 1.0 / (FPS * sub)
    for f in range(nf):
        m = m_at(f / FPS)
        vals = np.array([catmull(s, m) for s in SER])
        changed = True
        while changed:
            changed = False
            for i in range(B - 1):
                a_, b_ = order[i], order[i + 1]
                if vals[b_] > vals[a_] * 1.004:
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
        ax += (float(vals.max()) - ax) * (1 - math.exp(-1.0 / (FPS * 0.5)))
        AX_[f] = ax
    return P_, AX_


# ------------------------------------------------------------------ frame
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
    for ev in CFG["pills"]:
        k = ev["at"] - YEARS[0] if not isinstance(ev["at"], dict) else None
        if k is None:
            continue
        c = (255, 120, 90) if m >= k else (90, 100, 150)
        d.ellipse([X(k) - 9 * ss, cy - 9 * ss, X(k) + 9 * ss, cy + 9 * ss], fill=c + (255,), outline=(255, 255, 255, 255),
                  width=2 * ss)
    mx = X(m)
    d.ellipse([mx - 26 * ss, cy - 26 * ss, mx + 26 * ss, cy + 26 * ss], fill=GOLD + (60,))
    d.ellipse([mx - 16 * ss, cy - 16 * ss, mx + 16 * ss, cy + 16 * ss], fill=GOLD + (255,), outline=(255, 244, 190, 255),
              width=3 * ss)
    d.ellipse([mx - 9 * ss, cy - 11 * ss, mx - 1 * ss, cy - 4 * ss], fill=(255, 255, 255, 190))
    im = im.resize((bw, bh), Image.LANCZOS)
    canvas.paste(im, (bx, by), im)


def static_layer(canvas):
    cards = R.bake_cards()
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (VW / 2, 84), CFG["title1"], "title", 72, TXT, anchor="mm", stroke=3)
    T.TX(dr, (VW / 2, 156), CFG["title2"], "title", 56, GOLD, anchor="mm")
    R.paste_card(canvas, 290, 218, cards["date"])
    R.paste_card(canvas, 34, YTOP - 56, cards["band"])
    R.paste_card(canvas, 50, TL_Y - 56, cards["timeline"])
    R.paste_card(canvas, 50, 1650, cards["caption"])
    dr = ImageDraw.Draw(canvas)
    for yr in range(YEARS[0], YEARS[-1] + 1, 10):
        dr.text((90 + 900 * (yr - YEARS[0]) / (N - 1), TL_Y + 36), str(yr), font=F("lab", 24), fill=MUTED, anchor="mm")
    dr.text((90 + 900, TL_Y + 36), str(YEARS[-1]), font=F("lab", 24), fill=MUTED, anchor="mm")
    dr.text((VW / 2, 1836), "@therankrace", font=F("labb", 28), fill=GOLD, anchor="mm")
    dr.text((VW / 2, 1874), CFG["source"], font=F("small", 19), fill=MUTED, anchor="mm")
    return dr


def render_frame(k):
    t = k / FPS
    m = m_at(t)
    canvas = T.BGIMG.copy()
    R.particles(canvas, t)
    static_layer(canvas)
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (VW / 2, 272), str(YEARS[min(N - 1, int(round(m)))]), "num", 80, GOLD, anchor="mm", stroke=2)
    pk, ax = P[k], float(AX[k])
    vals = [catmull(s, m) for s in SER]
    for i in sorted(range(B), key=lambda j: -pk[j]):
        e, v, p = ENT[i], vals[i], float(pk[i])
        a = max(0.0, min(1.0, (v / ax - 0.004) / 0.012)) * max(0.0, min(1.0, 9.9 - p))
        if a < 0.02:
            continue
        y = YTOP + p * PITCH
        hi = e["iso"] == HIGH
        L = min(max(BAR_H * 1.05, v / ax * MAXLEN), MAXLEN * 1.05)
        top = y + 10 - BAR_H / 2 - BAR_PAD
        iy = int(math.floor(top))
        shine = (t * 0.42 + i * 0.19) % 1.6 - 0.3
        spr = R.bar_sprite(L, COLS[i], top - iy, shine, a)
        canvas.paste(spr, (X0 - BAR_PAD, iy), spr)
        im, pad = V.flag_badge(e["iso"], hi)
        R.paste_sub(canvas, im, 100 - pad, y - 38 - pad + 6, a)
        T.TX(dr, (X0 + 4, y - 28), e["name"], "labb", 30, T.mix(GOLD if hi else TXT, a), anchor="lm")
        T.TX(dr, (X0 + L + 22, y + 10), fmt(v), "num", 44, T.mix(TXT, a), anchor="lm")
        rank = int(round(p)) + 1
        if rank in R.MEDAL:
            sp = T.sphere_sprite(R.MEDAL[rank], 26)
            R.paste_sub(canvas, sp, 58 - 26, y - 26 + 6, a)
            dr.text((58, y + 7), str(rank), font=F("num", 30), fill=T.mix((60, 36, 0), a), anchor="mm")
        else:
            R.paste_sub(canvas, R.chip(), 58 - 25, y - 25 + 6, a)
            dr.text((58, y + 7), str(min(rank, 10)), font=F("num", 26), fill=T.mix(MUTED, a), anchor="mm")
        lead = max(0.0, 1.0 - p / 0.5)
        if lead > 0.03:
            R.paste_sub(canvas, R.crown_sprite(), 100 + 24, y - 60, lead * a)

    timeline_dyn(canvas, m)
    for t0, t1, text in EVENTS:
        if t0 <= t < t1:
            a = R.alpha_window(t, t0, t1, 0.3)
            sc = 0.6 + 0.4 * R.ease_back((t - t0) / 0.45)
            spr = R.pill_sprite(text)
            sp2 = spr.resize((max(2, int(spr.width * sc)), max(2, int(spr.height * sc))), Image.BICUBIC)
            if a < 0.99:
                sp2.putalpha(sp2.getchannel("A").point([int(v * a) for v in range(256)]))
            canvas.paste(sp2, (int(VW / 2 - sp2.width / 2), int(1450 - sp2.height / 2)), sp2)
    dr = ImageDraw.Draw(canvas)
    for j, (c0, text) in enumerate(CAPS):
        c1 = CAPS[j + 1][0] if j + 1 < len(CAPS) else TOTAL
        if c0 <= t < c1:
            a = R.alpha_window(t, c0, c1, 0.5) if j else min(R.ease(t / 0.8), R.ease((c1 - t) / 0.5))
            fnt = F("cap", 36)
            lines = wrap(dr, text, fnt, 880, False)
            for q, ln in enumerate(lines):
                yy = 1725 + (q - (len(lines) - 1) / 2) * 52
                dr.text((VW / 2, yy), ln, font=fnt, fill=R.lerpc(CARD, TXT, a), anchor="mm")
    prog = t / TOTAL
    dr.rectangle([0, VH - 10, VW, VH], fill=CARD)
    dr.rectangle([0, VH - 10, int(VW * prog), VH], fill=GOLD)
    return canvas.tobytes()


def init_worker(fps):
    global FPS, P, AX
    FPS = fps
    P, AX = simulate()


def render(out_video, work, fps):
    init_worker(fps)
    n = int(round(TOTAL * FPS))
    music = f"{work}/music_race_gen.wav"
    from music_race import make_race_music
    bounds = [0.0] + [round(h, 3) for h in HITS] + [TOTAL]
    bpms = [102, 111, 116, 120, 116]
    bars = [max(3, round((bounds[i + 1] - bounds[i]) * bpms[i] / 240.0)) for i in range(5)]
    print("music bounds", bounds, "bars", bars, "tempos", [round(bars[i] * 240 / (bounds[i + 1] - bounds[i])) for i in range(5)],
          flush=True)
    make_race_music(bounds, bars, music)
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", music,
           "-af", "volume=0.5,alimiter=limit=0.8", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", out_video]
    log = open(f"{work}/ffmpeg_race_gen.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, os.cpu_count() or 2), initializer=init_worker, initargs=(fps,)) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            proc.stdin.write(b)
            if i % 600 == 0:
                print("frame", i, "/", n, flush=True)
    proc.stdin.close()
    proc.wait()
    log.close()


def report():
    print(f"motion {T_MOTION:.1f}s  total {TOTAL:.1f}s  entities {B}  years {YEARS[0]}-{YEARS[-1]}")
    for j, (c0, text) in enumerate(CAPS):
        c1 = CAPS[j + 1][0] if j + 1 < len(CAPS) else TOTAL
        print(f"  caption {c0:6.1f}s +{c1 - c0:4.1f}s  {text}")
    for t0, t1, text in EVENTS:
        print(f"  pill    {t0:6.1f}s +{t1 - t0:4.1f}s  {text}")
    print("  hits", [round(h, 1) for h in HITS])


if __name__ == "__main__":
    report()
    if len(sys.argv) > 2:
        fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else FPS
        os.makedirs(sys.argv[2], exist_ok=True)
        render(sys.argv[1], sys.argv[2], fps)
        print("done", sys.argv[1])
