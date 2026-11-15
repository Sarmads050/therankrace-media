"""Generic "Top 10" countdown Reel (1080x1920, 60 fps, ~50 s, music, no voice-over).

    python top10_video.py CONFIG.json OUT.mp4 WORKDIR [--fps 60]

CONFIG comes from topics.py (title, 10 items with iso/name/value, per-rank captions, pills, source).
Countdown 10 -> 1: each rank's bar grows with a count-up, its flag pops in, the row flashes. Smooth 3D capsule bars, soft shadows.
Timeline is fixed so the music (made for exactly these section lengths) always lands on the big moments.
"""
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bar_race_smooth as R  # noqa  (3D bar sprite, cards, pills, particles)
import mobile_stats_3d_en as T  # noqa
from common import FFMPEG, wrap  # noqa
from flags import flag  # noqa

VW, VH = 1080, 1920
F, TXT, MUTED, GOLD, CARD = T.F, T.TXT, T.MUTED, T.GOLD, R.CARD
PAL = [(255, 200, 40), (255, 150, 50), (255, 100, 90), (235, 90, 170), (170, 100, 240),
       (110, 120, 255), (60, 160, 255), (50, 205, 235), (50, 215, 170), (110, 225, 110)]
X0, MAXLEN, PITCH, YTOP = 250, 610, 124, 360
BAR_H, BAR_PAD = R.BAR_H, R.BAR_PAD

# fixed timeline (seconds). Music sections: [0,9.6] [9.6,19.6] [19.6,31.6] [31.6,37.6] [37.6,49.6]
START = {10: 2.2, 9: 5.9, 8: 9.6, 7: 13.0, 6: 16.3, 5: 19.6, 4: 23.3, 3: 27.3, 2: 31.6, 1: 37.6}
OUTRO = 44.0
TOTAL = 49.6
BOUNDS = [0.0, 9.6, 19.6, 31.6, 37.6, TOTAL]
BARS = [4, 5, 6, 3, 6]

CFG = None
FPS = 60


def load(path):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    it = c["items"]
    assert len(it) == 10 and [x["rank"] for x in it] == list(range(1, 11))
    assert all(it[i]["value"] >= it[i + 1]["value"] for i in range(9)), "items must be sorted"
    assert set(c["captions"]) == {str(i) for i in range(1, 11)} or set(c["captions"]) == set(range(1, 11))
    c["captions"] = {int(k): v for k, v in c["captions"].items()}
    c["pills"] = {int(k): v for k, v in c.get("pills", {}).items()}
    return c


def fmt(value, kind, ref):
    """ref = final value of that row so the unit never flips while counting up."""
    if kind == "count":
        return f"{value / 1e9:.2f} B" if ref >= 1e9 else f"{value / 1e6:.0f} M"
    if kind == "usd":
        return f"${value / 1e12:.1f}T" if ref >= 1e12 else f"${value / 1e9:.0f}B"
    return f"{value:,.0f}"


def row_y(rank):
    return YTOP + (rank - 1) * PITCH


# ------------------------------------------------------------------ sprites
_cache = {}


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def flag_badge(iso, hi=False):
    def build():
        ss, pad, w, h = 3, 16, 112, 76
        W, H = w + 2 * pad, h + 2 * pad
        im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
        from PIL import ImageFilter
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([(pad + 3) * ss, (pad + 8) * ss, (pad + w + 3) * ss, (pad + h + 8) * ss],
                                             radius=16 * ss, fill=(2, 4, 14, 175))
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5 * ss)))
        d = ImageDraw.Draw(im)
        edge = GOLD if hi else (255, 255, 255)
        d.rounded_rectangle([pad * ss, pad * ss, (pad + w) * ss, (pad + h) * ss], radius=16 * ss, fill=edge + (255,))
        inset = 5
        iw, ih = (w - 2 * inset) * ss, (h - 2 * inset) * ss
        fl = flag(iso)
        if fl is not None:
            sc = max(iw / fl.width, ih / fl.height)
            f2 = fl.resize((max(1, int(fl.width * sc)), max(1, int(fl.height * sc))), Image.LANCZOS)
            l, t = (f2.width - iw) // 2, (f2.height - ih) // 2
            f2 = f2.crop((l, t, l + iw, t + ih))
        else:
            f2 = Image.new("RGBA", (iw, ih), (90, 100, 140, 255))
            ImageDraw.Draw(f2).text((iw // 2, ih // 2), iso, font=F("num", 30 * ss), fill=(255, 255, 255, 255), anchor="mm")
        mask = Image.new("L", (iw, ih), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, iw - 1, ih - 1], radius=12 * ss, fill=255)
        im.paste(f2, ((pad + inset) * ss, (pad + inset) * ss), mask)
        return im.resize((W, H), Image.LANCZOS), pad
    return cached(("flag", iso, hi), build)


def unknown_badge():
    def build():
        ss, pad, w, h = 3, 16, 112, 76
        W, H = w + 2 * pad, h + 2 * pad
        im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([pad * ss, pad * ss, (pad + w) * ss, (pad + h) * ss], radius=16 * ss, fill=(34, 42, 96, 255),
                            outline=(70, 84, 150, 255), width=3 * ss)
        d.text((W * ss // 2, H * ss // 2 + 2 * ss), "?", font=F("num", 50 * ss), fill=(110, 126, 190, 255), anchor="mm")
        return im.resize((W, H), Image.LANCZOS), pad
    return cached("unk", build)


def track():
    return cached("track", lambda: R.bar_sprite(MAXLEN, (36, 44, 100), 0.0, -2.0, 0.5))


def paste_scaled(canvas, spr, cx, cy, scale, a):
    im, pad = spr
    w, h = max(2, int(im.width * scale)), max(2, int(im.height * scale))
    im2 = im.resize((w, h), Image.BICUBIC)
    if a < 0.99:
        im2.putalpha(im2.getchannel("A").point([int(v * a) for v in range(256)]))
    canvas.paste(im2, (int(cx - w / 2), int(cy - h / 2)), im2)


# ------------------------------------------------------------------ frame
def render_frame(k):
    t = k / FPS
    c = CFG
    canvas = T.BGIMG.copy()
    R.particles(canvas, t)
    cards = R.bake_cards()
    dr = ImageDraw.Draw(canvas)
    T.TX(dr, (VW / 2, 84), c["title1"], "title", 72, TXT, anchor="mm", stroke=3)
    T.TX(dr, (VW / 2, 156), c["title2"], "title", 60, GOLD, anchor="mm")
    R.paste_card(canvas, VW / 2 - 270, 205, cached("yearcard", lambda: R.soft_card(540, 80, 30, (46, 36, 100), edge=GOLD)))
    R.paste_card(canvas, 50, 1650, cards["caption"])
    dr = ImageDraw.Draw(canvas)
    dr.text((VW / 2, 245), f"WORLD BANK DATA  {c['year']}", font=F("labb", 34), fill=GOLD, anchor="mm")

    items = {x["rank"]: x for x in c["items"]}
    # empty slots first
    for r in range(1, 11):
        y = row_y(r)
        st = START[r]
        if t < st + 0.05:
            R.paste_sub(canvas, track(), X0 - BAR_PAD, y + 8 - BAR_H / 2 - BAR_PAD, 1.0)
            canvas.paste(R.chip(), (58 - 25, int(y - 25)), R.chip())
            dr.text((58, y + 1), str(r), font=F("num", 26), fill=(120, 134, 190), anchor="mm")
            ub, pad = unknown_badge()
            canvas.paste(ub, (100 - pad, int(y - 38 - pad + 0)), ub)
    dr = ImageDraw.Draw(canvas)

    # revealed rows (draw from rank 10 up so higher ranks sit on top)
    for r in range(10, 0, -1):
        st = START[r]
        if t < st:
            continue
        it = items[r]
        y = row_y(r)
        col = PAL[r - 1]
        hi = it["iso"] == c.get("highlight")
        dt = t - st
        # row flash
        fl_a = max(0.0, 1.0 - dt / 1.3) * 0.9
        if fl_a > 0.02:
            R.paste_sub(canvas, cards["band"][0], 34 - cards["band"][1], y - 56 - cards["band"][1], fl_a)
        # track under the growing bar
        R.paste_sub(canvas, track(), X0 - BAR_PAD, y + 8 - BAR_H / 2 - BAR_PAD, 0.8)
        pb = R.ease((dt - 0.12) / 1.25)
        ref = it["value"]
        frac = it["value"] / c["items"][0]["value"]
        L = max(BAR_H * 1.05, frac * MAXLEN * pb)
        top = y + 8 - BAR_H / 2 - BAR_PAD
        iy = int(math.floor(top))
        shine = (t * 0.42 + r * 0.19) % 1.6 - 0.3
        a_bar = min(1.0, dt / 0.15)
        spr = R.bar_sprite(L, col, top - iy, shine, a_bar)
        canvas.paste(spr, (X0 - BAR_PAD, iy), spr)
        # badge pop
        sc = 0.35 + 0.65 * R.ease_back(dt / 0.55)
        paste_scaled(canvas, flag_badge(it["iso"], hi), 100 + 56, y, sc, min(1.0, dt / 0.2))
        # rank chip / medal
        if r <= 3:
            sp = T.sphere_sprite(R.MEDAL[r], 26)
            R.paste_sub(canvas, sp, 58 - 26, y - 26, 1.0)
            dr.text((58, y + 1), str(r), font=F("num", 30), fill=(60, 36, 0), anchor="mm")
        else:
            canvas.paste(R.chip(), (58 - 25, int(y - 25)), R.chip())
            dr.text((58, y + 1), str(r), font=F("num", 26), fill=TXT, anchor="mm")
        na = min(1.0, max(0.0, (dt - 0.15) / 0.3))
        T.TX(dr, (X0 + 4, y - 32), it["name"], "labb", 32, T.mix(GOLD if hi else TXT, na), anchor="lm")
        T.TX(dr, (X0 + L + 22, y + 8), fmt(it["value"] * pb, c["fmt"], ref), "num", 44, T.mix(TXT, a_bar), anchor="lm")
        if r == 1 and dt > 0.8:
            cs = 0.6 + 0.4 * R.ease_back((dt - 0.8) / 0.5)
            cr = R.crown_sprite()
            cw, ch = int(cr.width * cs), int(cr.height * cs)
            c2 = cr.resize((cw, ch), Image.BICUBIC)
            canvas.paste(c2, (100 + 56 - cw // 2, int(y - 38 - ch - 2 + 6 * math.sin(t * 3))), c2)
    dr = ImageDraw.Draw(canvas)

    # pills
    spots = [(START[0 + r] + 0.35, (OUTRO if r == 1 else START[r] + 3.0), txt) for r, txt in c["pills"].items()]
    if t < START[10]:
        spots.append((0.3, START[10], "GUESS THE #1!"))
    if t >= OUTRO:
        spots.append((OUTRO + 0.2, TOTAL, "@therankrace  |  NEW TOP 10 DAILY"))
    for t0, t1, text in spots:
        if t0 <= t < t1:
            a = R.alpha_window(t, t0, t1, 0.3)
            sc = 0.6 + 0.4 * R.ease_back((t - t0) / 0.45)
            sp = R.pill_sprite(text)
            sp2 = sp.resize((max(2, int(sp.width * sc)), max(2, int(sp.height * sc))), Image.BICUBIC)
            if a < 0.99:
                sp2.putalpha(sp2.getchannel("A").point([int(v * a) for v in range(256)]))
            canvas.paste(sp2, (int(VW / 2 - sp2.width / 2), int(1582 - sp2.height / 2)), sp2)
    dr = ImageDraw.Draw(canvas)

    # caption
    caps = [(0.0, START[10], c["intro"])] + [(START[r] + 0.45, (START[r - 1] if r > 1 else OUTRO), c["captions"][r])
                                              for r in range(10, 0, -1)] + [(OUTRO, TOTAL, c["outro"])]
    for j, (c0, c1, text) in enumerate(caps):
        if c0 <= t < c1:
            a = R.alpha_window(t, c0, c1, 0.4) if j else min(R.ease(t / 0.6), R.ease((c1 - t) / 0.4))
            fnt = F("cap", 36)
            lines = wrap(dr, text, fnt, 880, False)
            for q, ln in enumerate(lines):
                yy = 1725 + (q - (len(lines) - 1) / 2) * 52
                dr.text((VW / 2, yy), ln, font=fnt, fill=R.lerpc(CARD, TXT, a), anchor="mm")
    dr.text((VW / 2, 1836), "@therankrace", font=F("labb", 28), fill=GOLD, anchor="mm")
    dr.text((VW / 2, 1874), c["source"], font=F("small", 19), fill=MUTED, anchor="mm")
    prog = t / TOTAL
    dr.rectangle([0, VH - 10, VW, VH], fill=CARD)
    dr.rectangle([0, VH - 10, int(VW * prog), VH], fill=GOLD)
    return canvas.tobytes()


def init_worker(cfg_path, fps):
    global CFG, FPS
    CFG = load(cfg_path)
    FPS = fps


def render(cfg_path, out_video, work, fps):
    init_worker(cfg_path, fps)
    n = int(round(TOTAL * FPS))
    music = f"{work}/music_top10.wav"
    from music_race import make_race_music
    make_race_music(BOUNDS, BARS, music)
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", music,
           "-af", "volume=0.5,alimiter=limit=0.8", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", out_video]
    log = open(f"{work}/ffmpeg_top10.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, os.cpu_count() or 2), initializer=init_worker, initargs=(cfg_path, fps)) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            proc.stdin.write(b)
            if i % 600 == 0:
                print("frame", i, "/", n, flush=True)
    proc.stdin.close()
    proc.wait()
    log.close()


if __name__ == "__main__":
    assert TOTAL <= 119.5
    fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 60
    os.makedirs(sys.argv[3], exist_ok=True)
    render(sys.argv[1], sys.argv[2], sys.argv[3], fps)
    print("done", sys.argv[2])
