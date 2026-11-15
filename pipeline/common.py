import asyncio
import os
import subprocess
import wave

import arabic_reshaper
import edge_tts
import imageio_ffmpeg
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FONT_DIR = "C:/Windows/Fonts/"
_font_cache = {}
_reshaper = arabic_reshaper.ArabicReshaper()


def font(name, size):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_DIR + name, size)
    return _font_cache[key]


def ur(text):
    return get_display(_reshaper.reshape(text))


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def wrap(draw, text, fnt, maxw, rtl):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        shown = ur(trial) if rtl else trial
        if draw.textlength(shown, font=fnt) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def tts(text, voice, path, rate="+0%"):
    async def go():
        await edge_tts.Communicate(text, voice, rate=rate).save(path)

    asyncio.run(go())


def to_wav(mp3, wav, pad):
    subprocess.run(
        [FFMPEG, "-y", "-i", mp3, "-af", f"apad=pad_dur={pad}", "-ar", "44100", "-ac", "1", wav],
        check=True, capture_output=True)


def wav_len(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def concat_wavs(paths, out):
    with wave.open(paths[0]) as w0:
        params = w0.getparams()
    with wave.open(out, "wb") as o:
        o.setparams(params)
        for p in paths:
            with wave.open(p) as w:
                o.writeframes(w.readframes(w.getnframes()))


def build(scenes, voice, out, workdir, bg, rate="+0%", pad=0.6, overlay=None, size=None):
    fw, fh = size or (W, H)
    os.makedirs(workdir, exist_ok=True)
    wavs, durs = [], []
    for i, sc in enumerate(scenes):
        mp3 = f"{workdir}/s{i}.mp3"
        wav = f"{workdir}/s{i}.wav"
        if not os.path.exists(mp3):
            tts(sc["say"], voice, mp3, rate)
        to_wav(mp3, wav, pad)
        wavs.append(wav)
        durs.append(wav_len(wav))
    full = f"{workdir}/full.wav"
    concat_wavs(wavs, full)
    total = sum(durs)
    log = open(f"{workdir}/ffmpeg.log", "wb")
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}", "-r", str(FPS),
           "-i", "-", "-i", full, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
           "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    done = 0.0
    for sc, d in zip(scenes, durs):
        n = max(1, round(d * FPS))
        for k in range(n):
            t = k / max(1, n - 1)
            s = k / FPS
            img = Image.new("RGB", (fw, fh), bg)
            dr = ImageDraw.Draw(img)
            sc["draw"](img, dr, t, s, d)
            if overlay:
                overlay(img, dr, (done + s) / total)
            proc.stdin.write(img.tobytes())
        done += d
    proc.stdin.close()
    proc.wait()
    log.close()
    return total
