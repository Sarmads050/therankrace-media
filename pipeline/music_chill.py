"""Relaxing ambient music (no drums): soft pads + a slow pluck arpeggio + deep sub, with a room reverb. Pure numpy, no samples.

    make_chill_music(total_seconds, swell_at, out_wav)

swell_at = time of the big reveal (#1): the pad opens up a little and a few bell notes sing on top.
"""
import wave

import numpy as np

SR = 44100
PROG = [  # (root midi, chord tones as semitone offsets)  Cmaj7 - Am7 - Fmaj7 - G6   (slow, warm, hopeful)
    (48, [0, 4, 7, 11]), (45, [0, 3, 7, 10]), (53, [0, 4, 7, 11]), (55, [0, 4, 7, 9])]
BPM = 66.0


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _env_pad(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = min(n // 2, int(a * SR)), min(n // 2, int(r * SR))
    e[:na] = np.linspace(0, 1, na) ** 2
    e[n - nr:] = np.linspace(1, 0, nr) ** 2
    return e


def _tone(freq, n, harm=((1, 1.0), (2, 0.18), (3, 0.05))):
    t = np.arange(n, dtype=np.float32) / SR
    y = np.zeros(n, np.float32)
    for h, a in harm:
        y += a * np.sin(2 * np.pi * freq * h * t)
    return y


def _reverb(x, seconds=2.4, wet=0.33, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (0.42 * SR)).astype(np.float32)
    ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # soft attack
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    X, H = np.fft.rfft(x, size), np.fft.rfft(ir, size)
    y = np.fft.irfft(X * H, size)[:len(x)].astype(np.float32)
    return (1 - wet) * x + wet * y * 1.0


def make_chill_music(total, swell_at, out_wav, seed=11):
    n = int(total * SR)
    rng = np.random.default_rng(seed)
    L = np.zeros(n, np.float32)
    Rr = np.zeros(n, np.float32)
    beat = 60.0 / BPM
    chord_len = 8 * beat  # two bars per chord
    nch = int(np.ceil(total / chord_len)) + 1
    for c in range(nch):
        root, tones = PROG[c % 4]
        t0 = c * chord_len
        i0 = int(t0 * SR)
        dur = chord_len + 2.5  # overlap into the next chord (long release)
        nn = min(int(dur * SR), n - i0)
        if nn <= 0:
            continue
        env = _env_pad(int(dur * SR), 2.0, 2.5)[:nn]
        swell = 1.0 + 0.35 * np.clip((np.arange(nn) / SR + t0 - swell_at + 2.0) / 3.0, 0, 1) * (1.0 if t0 + 1 >= swell_at - 8 else 0.0)
        pad = np.zeros(nn, np.float32)
        for k, off in enumerate(tones):
            for det in (-0.12, 0.12):  # slow chorus: two slightly detuned voices per note
                pad += _tone(mtof(root + 12 + off) * 2 ** (det / 12.0), nn, ((1, 1.0), (2, 0.12))) * 0.07
            pad += _tone(mtof(root + 24 + off), nn, ((1, 1.0),)) * 0.022
        sub = _tone(mtof(root - 12), nn, ((1, 1.0), (2, 0.1))) * 0.20  # deep sub note
        wob = 0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * np.arange(nn) / SR + c)
        sig = (pad * wob + sub) * env * swell
        L[i0:i0 + nn] += sig * 0.95
        Rr[i0:i0 + nn] += sig * 0.95
        # slow plucks: one note per beat, soft up-and-down arpeggio
        pat = [0, 2, 1, 3, 2, 1, 3, 2]
        for b in range(8):
            tb = t0 + b * beat
            if tb >= total - 0.5:
                break
            f = mtof(root + 24 + tones[pat[b]])
            ln = int(1.6 * SR)
            ln = min(ln, n - int(tb * SR))
            if ln <= 0:
                continue
            tt = np.arange(ln, dtype=np.float32) / SR
            y = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt * 3.2) * (1 - np.exp(-tt * 400))
            y *= 0.075 * (0.85 + 0.3 * rng.random())
            pan = 0.35 + 0.3 * ((b % 2) * 0.9 + rng.random() * 0.1)
            j = int(tb * SR)
            L[j:j + ln] += y * (1 - pan)
            Rr[j:j + ln] += y * pan
    # bell notes when the winner is revealed
    for q, (off, dt) in enumerate(((12, 0.0), (19, 0.9), (16, 1.8), (24, 2.9))):
        j = int((swell_at + dt) * SR)
        if j >= n:
            continue
        ln = min(int(3.2 * SR), n - j)
        tt = np.arange(ln, dtype=np.float32) / SR
        f = mtof(72 + off - 12)
        y = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2.76 * f * tt) * np.exp(-tt * 4)) * np.exp(-tt * 1.4) * 0.09
        L[j:j + ln] += y * 0.6
        Rr[j:j + ln] += y * 0.8
    L, Rr = _reverb(L, seed=3), _reverb(Rr, seed=4)
    fade_in, fade_out = int(2.0 * SR), int(4.0 * SR)
    for ch in (L, Rr):
        ch[:fade_in] *= np.linspace(0, 1, fade_in)
        ch[n - fade_out:] *= np.linspace(1, 0, fade_out)
    st = np.stack([L, Rr], 1)
    rms = float(np.sqrt((st[int(3 * SR):] ** 2).mean())) + 1e-9
    st *= 0.115 / rms  # about -19 dBFS RMS: calm but clearly audible
    st = np.tanh(st * 1.1) / 1.1  # soft limiter
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out_wav


if __name__ == "__main__":
    import sys
    make_chill_music(float(sys.argv[1]) if len(sys.argv) > 1 else 95.0, float(sys.argv[2]) if len(sys.argv) > 2 else 60.0,
                     sys.argv[3] if len(sys.argv) > 3 else "chill.wav")
