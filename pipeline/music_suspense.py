"""Original suspense/thriller background music, synthesised from scratch with numpy (no samples, no third-party audio).

make_music(duration_s, hits, out_wav): hits = list of times (s) where an impact + riser should land.
"""
import wave

import numpy as np

SR = 44100


def _add(buf, start, sig):
    i = int(start * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def _lowpass(x, k):
    # cheap moving-average low-pass
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    return np.concatenate([np.zeros(k - 1), y])


def make_music(duration, hits, out_wav, seed=7):
    rng = np.random.default_rng(seed)
    N = int((duration + 1.0) * SR)
    t = np.arange(N) / SR
    inten = 0.35 + 0.65 * np.clip(t / duration, 0, 1) ** 1.1

    # --- dark drone: A1 + tritone + octave, harmonics shimmer slowly (a "filter" without a filter)
    drone = np.zeros(N)
    for f, g in [(55.0, 1.0), (55.35, 0.8), (77.78, 0.55), (110.0, 0.5)]:
        for k in range(1, 9):
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.05 + 0.013 * k) * t + k * 1.7)
            drone += g * np.sin(2 * np.pi * f * k * t) / (k ** 1.15) * (0.35 + 0.65 * lfo if k > 2 else 1.0)
    drone *= 0.11 * (0.55 + 0.45 * inten)

    # --- heartbeat (lub-dub) that speeds up
    beat = np.zeros(N)
    tb = 0.8
    while tb < duration:
        bpm = 64 + 34 * min(1.0, tb / duration)
        period = 60.0 / bpm
        n = int(0.5 * SR)
        tt = np.arange(n) / SR
        thump = np.sin(2 * np.pi * np.cumsum(62 - 90 * tt) / SR) * np.exp(-tt * 13)
        a = 0.55 * (0.5 + 0.5 * inten[min(N - 1, int(tb * SR))])
        _add(beat, tb, thump * a)
        _add(beat, tb + 0.27 * period / 0.6 * 0.6, thump * a * 0.62)
        tb += period
    # --- plucked ostinato in A phrygian, fades in after ~10 s
    notes = [55.0, 55.0, 58.27, 55.0, 55.0, 77.78, 58.27, 65.41]
    pluck = np.zeros(N)
    tp, i = 0.0, 0
    while tp < duration:
        bpm = 64 + 34 * min(1.0, tp / duration)
        step = 60.0 / bpm / 2
        f = notes[i % len(notes)] * (2 if (i // 16) % 3 == 2 else 1)
        n = int(0.45 * SR)
        tt = np.arange(n) / SR
        s = sum(np.sin(2 * np.pi * f * k * tt) / k * np.exp(-tt * (7 + 3 * k)) for k in range(1, 7))
        amp = np.clip((tp - 10) / 18, 0, 1) * 0.16 * (0.5 + 0.5 * inten[min(N - 1, int(tp * SR))])
        _add(pluck, tp, s * amp)
        tp += step
        i += 1

    # --- high tense strings (minor second cluster) with vibrato, swelling, tremolo late in the piece
    strings = np.zeros(N)
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24.0 - 1.2)
    trem = 1 - 0.35 * (np.clip((t - duration * 0.45) / (duration * 0.3), 0, 1)) * (0.5 + 0.5 * np.sin(2 * np.pi * 7.5 * t))
    for f in (440.0, 466.16, 220.0):
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.3 * t + f)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        strings += (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) / 3
    strings *= 0.05 * swell * trem * (0.3 + 0.7 * inten) * np.clip(t / 6, 0, 1)

    # --- risers and impacts on the hit times
    fx = np.zeros(N)
    for h in hits:
        r = 3.2
        n = int(r * SR)
        tt = np.arange(n) / SR
        ramp = (tt / r) ** 2.2
        f_inst = 180 + 1700 * (tt / r) ** 1.6
        ph = 2 * np.pi * np.cumsum(f_inst) / SR
        noise = np.diff(rng.standard_normal(n + 1))
        riser = (np.sin(ph) * 0.5 + np.sin(ph * 1.5) * 0.25 + noise * 0.35) * ramp * 0.30
        _add(fx, h - r, riser)
        m = int(2.6 * SR)
        tm = np.arange(m) / SR
        boom = np.sin(2 * np.pi * np.cumsum(70 - 38 * np.minimum(tm, 1.2) / 1.2) / SR) * np.exp(-tm * 1.6)
        burst = _lowpass(rng.standard_normal(m), 12) * np.exp(-tm * 5.5)
        _add(fx, h, boom * 0.85 + burst * 1.0)

    # --- reverb on strings + fx
    ir_n = int(2.4 * SR)
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR * 2.2)
    ir[:int(0.02 * SR)] *= 0.2
    bus = strings + fx
    size = 1 << int(np.ceil(np.log2(N + ir_n)))
    wet = np.fft.irfft(np.fft.rfft(bus, size) * np.fft.rfft(ir, size), size)[:N]
    wet /= (np.max(np.abs(wet)) + 1e-9) / (np.max(np.abs(bus)) + 1e-9) * 1.0

    dry = drone + beat + pluck + strings + fx
    L = dry + 0.45 * wet
    R = dry + 0.45 * np.roll(wet, 640)
    st = np.stack([L, R], axis=1)
    st = np.tanh(st * 1.3) / 1.3
    fade_in = np.clip(t / 1.5, 0, 1)
    fade_out = np.clip((duration + 0.6 - t) / 2.5, 0, 1)
    st *= (fade_in * fade_out)[:, None]
    st *= 0.89 / np.max(np.abs(st))
    pcm = (st * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import sys
    make_music(float(sys.argv[2]), [float(x) for x in sys.argv[3:]], sys.argv[1])
    print("ok")
