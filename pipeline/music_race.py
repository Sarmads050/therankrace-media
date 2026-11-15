"""Original cinematic "race" soundtrack, synthesised from scratch with numpy (no samples, nothing third-party).

make_race_music(bounds, bars, out_wav):
    bounds = [0, t1, t2, t3, t4, total]   section boundaries (the video's big moments)
    bars   = [8, 14, 26, 11, 5]           whole bars per section -> each boundary lands exactly on a bar downbeat,
                                          and the tempo glides ~102 -> 120 bpm, so the music "speeds up" with the race.
Story: 1 mysterious intro -> 2 beat drops in -> 3 full groove + hook (with a breakdown) -> 4 key lift, climax -> 5 resolve.
Sound: pumping sidechain on pad/bass/arps, arpeggio plucks, sub bass, kick/clap/hats, lead hook with echo, risers, impacts.
"""
import wave

import numpy as np

SR = 44100
G_DR, G_BA, G_PA, G_AR, G_LE, G_FX = 0.55, 0.75, 0.6, 1.7, 0.7, 0.8
rng = np.random.default_rng(11)


def mf(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def tt(n):
    return np.arange(n) / SR


def lowpass(x, k):
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    return np.concatenate([np.zeros(k - 1), y])


def hp(x):
    return np.concatenate([[0.0], np.diff(x)])


class Bus:
    def __init__(self, n):
        self.L = np.zeros(n)
        self.R = np.zeros(n)
        self.n = n

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= self.n or i + len(sig) <= 0:
            return
        a = max(0, -i)
        j = min(self.n, i + len(sig))
        seg = sig[a: j - i] * gain
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.L[max(i, 0): j] += seg * gl * 1.414
        self.R[max(i, 0): j] += seg * gr * 1.414


# ------------------------------------------------------------------ voices
def kick(power=1.0):
    n = int(0.42 * SR)
    t = tt(n)
    f = 44 + 110 * np.exp(-t / 0.032)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.17)
    y += 0.35 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.004)
    return np.tanh(y * 1.6 * power) * 0.9


def snare():
    n = int(0.3 * SR)
    t = tt(n)
    nz = hp(rng.standard_normal(n))
    return (nz * np.exp(-t / 0.085) * 0.26 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.5)


def clap():
    n = int(0.35 * SR)
    t = tt(n)
    nz = hp(rng.standard_normal(n))
    y = np.zeros(n)
    for d in (0.0, 0.011, 0.022):
        k = int(d * SR)
        y[k:] += nz[: n - k] * np.exp(-t[: n - k] / 0.018) * 0.32
    y += nz * np.exp(-t / 0.11) * 0.13
    return y


def hat(open_=False):
    n = int((0.22 if open_ else 0.06) * SR)
    t = tt(n)
    nz = hp(hp(rng.standard_normal(n)))
    return nz * np.exp(-t / (0.09 if open_ else 0.018)) * 0.17


def bass_note(f, dur):
    n = int((dur + 0.08) * SR)
    t = tt(n)
    y = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(4 * np.pi * f * t + 0.5) + 0.2 * np.sin(6 * np.pi * f * t)
    env = np.minimum(t / 0.006, 1.0) * np.exp(-t / (dur * 1.1 + 0.05))
    return np.tanh(y * 1.3) * env * 0.75


def pluck(f, dur=0.34):
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    for k in range(1, 9):
        y += np.sin(2 * np.pi * f * k * t + k) / k ** 1.05 * np.exp(-t * (3.0 + 5.5 * k))
    return y * np.exp(-t / 0.16)


def pad_chord(freqs, dur, att=0.7, rel=0.9, shimmer=0.0):
    n = int((dur + rel) * SR)
    t = tt(n)
    y = np.zeros(n)
    for f in freqs:
        for cents in (-9, 0, 9):
            ff = f * 2 ** (cents / 1200.0)
            for k in range(1, 6):
                lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.11 * k) * t + k)
                y += np.sin(2 * np.pi * ff * k * t + cents) / k ** 1.4 * lfo * (1.0 if k < 3 else 0.6 + shimmer)
    env = np.minimum(t / att, 1.0) * np.clip((dur + rel - t) / rel, 0, 1)
    return y * env * 0.06


def lead_note(f, dur, bright=1.0):
    n = int((dur + 0.3) * SR)
    t = tt(n)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.12) / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    y = np.zeros(n)
    for k in range(1, 9):
        y += np.sin(k * ph) / k ** (1.0 + 0.4 / bright)
    env = np.minimum(t / 0.025, 1.0) * np.clip((dur + 0.3 - t) / 0.3, 0, 1) * (0.75 + 0.25 * np.exp(-t / 0.4))
    return y * env * 0.5


def riser(dur):
    n = int(dur * SR)
    t = tt(n)
    x = (t / dur) ** 2.2
    nz = hp(rng.standard_normal(n)) * x * 0.10
    tone = np.sin(2 * np.pi * np.cumsum(180 + 3200 * x) / SR) * x * 0.12
    return nz + tone


def impact(power=1.0):
    n = int(2.6 * SR)
    t = tt(n)
    boom = np.sin(2 * np.pi * np.cumsum(30 + 70 * np.exp(-t / 0.5)) / SR) * np.exp(-t / 0.95)
    nz = lowpass(rng.standard_normal(n), 6) * np.exp(-t / 0.5) * 0.5
    crash = hp(rng.standard_normal(n)) * np.exp(-t / 0.9) * 0.07
    return (boom * 1.1 + nz + crash) * power


def crash():
    n = int(2.4 * SR)
    t = tt(n)
    return hp(hp(rng.standard_normal(n))) * np.exp(-t / 0.8) * 0.1


# ------------------------------------------------------------------ harmony
CH = {"Am": (0, "m"), "F": (-4, "M"), "C": (3, "M"), "G": (-2, "M"), "Dm": (-7, "m"), "E": (-5, "M"), "AM": (0, "M")}
HOOK = {  # (beat, length in beats, midi) per bar of the Am - F - C - G progression
    "Am": [(0, 1.5, 76), (1.5, 0.5, 74), (2, 2, 72)],
    "F": [(0, 1.5, 72), (1.5, 0.5, 69), (2, 2, 77)],
    "C": [(0, 1.5, 79), (1.5, 0.5, 76), (2, 2, 72)],
    "G": [(0, 1, 74), (1, 1, 71), (2, 1, 74), (3, 1, 76)],
}


def triad(name, shift):
    off, q = CH[name]
    root = 57 + off + shift
    return root, root + (3 if q == "m" else 4), root + 7


def bass_root(name, shift):
    return 33 + ((CH[name][0] + shift) % 12)


def reverb(x, secs=2.2, wet=1.0):
    n = int(secs * SR)
    t = tt(n)
    ir = rng.standard_normal(n) * np.exp(-t / (secs / 4.2))
    ir = lowpass(ir, 3)
    ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)] * 0.0035 * wet


def echo(x, delay, fb=0.38, taps=4):
    d = int(delay * SR)
    y = x.copy()
    for k in range(1, taps + 1):
        y[k * d:] += x[: len(x) - k * d] * (fb ** k)
    return y


# ------------------------------------------------------------------ the composition
def make_race_music(bounds, bars, out_wav):
    total = bounds[-1]
    N = int((total + 3.0) * SR)
    drums, bass, pad, arp, lead, fx = (Bus(N) for _ in range(6))
    kicks = []
    plans = [  # chords per bar, key shift
        ((["Am", "Am", "F", "F", "Am", "Am", "Dm"] * 3)[: bars[0] - 1] + ["E"], 0),
        ([c for _ in range(9) for c in ("Am", "F", "C", "G")][: bars[1]], 0),
        ([c for _ in range(12) for c in ("Am", "F", "C", "G")][: bars[2]], 0),
        ([c for _ in range(6) for c in ("Am", "F", "C", "G")][: bars[3]], 2),
        ([("Am", "F", "C", "G")[i % 4] for i in range(max(0, bars[4] - 5))] + ["F", "C", "G", "Am", "AM"][-min(bars[4], 5):], 2),
    ]
    for s in range(5):
        t0, t1, nb = bounds[s], bounds[s + 1], bars[s]
        bl = (t1 - t0) / (nb * 4)  # beat length
        s16 = bl / 4
        chords, shift = plans[s]
        T = lambda b, beat: t0 + (b * 4 + beat) * bl  # noqa
        # entry accent on every section start
        if s > 0:
            fx.add(impact(1.0 if s != 4 else 0.8), t0 - 0.02, 0.9)
            fx.add(crash(), t0, 0.7)
        for b in range(nb):
            name = chords[b]
            r, th, fi = triad(name, shift)
            fr = [mf(r), mf(th), mf(fi), mf(r + 12)]
            last_build = b >= nb - (2 if s in (0, 2) else 1) and s in (0, 1, 2)
            # ---- pad: one long chord per bar, overlapping
            pg = {0: 0.25 + 0.75 * min(1.0, b / 5.0), 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.0 if b < 4 else 1.25}[s]
            pad.add(pad_chord([mf(r - 12), mf(th - 12), mf(fi - 12), mf(th), mf(fi)], 4 * bl, 0.55, 0.8,
                              shimmer=0.4 if s >= 3 else 0.0), T(b, 0), pg)
            # ---- drums
            groove = s in (1, 2, 3)
            bs = int(nb * 0.46)  # breakdown start (section 3)
            breakdown = s == 2 and bs <= b < bs + 4
            if s == 0:
                if b >= 0:
                    for beat in (0, 2):
                        drums.add(kick(0.55), T(b, beat), 0.35 + 0.05 * b)
                        kicks.append(T(b, beat))
                if b >= 4:
                    for q in range(8):
                        drums.add(hat(), T(b, q * 0.5), 0.18 + 0.02 * (b - 4) + (0.05 if q % 2 else 0))
            elif s in (1, 2, 3, 4):
                full = groove and not breakdown
                if s == 4:
                    kb = (0, 2) if b < nb - 1 else ()
                elif full:
                    kb = (0, 1, 2, 3)
                else:
                    kb = ()
                for beat in kb:
                    drums.add(kick(1.0), T(b, beat), 0.95)
                    kicks.append(T(b, beat))
                if full and (b >= 2 or s != 1):
                    for beat in (1, 3):
                        drums.add(clap(), T(b, beat), 0.55)
                if full or breakdown:
                    for q in range(8):
                        drums.add(hat(), T(b, q * 0.5 + 0.5 * (q % 2 == 0) * 0.0), 0.2 + (0.08 if q % 2 else 0))
                    drums.add(hat(True), T(b, 1.5), 0.2)
                    drums.add(hat(True), T(b, 3.5), 0.2)
                if s in (2, 3) and full and (b >= nb // 3 or s == 3):  # 16th hats for more drive
                    for q in range(16):
                        if q % 2:
                            drums.add(hat(), T(b, q * 0.25), 0.1)
            # ---- snare-roll build into the next section
            if last_build or (s == 3 and b == nb - 1):
                for q in range(16 if b == nb - 1 else 8):
                    drums.add(snare(), T(b, q * (0.25 if b == nb - 1 else 0.5)),
                              0.12 + 0.5 * (q / (16 if b == nb - 1 else 8)))
            if b == nb - 2 and s < 4:  # riser sweeps up for the last two bars and ends exactly on the next downbeat
                fx.add(riser(t1 - T(b, 0)), T(b, 0), 0.8)
            # ---- bass: driving 8ths in groove sections, sub notes elsewhere
            br = bass_root(name, shift)
            if groove and not breakdown:
                for q in range(8):
                    f = mf(br + (12 if q in (3, 7) else 0))
                    bass.add(bass_note(f, bl * 0.45), T(b, q * 0.5), 0.8 if q % 2 == 0 else 0.55)
            elif s == 4 or s == 0:
                bass.add(bass_note(mf(br), bl * 3.6), T(b, 0), 0.6 if s == 0 else 0.8)
            # ---- arpeggio (16ths in groove, 8ths in the intro)
            tones = [r + 12, th + 12, fi + 12, r + 24, fi + 12, th + 12, r + 24, fi + 12]
            if s == 3:
                tones = [r + 12, th + 12, fi + 12, r + 24, th + 24, r + 24, fi + 12, th + 12]
            if s == 0 and b >= 2:
                for q in range(8):
                    arp.add(pluck(mf(tones[q % 8]), 0.5), T(b, q * 0.5), 0.12 + 0.05 * min(b - 2, 4),
                            pan=-0.4 if q % 2 else 0.4)
            elif s in (1, 2, 3):
                for q in range(16):
                    g = 0.2 if q % 4 == 0 else 0.13
                    arp.add(pluck(mf(tones[q % 8]), 0.3), T(b, q * 0.25), g * (1.0 if s != 3 else 1.15),
                            pan=-0.5 if q % 2 else 0.5)
            elif s == 4 and b < nb - 1:
                for q in range(8):
                    arp.add(pluck(mf(tones[q % 8]), 0.55), T(b, q * 0.5), 0.12, pan=-0.3 if q % 2 else 0.3)
            # ---- lead hook
            if (s == 2 and (4 <= b < bs or bs + 4 <= b < nb - 2)) or s == 3 or (s == 4 and b < nb - 1):
                key = ["Am", "F", "C", "G"][b % 4] if s in (2, 3) else ["F", "C", "G", "Am"][(b + 5 - nb) % 4]
                base = HOOK[key] if key in HOOK else HOOK["Am"]
                octv = 12 if s == 3 or (s == 2 and b >= bs + 4) else 0
                gl = 0.55 if s != 4 else 0.45
                for beat, ln, note in base:
                    lead.add(lead_note(mf(note + shift + octv), ln * bl * 0.95, 1.0 if s != 3 else 1.4), T(b, beat), gl,
                             pan=0.1)
            if s == 2 and bs <= b < bs + 4:  # breakdown: shimmering high chord tones, no drums
                for q in range(4):
                    lead.add(lead_note(mf(fi + 24), bl * 0.9, 0.6), T(b, q), 0.18, pan=-0.2 + 0.13 * q)
    # final held chord: bright B major shimmer with a long tail
    last = bounds[4] + (bars[4] - 1) * (bounds[5] - bounds[4]) / bars[4]
    r, th, fi = triad("AM", 2)
    pad.add(pad_chord([mf(r), mf(th), mf(fi), mf(r + 12), mf(th + 12)], 4.0, 0.3, 1.6, shimmer=0.8), last, 1.4)
    fx.add(crash(), last, 0.8)
    fx.add(impact(0.9), last, 0.7)

    # ---- sidechain pump on pad / bass / arp (kick ducks them)
    pump = np.ones(N)
    for tk in kicks:
        i = int(tk * SR)
        m = int(0.38 * SR)
        if i >= N:
            continue
        j = min(N, i + m)
        env = 1 - 0.62 * np.exp(-tt(j - i) / 0.13)
        pump[i:j] = np.minimum(pump[i:j], env)
    pump_soft = 0.55 + 0.45 * pump

    def lr(*buses):
        return sum(b.L for b in buses), sum(b.R for b in buses)

    pad.L *= pump_soft
    pad.R *= pump_soft
    bass.L *= pump
    bass.R *= pump
    arp.L *= pump_soft
    arp.R *= pump_soft
    lead_e = Bus(N)
    lead_e.L, lead_e.R = echo(lead.L, 0.375, 0.34), echo(lead.R, 0.5, 0.3)
    wetL, wetR = lr(pad, arp, lead_e)
    outL = drums.L * G_DR + bass.L * G_BA + pad.L * G_PA + arp.L * G_AR + lead_e.L * G_LE + fx.L * G_FX + reverb(wetL, wet=1.0)
    outR = drums.R * G_DR + bass.R * G_BA + pad.R * G_PA + arp.R * G_AR + lead_e.R * G_LE + fx.R * G_FX + reverb(wetR, wet=1.0)
    for nm, bus in (("drums", drums), ("bass", bass), ("pad", pad), ("arp", arp), ("lead", lead_e), ("fx", fx)):
        print(f"  bus {nm:6s} rms {np.sqrt(np.mean(bus.L ** 2)):.4f} peak {np.abs(bus.L).max():.3f}")
    print(f"  reverb rms {np.sqrt(np.mean(reverb(wetL) ** 2)):.4f}")
    out = np.stack([outL, outR], 1)[: int(total * SR)]
    out = np.tanh(out * 1.05)
    out /= max(1e-6, np.abs(out).max())
    out *= 0.9
    fade = np.ones(len(out))
    nfade = int(2.5 * SR)
    fade[-nfade:] = np.linspace(1, 0, nfade) ** 1.5
    fade[: int(0.5 * SR)] = np.linspace(0, 1, int(0.5 * SR))
    out *= fade[:, None]
    pcm = (out * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out_wav
