import numpy as np
import matplotlib.pyplot as plt
import wave

# -----------------------------
# Audio nastavení
# -----------------------------
sr = 44100
bpm = 76
beats_per_bar = 4
bar_dur = 60.0 / bpm * beats_per_bar
bars = 6
dur = bars * bar_dur
t = np.arange(int(sr * dur)) / sr

def save_wav_mono(filename, s, sr=44100):
    s = s / (np.max(np.abs(s)) + 1e-12)
    data = (s * 32767.0).astype(np.int16)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

# -----------------------------
# "Duha" mapování (stejné kouzlo)
# -----------------------------
f_ref = 220.0      # A3
lam0  = 555.0
alpha = 24.0

lam_min, lam_max = 380.0, 740.0
lam = np.linspace(lam_min, lam_max, 1400)
sigma_l = 16.0

def pitch_semitones(f):
    return 12.0 * np.log2(f / f_ref)

def pitch_to_lambda_nm(p):
    return lam0 * np.exp(-p / alpha)

def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu)/sig)**2)

# -----------------------------
# Jemná progresie (Cmaj7 -> Am7 -> Fmaj7 -> Gsus4 -> G -> Cmaj7)
# MIDI: C4=60
# -----------------------------
prog = [
    ("Cmaj7",  [60, 64, 67, 71]),      # C E G B
    ("Am7",    [57, 60, 64, 67]),      # A C E G
    ("Fmaj7",  [53, 57, 60, 64]),      # F A C E
    ("Gsus4",  [55, 60, 62, 67]),      # G C D G
    ("G",      [55, 59, 62, 67]),      # G B D G
    ("Cmaj7",  [60, 64, 67, 71]),
]

# délka jednoho akordu = 1 takt
chord_len = bar_dur
fade = 0.12  # crossfade (s)
fade_samp = int(fade * sr)

# "dech" (velmi jemný)
breath = 0.55 + 0.45 * (0.5 + 0.5*np.sin(2*np.pi*0.10*t))

# -----------------------------
# Syntéza padů (jemné, bez třpytu)
# -----------------------------
s = np.zeros_like(t)

# zároveň posbíráme "duhový otisk" přes celou progresi
I_total = np.zeros_like(lam)

def soft_env(n, fade_samp):
    env = np.ones(n)
    if fade_samp > 0:
        env[:fade_samp] = np.linspace(0, 1, fade_samp)
        env[-fade_samp:] = np.linspace(1, 0, fade_samp)
    return env

for i, (name, chord_midi) in enumerate(prog):
    start = i * chord_len
    end = (i + 1) * chord_len
    idx = (t >= start) & (t < end)
    tt = t[idx] - start

    # akord -> Hz
    freqs = np.array([midi_to_hz(m) for m in chord_midi], dtype=float)

    # měkké rozložení hlasitosti (nižší tóny jemněji)
    amps = np.array([0.85, 0.75, 0.65, 0.55], dtype=float)
    amps = amps / amps.sum()

    # čisté fáze (klid)
    phi = np.zeros(len(freqs))

    # pad: součet sinusů
    seg = np.zeros_like(tt)
    for A, f, ph in zip(amps, freqs, phi):
        seg += A * np.sin(2*np.pi*f*tt + ph)

    # envelope pro akord + crossfade
    env = soft_env(len(seg), fade_samp)
    seg *= env

    s[idx] += seg

    # duhový otisk tohoto akordu
    p = pitch_semitones(freqs)
    lam_notes = pitch_to_lambda_nm(p)
    I = np.zeros_like(lam)
    for A, ln in zip(amps, lam_notes):
        I += A * gaussian(lam, ln, sigma_l)
    I_total += I

# aplikuj dech + global fade
s *= breath
global_fade = int(0.20 * sr)
g = np.ones_like(s)
g[:global_fade] = np.linspace(0, 1, global_fade)
g[-global_fade:] = np.linspace(1, 0, global_fade)
s *= g

# normalizace a export
save_wav_mono("unity_rainbow_pad.wav", s, sr)
print("Uloženo: unity_rainbow_pad.wav")

# -----------------------------
# Graf duhového otisku progresie
# -----------------------------
I_total /= I_total.max() + 1e-12

plt.figure()
plt.plot(lam, I_total)
plt.xlabel("λ [nm]")
plt.ylabel("I(λ)")
plt.title("Unity Rainbow: duhový otisk celé harmonie")
plt.tight_layout()
plt.show()

# -----------------------------
# FFT spektrum (pro krásu)
# -----------------------------
S = np.fft.rfft(s)
freq = np.fft.rfftfreq(len(s), d=1/sr)
mag = np.abs(S)
mag /= mag.max() + 1e-12

plt.figure()
plt.plot(freq, mag)
plt.xlim(0, 1500)
plt.xlabel("f [Hz]")
plt.ylabel("norm |FFT|")
plt.title("Spektrum: unity_rainbow_pad.wav")
plt.tight_layout()
plt.show()
