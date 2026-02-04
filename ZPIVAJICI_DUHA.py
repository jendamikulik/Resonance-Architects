import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

# -----------------------------
# 1) "Duha" = spektrum I(lambda)
# -----------------------------
# Vlnové délky ve viditelném rozsahu (nm)
lam_min, lam_max = 380.0, 740.0
N_lam = 1400
lam = np.linspace(lam_min, lam_max, N_lam)

# 7 "barev" jako Gaussy (nm)
centers = np.array([650, 600, 570, 530, 470, 425, 400], dtype=float)  # R O Y G B I V
amps    = np.array([1.00, 0.95, 0.90, 0.95, 0.90, 0.80, 0.75], dtype=float)
sigma_l = 18.0  # šířka pásma (nm) – uprav pro ostřejší/měkčí duhu

I = np.zeros_like(lam)
for A, c in zip(amps, centers):
    I += A * np.exp(-0.5 * ((lam - c) / sigma_l) ** 2)

# Normalizace intenzity (0..1)
I /= I.max()

# ------------------------------------------
# 2) Mapování lambda -> slyšitelná frekvence
# ------------------------------------------
# Hudební reference:
f_ref = 220.0         # A3
lam0  = 555.0         # referenční lambda (nm) pro mapování
alpha = 24.0          # "rozsah" v půltónech přes log(λ0/λ) (větší = širší melodie)

# pitch v půltónech (log mapování, elegantní pro hudbu)
p = alpha * np.log(lam0 / lam)            # v půltónech (relativně)
f_audio = f_ref * (2.0 ** (p / 12.0))     # Hz

# Ořez do rozumného slyšitelného rozsahu
f_lo, f_hi = 80.0, 2000.0
mask = (f_audio >= f_lo) & (f_audio <= f_hi)
lam_m = lam[mask]
I_m = I[mask]
f_m = f_audio[mask]

# -----------------------------
# 3) Syntéza zvuku s(t)
# -----------------------------
sr = 44100
dur = 5.0
t = np.arange(int(sr * dur)) / sr

# Fázový "třpyt" (můžeš dát 0 pro čistý akord)
rng = np.random.default_rng(7)
phi = rng.uniform(0, 2*np.pi, size=len(f_m))

# Amplitudy: z I(lambda), lehce komprimované, aby to nebyla jen jedna dominantní barva
amp = I_m ** 0.85
amp /= amp.sum()  # normalizace energie přes složky

# Superpozice sinusů (vektorově)
# s(t) = sum_k amp_k * sin(2π f_k t + phi_k)
s = np.sin(2*np.pi * f_m[:, None] * t[None, :] + phi[:, None])
s = (amp[:, None] * s).sum(axis=0)

# Jemný envelope (fade in/out), ať to necvaká
fade = int(0.02 * sr)
env = np.ones_like(s)
env[:fade] = np.linspace(0, 1, fade)
env[-fade:] = np.linspace(1, 0, fade)
s *= env

# Normalizace na -1..1
s /= np.max(np.abs(s) + 1e-12)

# -----------------------------
# 4) Uložení WAV (16-bit PCM)
# -----------------------------
out_wav = "singing_rainbow.wav"
with wave.open(out_wav, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sr)
    # převod na int16
    data = (s * 32767.0).astype(np.int16)
    wf.writeframes(data.tobytes())

print(f"Uloženo: {out_wav}")

# -----------------------------
# 5) Grafy: duha a spektrum zvuku
# -----------------------------
plt.figure()
plt.plot(lam, I)
plt.xlabel("λ [nm]")
plt.ylabel("I(λ)")
plt.title("Duha: spektrum intenzity I(λ)")
plt.tight_layout()
plt.show()

# FFT spektrum zvuku (jen pro přehled)
S = np.fft.rfft(s)
freq = np.fft.rfftfreq(len(s), d=1/sr)
mag = np.abs(S)
mag /= mag.max() + 1e-12

plt.figure()
plt.plot(freq, mag)
plt.xlim(0, 3000)
plt.xlabel("f [Hz]")
plt.ylabel("norm |FFT|")
plt.title("Zpívající duha: spektrum výsledného zvuku")
plt.tight_layout()
plt.show()
