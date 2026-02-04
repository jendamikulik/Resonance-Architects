import numpy as np
import matplotlib.pyplot as plt
import wave

# -----------------------------
# Nastavení mapování (stejné kouzlo jako duha->zpěv)
# -----------------------------
f_ref = 220.0      # reference A3
lam0  = 555.0      # referenční lambda (nm)
alpha = 24.0       # jak "široce" mapujeme do barev (větší = větší rozptyl barev)

# Duha osa (nm)
lam_min, lam_max = 380.0, 740.0
lam = np.linspace(lam_min, lam_max, 1400)

# Jak široký je "barevný otisk" jednoho tónu (nm)
sigma_l = 14.0

# Audio
sr = 44100
dur = 4.0
t = np.arange(int(sr*dur)) / sr

# -----------------------------
# Pomocné funkce
# -----------------------------
def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def pitch_semitones(f):
    # relativně k f_ref
    return 12.0 * np.log2(f / f_ref)

def pitch_to_lambda_nm(p):
    # Inverze p = alpha * ln(lam0/lam)  =>  lam = lam0 * exp(-p/alpha)
    return lam0 * np.exp(-p / alpha)

def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu)/sig)**2)

def save_wav_mono(filename, s, sr=44100):
    s = s / (np.max(np.abs(s)) + 1e-12)
    data = (s * 32767.0).astype(np.int16)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

# -----------------------------
# 1) Zadej akord
# -----------------------------
# Varianta A: MIDI noty (změň si podle sebe)
# Např. Cmaj9: C4 E4 G4 B4 D5 = 60 64 67 71 74
chord_midi = [60, 64, 67, 71, 74]

# Varianta B: rovnou Hz (odkomentuj a zadej)
# chord_hz = [261.63, 329.63, 392.00]

chord_hz = [midi_to_hz(m) for m in chord_midi]

# Hlasitosti tónů (můžeš si upravit)
amps = np.ones(len(chord_hz), dtype=float)
amps = amps / amps.sum()

# -----------------------------
# 2) Akord -> barvy (lambda) + "duhové spektrum"
# -----------------------------
p = pitch_semitones(np.array(chord_hz))
lam_notes = pitch_to_lambda_nm(p)

# Duhové spektrum akordu: I(lam) = sum Gaussů kolem lam_notes
I = np.zeros_like(lam)
for A, ln in zip(amps, lam_notes):
    I += A * gaussian(lam, ln, sigma_l)
I /= I.max() + 1e-12

# -----------------------------
# 3) Zpívající akord (audio)
# -----------------------------
rng = np.random.default_rng(7)
phi = rng.uniform(0, 2*np.pi, size=len(chord_hz))  # "třpyt"; dej 0 pro čistý akord
# phi = np.zeros(len(chord_hz))

s = np.zeros_like(t)
for A, f, ph in zip(amps, chord_hz, phi):
    s += A * np.sin(2*np.pi*f*t + ph)

# Jemný fade in/out
fade = int(0.02 * sr)
env = np.ones_like(s)
env[:fade] = np.linspace(0, 1, fade)
env[-fade:] = np.linspace(1, 0, fade)
s *= env

save_wav_mono("rainbow_chord.wav", s, sr)
print("Uloženo: rainbow_chord.wav")

# -----------------------------
# 4) Výpis "barvy tónů" + graf
# -----------------------------
print("\nTóny akordu:")
for f, ln in zip(chord_hz, lam_notes):
    print(f"  f = {f:8.2f} Hz  ->  λ ≈ {ln:7.1f} nm")

plt.figure()
plt.plot(lam, I)
plt.xlabel("λ [nm]")
plt.ylabel("I(λ) (duhový otisk akordu)")
plt.title("Zpívající duha: spektrum akordu v barvách")
plt.tight_layout()
plt.show()

# FFT spektrum zvuku (pro krásu)
S = np.fft.rfft(s)
freq = np.fft.rfftfreq(len(s), d=1/sr)
mag = np.abs(S); mag /= mag.max() + 1e-12

plt.figure()
plt.plot(freq, mag)
plt.xlim(0, 2000)
plt.xlabel("f [Hz]")
plt.ylabel("norm |FFT|")
plt.title("Spektrum zvuku: 'rainbow_chord.wav'")
plt.tight_layout()
plt.show()
