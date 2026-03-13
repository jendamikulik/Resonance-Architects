import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) Model parameters
# --------------------------------------------------

hbar = 1.0                  # scaled units
lambda_c = 1.0             # coupling prefactor
A_drive = 1.0              # drive amplitude

omega_el = 10.0            # resonance center
gamma_el = 0.8             # resonance width

P_in = 5.0                 # input power / loading source
Gamma_loss = 1.5           # resonant-sector loss
Gamma_rel = 1.0            # pair-sector relaxation

E_c = 1.8                  # model threshold for "fusion-relevant" pair loading

# detectability parameters
n_samples = 1000
K = 1.0
tau_ent = 1.0
sigma_noise = 0.2
kappa = 1.0

# --------------------------------------------------
# 2) Resonance profile and anchored coupling
# --------------------------------------------------

def L(omega: np.ndarray) -> np.ndarray:
    """Lorentzian anchor profile."""
    return gamma_el**2 / ((omega - omega_el)**2 + gamma_el**2)

def alpha(omega: np.ndarray) -> np.ndarray:
    """Anchored coupling."""
    return lambda_c * A_drive * hbar * omega_el * L(omega)

# --------------------------------------------------
# 3) Transfer law
#    Monotone saturating map G(alpha)
# --------------------------------------------------

Gamma_tr_max = 3.5
alpha_sat = 2.0

def G(alpha_val: np.ndarray) -> np.ndarray:
    """Monotone transfer law G'(alpha) > 0."""
    return Gamma_tr_max * alpha_val / (alpha_val + alpha_sat)

def Gamma_tr(omega: np.ndarray) -> np.ndarray:
    return G(alpha(omega))

# --------------------------------------------------
# 4) Steady-state energies
# --------------------------------------------------

def E_res_star(omega: np.ndarray) -> np.ndarray:
    gtr = Gamma_tr(omega)
    return P_in / (Gamma_loss + gtr)

def E_pair_star(omega: np.ndarray) -> np.ndarray:
    gtr = Gamma_tr(omega)
    return (gtr / Gamma_rel) * (P_in / (Gamma_loss + gtr))

# --------------------------------------------------
# 5) Detectability ratio
# --------------------------------------------------

def R_n(omega: np.ndarray) -> np.ndarray:
    prefactor = np.sqrt(n_samples) * K * lambda_c * A_drive * hbar * omega_el * kappa * tau_ent / sigma_noise
    return prefactor * L(omega)

# --------------------------------------------------
# 6) Frequency scan
# --------------------------------------------------

omegas = np.linspace(5.0, 15.0, 2000)

L_vals = L(omegas)
alpha_vals = alpha(omegas)
Gamma_tr_vals = Gamma_tr(omegas)
E_res_vals = E_res_star(omegas)
E_pair_vals = E_pair_star(omegas)
R_n_vals = R_n(omegas)

# peak data
idx_peak = np.argmax(E_pair_vals)
omega_peak = omegas[idx_peak]
E_pair_peak = E_pair_vals[idx_peak]
R_n_peak = R_n_vals[idx_peak]

# threshold crossing
crossing_mask = E_pair_vals >= E_c
has_crossing = np.any(crossing_mask)

# --------------------------------------------------
# 7) Report
# --------------------------------------------------

print("=== Resonance-threshold kill test ===")
print(f"Resonance center omega_el      = {omega_el:.4f}")
print(f"Peak of E_pair*(omega) at      = {omega_peak:.4f}")
print(f"Peak E_pair*(omega)            = {E_pair_peak:.6f}")
print(f"Peak detectability R_n(omega)  = {R_n_peak:.6f}")
print(f"Threshold E_c                  = {E_c:.6f}")

if has_crossing:
    idx_first = np.argmax(crossing_mask)
    omega_first = omegas[idx_first]
    print(f"\nRESULT: threshold crossed.")
    print(f"First crossing near omega      = {omega_first:.6f}")
    print("Interpretation: pair channel is fusion-relevant IN THE MODEL.")
else:
    print("\nRESULT: no threshold crossing.")
    print("Interpretation: no fusion-relevant pair loading IN THE MODEL.")
    print("Kill test verdict: if no measured peak appears, the mechanism dies.")

# --------------------------------------------------
# 8) Plot
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Lorentzian anchor
axes[0, 0].plot(omegas, L_vals)
axes[0, 0].axvline(omega_el, linestyle="--")
axes[0, 0].set_title("Lorentzian anchor L(ω)")
axes[0, 0].set_xlabel("ω")
axes[0, 0].set_ylabel("L(ω)")
axes[0, 0].grid(True, alpha=0.3)

# Anchored transfer rate
axes[0, 1].plot(omegas, Gamma_tr_vals)
axes[0, 1].axvline(omega_el, linestyle="--")
axes[0, 1].set_title("Transfer rate Γ_tr(ω)")
axes[0, 1].set_xlabel("ω")
axes[0, 1].set_ylabel("Γ_tr(ω)")
axes[0, 1].grid(True, alpha=0.3)

# Pair loading with threshold
axes[1, 0].plot(omegas, E_pair_vals, label=r"$E_{\mathrm{pair}}^*(\omega)$")
axes[1, 0].axhline(E_c, linestyle="--", label=r"$E_c$")
axes[1, 0].axvline(omega_el, linestyle="--")
axes[1, 0].set_title("Steady-state pair loading")
axes[1, 0].set_xlabel("ω")
axes[1, 0].set_ylabel(r"$E_{\mathrm{pair}}^*(\omega)$")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Detectability
axes[1, 1].plot(omegas, R_n_vals)
axes[1, 1].axvline(omega_el, linestyle="--")
axes[1, 1].set_title("Detectability ratio R_n(ω)")
axes[1, 1].set_xlabel("ω")
axes[1, 1].set_ylabel("R_n(ω)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()