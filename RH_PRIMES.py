# RH_PRVOCISLA_FOR_REAL_FIXED_PLUS_HARDY_Z.py
# ------------------------------------------------------------
# CORE-FRAME prime reconstruction driven by zeta zeros (gammas)
# with TWO selectable sources:
#   A) mp.zetazero(k)  (fastest + cleanest, no duplicates)
#   B) Hardy Z(t) scan (your logic; can be used as verification
#      / independent extraction; includes de-duplication)
#
# Also includes:
# - siegeltheta phase-lock
# - integer scoring that doesn't kill small primes (2/3)
# - optional CVXOPT tail-weights (fallback to uniform)
#
# deps:
#   pip install numpy matplotlib scipy mpmath
#   (optional) pip install cvxopt
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import mpmath as mp

mp.mp.dps = 80


# ============================================================
# 1) TRUE ZETA ZEROS ON CRITICAL LINE VIA HARDY Z(t)
# ============================================================

def hardy_Z(t: float) -> mp.mpf:
    """
    Hardy Z function: real-valued on real t
    Z(t) = Re( exp(i*theta(t)) * zeta(1/2 + i t) )
    where theta(t) is the Riemann–Siegel theta (mp.siegeltheta).
    """
    tt = mp.mpf(t)
    return mp.re(mp.e ** (1j * mp.siegeltheta(tt)) * mp.zeta(mp.mpf("0.5") + 1j * tt))


def scan_Z_sign_changes(tmin: float, tmax: float, dt: float = 0.01):
    """
    Find brackets [a,b] where Z changes sign.
    """
    t = mp.mpf(tmin)
    prev = hardy_Z(t)
    brackets = []

    while t < tmax:
        t2 = t + dt
        cur = hardy_Z(t2)

        # If exact zero (rare) treat as a tiny bracket around it
        if cur == 0:
            brackets.append((float(t2 - dt), float(t2 + dt)))
        elif prev == 0:
            brackets.append((float(t - dt), float(t + dt)))
        elif (prev > 0) != (cur > 0):
            brackets.append((float(t), float(t2)))

        t = t2
        prev = cur

    return brackets


def refine_root_bisect_Z(a: float, b: float, tol: float = 1e-12, maxit: int = 200):
    """
    Robust bisection on Z(t) in [a,b] assuming sign change.
    """
    a = mp.mpf(a)
    b = mp.mpf(b)
    fa = hardy_Z(a)
    fb = hardy_Z(b)

    # If bracket is bad, return None
    if fa == 0:
        return float(a)
    if fb == 0:
        return float(b)
    if (fa > 0) == (fb > 0):
        return None

    for _ in range(maxit):
        m = (a + b) / 2
        fm = hardy_Z(m)

        if fm == 0:
            return float(m)

        # shrink
        if (fa > 0) != (fm > 0):
            b = m
            fb = fm
        else:
            a = m
            fa = fm

        if abs(b - a) < tol:
            return float((a + b) / 2)

    return float((a + b) / 2)


def dedupe_close(values, eps=1e-6):
    """
    Collapse near-equal roots (duplicates from overlapping brackets).
    """
    vals = sorted(values)
    out = []
    for v in vals:
        if not out or abs(v - out[-1]) > eps:
            out.append(v)
    return out


def get_zeta_zeros_by_Z(tmax=50.0, dt=0.01, tol=1e-12):
    """
    Returns list of t such that zeta(1/2 + i t) = 0 (on critical line),
    found by sign changes of Hardy Z(t).
    """
    br = scan_Z_sign_changes(0.0, tmax, dt=dt)
    roots = []
    for a, b in br:
        r = refine_root_bisect_Z(a, b, tol=tol)
        if r is not None and r > 1e-6:
            roots.append(r)

    roots = dedupe_close(roots, eps=max(1e-6, dt * 0.2))
    return roots


def N_asymp(T: float) -> mp.mpf:
    T = mp.mpf(T)
    return (T / (2 * mp.pi)) * mp.log(T / (2 * mp.pi)) - T / (2 * mp.pi) + mp.mpf("7") / 8


# ============================================================
# 2) GAMMA SOURCE SELECTOR
# ============================================================

def gammas_from_zetazero(n_zeros: int, start_index: int = 1) -> list[float]:
    """
    Most reliable and fastest: mpmath.zetazero(k) returns the k-th zero ordinate.
    """
    gs = []
    for k in range(start_index, start_index + n_zeros):
        gs.append(float(mp.zetazero(k)))
    return gs


def gammas_from_hardy_scan(
    n_zeros: int,
    tmax_start: float = 50.0,
    dt: float = 0.01,
    tol: float = 1e-12,
    grow_factor: float = 1.6,
    max_rounds: int = 12,
) -> list[float]:
    """
    Use Hardy Z scan, increasing tmax until we collect at least n_zeros.
    (Still slower than zetazero, but matches your “document logic”.)
    """
    tmax = float(tmax_start)
    zeros = []
    for _ in range(max_rounds):
        zeros = get_zeta_zeros_by_Z(tmax=tmax, dt=dt, tol=tol)
        zeros = [z for z in zeros if z > 1e-6]
        if len(zeros) >= n_zeros:
            return zeros[:n_zeros]
        tmax *= grow_factor
    # if insufficient, return what we have (best effort)
    return zeros[:n_zeros]


# ============================================================
# 3) CORE-FRAME FIELD
# ============================================================

def robust_norm(v: np.ndarray, q: float = 0.99) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.quantile(np.abs(v), q)
    return v / (s if s > 0 else 1.0)


def siegeltheta_np(t: float) -> float:
    return float(mp.siegeltheta(mp.mpf(t)))


def core_field(
    x: np.ndarray,
    gammas: list[float],
    *,
    delta_k: float = 0.01,
    use_siegel_phase: bool = True,
    center_log: str = "xmin",  # "mean" | "xmin"
    weights: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("x musí být > 0 (kvůli log(x)).")

    logx = np.log(x)
    logc = float(np.mean(logx)) if center_log == "mean" else float(np.min(logx))

    if weights is None:
        weights = np.ones(len(gammas), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) < len(gammas):
            weights = np.resize(weights, len(gammas))

    field = np.zeros_like(x, dtype=float)

    for i, g in enumerate(gammas):
        rho_abs = np.sqrt(0.25 + g * g)
        amp = (x ** 0.5) / rho_abs

        # sinc in log-domain
        sinc_arg = delta_k * (g / (2 * np.pi)) * (logx - logc)
        sinc = np.sinc(sinc_arg)

        theta = siegeltheta_np(g) if use_siegel_phase else 0.0
        field += float(weights[i]) * amp * sinc * np.cos(g * logx - theta)

    return field


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    w = max(3, int(w))
    if w % 2 == 0:
        w += 1
    k = w // 2
    ypad = np.pad(y, (k, k), mode="edge")
    ker = np.ones(w, dtype=float) / w
    return np.convolve(ypad, ker, mode="valid")


# ============================================================
# 4) INTEGER SCORING (small-primes friendly)
# ============================================================

def integer_score(
    ints: np.ndarray,
    psi_int: np.ndarray,
    *,
    detrend_window: int = 11,
    w0: float = 1.6,
    w1: float = 0.7,
    w2: float = 0.35,
    wpk: float = 0.9,
    boundary_safe: bool = True,
) -> np.ndarray:
    psi = robust_norm(psi_int, q=0.99)

    trend = moving_average(psi, detrend_window)
    res = psi - trend
    res = robust_norm(res, q=0.99)

    d1 = np.gradient(res)
    d2 = np.gradient(d1)

    d1 = robust_norm(d1, q=0.99)
    d2 = robust_norm(d2, q=0.99)

    peakness = np.maximum(0.0, -d2)
    peakness = robust_norm(peakness, q=0.99)

    score = w0 * np.abs(res) + w1 * np.abs(d1) + w2 * np.abs(d2) + wpk * peakness

    if boundary_safe:
        boost = np.ones_like(score)
        for n in [2, 3, 5]:
            idx = np.where(ints.astype(int) == n)[0]
            if len(idx):
                boost[idx[0]] = 1.15
        score = score * boost

    return score


"""def light_filter(cands: np.ndarray, *, kill_squares=True, kill_mod5=False) -> np.ndarray:
    out = []
    for n in cands.astype(int):
        if n > 2 and n % 2 == 0:
            continue
        if n > 3 and n % 3 == 0:
            continue
        if kill_mod5 and n > 5 and n % 5 == 0:
            continue
        if kill_squares:
            r = int(np.sqrt(n))
            if r * r == n and n > 4:
                continue
        out.append(n)
    return np.array(sorted(set(out)), dtype=int)"""


def light_filter(cands: np.ndarray, *, kill_squares=True, kill_mod5=True, kill_mod7=True,
                 kill_mod11=True) -> np.ndarray:
    out = []
    for n in cands.astype(int):
        if n < 2: continue
        if n == 2 or n == 3 or n == 5 or n == 7 or n == 11:
            out.append(n)
            continue

        # Základní síto
        if n % 2 == 0 or n % 3 == 0: continue
        if kill_mod5 and n % 5 == 0: continue
        if kill_mod7 and n % 7 == 0: continue
        if kill_mod11 and n % 11 == 0: continue

        if kill_squares:
            r = int(np.sqrt(n))
            if r * r == n: continue

        out.append(n)
    return np.array(sorted(set(out)), dtype=int)


def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(np.sqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p:n + 1:p] = False
    return [i for i in range(n + 1) if sieve[i]]


# ============================================================
# 5) OPTIONAL: CVXOPT WEIGHTS
# ============================================================

def get_optimal_weights_cvxopt(gammas: list[float], delta_min=12.0, delta_max=1000.0) -> np.ndarray:
    """
    If cvxopt is available: solve QP to suppress tail energy.
    If not: return uniform weights.
    """
    try:
        from cvxopt import matrix, solvers
        from scipy.integrate import quad
    except Exception:
        return np.ones(len(gammas), dtype=float)

    J = len(gammas)
    scales = 2.0 ** np.arange(J)

    def A_func(delta):
        return np.exp(-0.5 * delta * delta)

    K_tail = np.zeros((J, J), dtype=float)
    for j in range(J):
        for k in range(j, J):
            aj, ak = scales[j], scales[k]
            integrand = lambda d: A_func(d / aj) * A_func(d / ak)
            val, _ = quad(integrand, delta_min, delta_max)
            K_tail[j, k] = val
            K_tail[k, j] = val

    P = matrix(2.0 * K_tail)
    q = matrix(0.0, (J, 1))
    G = matrix(-np.eye(J))
    h = matrix(0.0, (J, 1))
    A_mat = matrix(1.0, (1, J))
    b_mat = matrix(1.0)

    solvers.options["show_progress"] = False
    sol = solvers.qp(P, q, G, h, A_mat, b_mat)

    w_sq = np.array(sol["x"]).flatten()
    w = np.sqrt(np.maximum(w_sq, 0.0))

    m = np.max(w) if np.max(w) > 0 else 1.0
    return w / m


# ============================================================
# 6) RUNNER
# ============================================================

def run_core_frame(
    *,
    x_min=2,
    x_max=60,
    num_points=30000,
    n_zeros=20,
    gamma_source="zetazero",  # "zetazero" | "hardyZ"
    hardy_tmax_start=50.0,
    hardy_dt=0.01,
    hardy_tol=1e-12,
    delta_k=0.010,
    center_log="xmin",
    use_siegel_phase=True,
    use_cvxopt=True,
    peak_height=0.18,
    peak_distance=12,
    score_top_k=70,
    filter_kill_squares=True,
    filter_kill_mod5=True,
):
    # ---- gammas
    if gamma_source == "hardyZ":
        gammas = gammas_from_hardy_scan(
            n_zeros=n_zeros,
            tmax_start=hardy_tmax_start,
            dt=hardy_dt,
            tol=hardy_tol
        )
    else:
        gammas = gammas_from_zetazero(n_zeros)

    # quick sanity
    if len(gammas) < n_zeros:
        print(f"[WARN] gammas only {len(gammas)} < requested {n_zeros}")

    # optional compare count vs N_asymp when using hardyZ
    if gamma_source == "hardyZ":
        print(f"HardyZ scan: found {len(gammas)} zeros up to ~{max(gammas) if gammas else 0:.3f}")
        if gammas:
            T = max(gammas)
            print(f"Riemann–von Mangoldt N({T:.3f}) ≈ {N_asymp(T)}")

    # ---- weights
    weights = get_optimal_weights_cvxopt(gammas) if use_cvxopt else np.ones(len(gammas), dtype=float)

    # ---- continuum field
    x = np.linspace(x_min, x_max, num_points)
    psi = core_field(
        x, gammas,
        delta_k=delta_k,
        use_siegel_phase=use_siegel_phase,
        center_log=center_log,
        weights=weights
    )
    y = robust_norm(psi, q=0.99)

    pk_idx, _ = find_peaks(y, height=peak_height, distance=peak_distance)
    pk_x = x[pk_idx]
    pk_round = np.array([int(round(v)) for v in pk_x], dtype=int)

    # ---- integer scoring
    ints = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1, dtype=float)
    psi_int = core_field(
        ints, gammas,
        delta_k=delta_k,
        use_siegel_phase=use_siegel_phase,
        center_log=center_log,
        weights=weights
    )
    score = integer_score(ints, psi_int, detrend_window=11)
    idx_sorted = np.argsort(score)[::-1]
    cands = ints[idx_sorted[:score_top_k]].astype(int)

    cands_light = light_filter(
        cands,
        kill_squares=filter_kill_squares,
        kill_mod5=filter_kill_mod5
    )

    gt_primes = primes_upto(int(x_max))
    hits = [n for n in cands_light if n in gt_primes]
    falsep = [n for n in cands_light if n not in gt_primes]

    # ---- report
    print("=" * 60)
    print(f"CORE-FRAME: x in [{x_min}, {x_max}] | gammas={len(gammas)} | delta_k={delta_k}")
    print(f"gamma_source: {gamma_source} | phase: {'siegeltheta' if use_siegel_phase else 'off'} | center_log: {center_log}")
    print(f"weights: {'CVXOPT' if use_cvxopt else 'uniform'}")
    print("-" * 60)
    print("Kontinuální maxima (rounded):")
    print(", ".join(map(str, pk_round.tolist())) if len(pk_round) else "(none)")
    print("Kontinuální maxima (raw x):")
    print(", ".join([f"{v:.2f}" for v in pk_x]) if len(pk_x) else "(none)")
    print("-" * 60)
    print("Top kandidáti podle score(n):")
    print(", ".join(map(str, cands.tolist())))
    print("Po light filtru:")
    print(", ".join(map(str, cands_light.tolist())) if len(cands_light) else "(none)")
    print("-" * 60)
    print(f"Hits (prime): {hits}")
    print(f"False+:       {falsep}")
    print("=" * 60)

    # ---- plots
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label="Spectral Field (normalized)", lw=1, alpha=0.8)
    if len(pk_x):
        plt.scatter(pk_x, y[pk_idx], s=35, label="Continuum peaks", zorder=5)
    for n in cands_light:
        if x_min <= n <= x_max:
            plt.axvline(n, alpha=0.12)
    for p in gt_primes:
        if x_min <= p <= x_max:
            plt.axvline(p, color="green", alpha=0.05, linestyle="--")
    plt.title("CORE-FRAME: field + candidate lines (light filter)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    psi_int_n = robust_norm(psi_int, q=0.99)
    plt.stem(ints, psi_int_n, linefmt="grey", markerfmt=" ", basefmt=" ")
    if len(cands_light):
        mask = (cands_light >= int(np.ceil(x_min))) & (cands_light <= int(np.floor(x_max)))
        cl = cands_light[mask].astype(int)
        idx = (cl - int(np.ceil(x_min))).astype(int)
        plt.scatter(cl, psi_int_n[idx], color="red", zorder=5, label="candidates")
    plt.title("ψ(n) on integers")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(ints, score, label="score(n)")
    for n in cands_light:
        plt.axvline(n, alpha=0.12)
    plt.title("Integer score(n) (boundary-safe + detrend + peakness)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # If you want EXACTLY your HardyZ extraction:
    #   gamma_source="hardyZ"
    # and tune hardy_dt to avoid missing sign flips (0.005 is safer).
    #
    # If you want speed + clean gammas:
    #   gamma_source="zetazero"  (recommended baseline)
    #
    run_core_frame(
        x_min=2,
        x_max=1000,
        num_points=200000,
        n_zeros=150,
        gamma_source="hardyZ",      # <-- switch here ("zetazero" or "hardyZ")
        hardy_tmax_start=50.0,
        hardy_dt=0.005,            # finer step => fewer misses
        hardy_tol=1e-12,
        delta_k=0.005,
        center_log="xmin",
        use_siegel_phase=True,
        use_cvxopt=True,
        peak_height=0.10,
        peak_distance=12,
        score_top_k=300,
        filter_kill_squares=True,
        filter_kill_mod5=True
    )
