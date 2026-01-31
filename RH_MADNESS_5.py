# ------------------------------------------------------------
# CORE-FRAME prime reconstruction driven by zeta zeros (gammas)
# with TWO selectable gamma sources:
#   A) mp.zetazero(k)  (fastest + cleanest, no duplicates)
#   B) Hardy Z(t) scan (independent extraction via sign-scan + bisection)
#
# Includes:
# - siegeltheta phase-lock
# - integer scoring (boundary-safe for 2/3/5)
# - optional CVXOPT tail-weights (fallback to uniform)
# - vectorized core_field (gamma-matrix broadcast with chunking)
# - safe reporting (tail-only) + optional saving of outputs
#
# Adds HIGH-N WINDOW MODE:
# - runs around huge N (e.g. 1e15..1e17) without float(N+k) integer-grid collapse
# - uses log(N+k) = logN + log1p(k/N)
# - uses deterministic Miller–Rabin for 64-bit primality verification (no global sieve)
#
# deps:
#   pip install numpy matplotlib scipy mpmath
#   (optional) pip install cvxopt
# ------------------------------------------------------------

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import mpmath as mp

mp.mp.dps = 80


# ============================================================
# 0) SMALL HELPERS
# ============================================================

def tail(seq: Iterable, k: int):
    """Return last k items as a python list."""
    if k is None:
        return list(seq)
    k = int(k)
    if k <= 0:
        return []
    try:
        return list(seq[-k:])  # numpy arrays support this
    except Exception:
        lst = list(seq)
        return lst[-k:]


def fmt_tail(seq: Iterable, k: int, *, max_chars: int = 2000) -> str:
    """Human-friendly string for last k items + hard cap on chars."""
    s = ", ".join(map(str, tail(seq, k)))
    if len(s) > max_chars:
        s = s[:max_chars] + " ..."
    return s


def demo_float_failure(N: int = 10**17, samples: int = 10):
    """Show why float64-based scans fail to resolve integer steps at very high N."""
    N = int(N)
    print("-" * 60)
    print("FLOAT64 FAILURE DEMO")
    print(f"N={N}")
    print("float(N) == float(N+1):", float(N) == float(N + 1))
    print("float(N) == float(N+2):", float(N) == float(N + 2))
    fN = float(N)
    ulp = np.nextafter(fN, np.inf) - fN
    print("ULP step near N:", ulp, "≈", int(ulp), "integers per float step")
    same = [float(N + k) for k in range(samples)]
    print("float(N+k) for k=0..:", same)
    print("-" * 60)


# ============================================================
# 1) TRUE ZETA ZEROS ON CRITICAL LINE VIA HARDY Z(t)
# ============================================================

def hardy_Z(t: float | mp.mpf) -> mp.mpf:
    """
    Hardy Z function: real-valued on real t
    Z(t) = Re( exp(i*theta(t)) * zeta(1/2 + i t) )
    where theta(t) is the Riemann–Siegel theta (mp.siegeltheta).
    """
    tt = mp.mpf(t)
    return mp.re(mp.e ** (1j * mp.siegeltheta(tt)) * mp.zeta(mp.mpf("0.5") + 1j * tt))


def scan_Z_sign_changes(tmin: float, tmax: float, dt: float = 0.01):
    """
    Find brackets [a,b] where Z changes sign on [tmin,tmax].
    Notes:
      - dt too large can miss sign flips (use 0.005 or smaller if needed)
      - extremely small dt can be slow
    """
    t = mp.mpf(tmin)
    tmax = mp.mpf(tmax)
    dt = mp.mpf(dt)

    prev = hardy_Z(t)
    brackets: list[tuple[float, float]] = []

    while t < tmax:
        t2 = t + dt
        cur = hardy_Z(t2)

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
    Returns float root or None if bracket is invalid.
    """
    a = mp.mpf(a)
    b = mp.mpf(b)
    tol = mp.mpf(tol)

    fa = hardy_Z(a)
    fb = hardy_Z(b)

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

        if (fa > 0) != (fm > 0):
            b = m
            fb = fm
        else:
            a = m
            fa = fm

        if abs(b - a) < tol:
            return float((a + b) / 2)

    return float((a + b) / 2)


def dedupe_close(values: list[float], eps: float = 1e-6):
    """Collapse near-equal roots (duplicates from overlapping brackets)."""
    vals = sorted(values)
    out: list[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > eps:
            out.append(v)
    return out


def get_zeta_zeros_by_Z(tmax: float = 50.0, dt: float = 0.01, tol: float = 1e-12, tmin: float = 0.0):
    """
    Returns list of t such that zeta(1/2 + i t) = 0 (on critical line),
    found by sign changes of Hardy Z(t) + bisection refinement.
    """
    br = scan_Z_sign_changes(tmin, tmax, dt=dt)
    roots: list[float] = []
    for a, b in br:
        r = refine_root_bisect_Z(a, b, tol=tol)
        if r is not None and r > 1e-9:
            roots.append(r)

    roots = dedupe_close(roots, eps=max(1e-6, float(dt) * 0.2))
    return roots


def N_asymp(T: float) -> mp.mpf:
    """
    Riemann–von Mangoldt asymptotic counting function:
    N(T) ~ (T/2π) log(T/2π) - T/2π + 7/8
    """
    T = mp.mpf(T)
    return (T / (2 * mp.pi)) * mp.log(T / (2 * mp.pi)) - T / (2 * mp.pi) + mp.mpf("7") / 8


# ============================================================
# 2) GAMMA SOURCE SELECTOR
# ============================================================

def gammas_from_zetazero(n_zeros: int, start_index: int = 1) -> list[float]:
    """
    mpmath.zetazero(k) is version-dependent:
      - some builds return t (real)
      - others return 0.5 + i*t (mpc)
    We always return gamma = t (positive real).
    """
    gs: list[float] = []
    for k in range(start_index, start_index + n_zeros):
        z = mp.zetazero(k)
        try:
            g = mp.im(z) if hasattr(z, "imag") else z
        except Exception:
            g = z
        gs.append(float(g))
    return gs


def gammas_from_hardy_scan(
    n_zeros: int,
    tmax_start: float = 50.0,
    dt: float = 0.01,
    tol: float = 1e-12,
    grow_factor: float = 1.6,
    max_rounds: int = 12,
    tmin: float = 0.0,
) -> list[float]:
    """
    Use Hardy Z scan, increasing tmax until we collect at least n_zeros.
    This is slower than zetazero, but is an independent extraction path.
    """
    tmax = float(tmax_start)
    all_roots: list[float] = []

    for _ in range(max_rounds):
        roots = get_zeta_zeros_by_Z(tmax=tmax, dt=dt, tol=tol, tmin=tmin)
        all_roots = roots  # full list up to current tmax (deduped)
        if len(all_roots) >= n_zeros:
            break
        tmax *= float(grow_factor)

    all_roots = sorted(all_roots)
    out = all_roots[:n_zeros]

    if out:
        T = max(out)
        print(f"[HardyZ] found {len(out)} zeros up to ~{T:.6f} | N_asymp({T:.3f})≈{N_asymp(T)}")

    return out


# ============================================================
# 3) CORE-FRAME FIELD (VECTOR)
# ============================================================

def robust_norm(v: np.ndarray, q: float = 0.99) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.quantile(np.abs(v), q)
    return v / (s if s > 0 else 1.0)


def core_field_vec(
    x: np.ndarray,
    gammas: list[float],
    *,
    delta_k: float = 0.01,
    use_siegel_phase: bool = True,
    center_log: str = "xmin",  # "mean" | "xmin"
    weights: np.ndarray | None = None,
    chunk_gammas: int = 256,
) -> np.ndarray:
    """
    Vectorized CORE-FRAME field:
      psi(x) = Σ_i w_i * (sqrt(x)/sqrt(0.25+γ_i^2))
               * sinc( delta_k * (γ_i/2π) * (log x - logc) )
               * cos(γ_i log x - θ_i)
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("x musí být > 0 (kvůli log(x)).")

    gam = np.asarray(gammas, dtype=float)
    J = gam.size

    logx = np.log(x)
    logc = float(np.mean(logx)) if center_log == "mean" else float(np.min(logx))
    sqrtx = np.sqrt(x)

    if weights is None:
        w = np.ones(J, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size < J:
            w = np.resize(w, J)

    if use_siegel_phase:
        thetas = np.array([float(mp.siegeltheta(mp.mpf(g))) for g in gam], dtype=float)
    else:
        thetas = np.zeros(J, dtype=float)

    rho = np.sqrt(0.25 + gam * gam)  # (J,)
    field = np.zeros_like(x, dtype=float)

    chunk_gammas = max(1, int(chunk_gammas))
    for s in range(0, J, chunk_gammas):
        e = min(J, s + chunk_gammas)
        g = gam[s:e]                 # (C,)
        th = thetas[s:e]             # (C,)
        ww = w[s:e]                  # (C,)
        rr = rho[s:e]                # (C,)

        phase = g[:, None] * logx[None, :] - th[:, None]  # (C,N)
        sinc_arg = delta_k * (g / (2 * np.pi))[:, None] * (logx - logc)[None, :]
        sinc = np.sinc(sinc_arg)

        amp = sqrtx[None, :] / rr[:, None]
        field += np.sum((ww[:, None] * amp) * sinc * np.cos(phase), axis=0)

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
# 4) INTEGER SCORING + LIGHT FILTER
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


def light_filter(
    cands: np.ndarray,
    *,
    kill_squares: bool = True,
    kill_mod5: bool = True,
    kill_mod7: bool = True,
    kill_mod11: bool = True,
) -> np.ndarray:
    out = []
    for n in cands.astype(int):
        if n < 2:
            continue
        if n in (2, 3, 5, 7, 11):
            out.append(n)
            continue

        if n % 2 == 0 or n % 3 == 0:
            continue
        if kill_mod5 and n % 5 == 0:
            continue
        if kill_mod7 and n % 7 == 0:
            continue
        if kill_mod11 and n % 11 == 0:
            continue

        if kill_squares:
            r = int(np.sqrt(n))
            if r * r == n:
                continue

        out.append(n)

    return np.array(sorted(set(out)), dtype=int)


def primes_upto(n: int, *, return_sieve: bool = False):
    """Classic sieve up to n. If return_sieve=True returns (primes_list, sieve_bool)."""
    if n < 2:
        empty: list[int] = []
        if return_sieve:
            return empty, np.zeros(n + 1, dtype=bool)
        return empty

    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    lim = int(np.sqrt(n))
    for p in range(2, lim + 1):
        if sieve[p]:
            sieve[p * p:n + 1:p] = False

    primes = np.flatnonzero(sieve).astype(int).tolist()
    return (primes, sieve) if return_sieve else primes


# ============================================================
# 5) OPTIONAL: CVXOPT WEIGHTS
# ============================================================

def get_optimal_weights_cvxopt(gammas: list[float], delta_min: float = 12.0, delta_max: float = 1000.0) -> np.ndarray:
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
# 6) DETERMINISTIC 64-bit MILLER–RABIN (WINDOW VERIFICATION)
# ============================================================

def is_prime_u64(n: int) -> bool:
    """Deterministic Miller–Rabin for n < 2^64 (10^17 is safe)."""
    n = int(n)
    if n < 2:
        return False
    # quick small primes
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    # n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    # deterministické báze pro 64-bit
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        if a % n == 0:
            return True
        if not check(a):
            return False
    return True


# ============================================================
# 7) STANDARD RUNNER (SMALL/MEDIUM RANGES)
# ============================================================

def run_core_frame(
    *,
    x_min: int = 2,
    x_max: int = 60,
    num_points: int = 30000,
    n_zeros: int = 20,
    gamma_source: str = "zetazero",  # "zetazero" | "hardyZ"
    hardy_tmax_start: float = 50.0,
    hardy_dt: float = 0.01,
    hardy_tol: float = 1e-12,
    delta_k: float = 0.010,
    center_log: str = "xmin",
    use_siegel_phase: bool = True,
    use_cvxopt: bool = True,
    peak_height: float = 0.18,
    peak_distance: int = 12,
    score_top_k: int = 70,
    filter_kill_squares: bool = True,
    filter_kill_mod5: bool = True,
    filter_kill_mod7: bool = True,
    filter_kill_mod11: bool = True,
    chunk_gammas: int = 256,
    print_last: int = 50,
    save_primes: bool = True,
    save_dir: str = "./out_primes",
    out_prefix: str | None = None,
    plot: bool = True,
):
    t0 = time.time()

    # ---- gammas
    if gamma_source == "hardyZ":
        gammas = gammas_from_hardy_scan(
            n_zeros=n_zeros,
            tmax_start=hardy_tmax_start,
            dt=hardy_dt,
            tol=hardy_tol,
        )
    else:
        gammas = gammas_from_zetazero(n_zeros)

    if len(gammas) < n_zeros:
        print(f"[WARN] gammas only {len(gammas)} < requested {n_zeros}")

    # ---- weights
    weights = get_optimal_weights_cvxopt(gammas) if use_cvxopt else np.ones(len(gammas), dtype=float)

    # ---- continuum field
    x = np.linspace(x_min, x_max, num_points)
    psi = core_field_vec(
        x, gammas,
        delta_k=delta_k,
        use_siegel_phase=use_siegel_phase,
        center_log=center_log,
        weights=weights,
        chunk_gammas=chunk_gammas,
    )
    y = robust_norm(psi, q=0.99)

    pk_idx, pk_props = find_peaks(y, height=peak_height, distance=peak_distance)
    pk_x = x[pk_idx]
    pk_round = np.array([int(round(v)) for v in pk_x], dtype=int)
    pk_round = np.array(sorted(set(pk_round.tolist())), dtype=int)

    # ---- integer scoring
    ints = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1, dtype=float)
    psi_int = core_field_vec(
        ints, gammas,
        delta_k=delta_k,
        use_siegel_phase=use_siegel_phase,
        center_log=center_log,
        weights=weights,
        chunk_gammas=chunk_gammas,
    )

    score = integer_score(ints, psi_int, detrend_window=11)
    idx_sorted = np.argsort(score)[::-1]
    cands = ints[idx_sorted[:score_top_k]].astype(int)

    cands_light = light_filter(
        cands,
        kill_squares=filter_kill_squares,
        kill_mod5=filter_kill_mod5,
        kill_mod7=filter_kill_mod7,
        kill_mod11=filter_kill_mod11,
    )

    gt_primes, gt_sieve = primes_upto(int(x_max), return_sieve=True)

    # ---- causal/echo filter + verification against sieve
    detected_so_far: list[int] = []
    final_verified_primes: list[int] = []
    final_candidates: list[int] = []

    for n in sorted(set(cands_light.tolist())):
        if n < 2 or n > int(x_max):
            continue

        # echo: multiples of already confirmed primes
        is_echo = False
        for p in detected_so_far:
            if p != n and n % p == 0:
                is_echo = True
                break

        final_candidates.append(n)

        if not is_echo and bool(gt_sieve[n]):
            final_verified_primes.append(n)
            detected_so_far.append(n)

    # also include peak-rounded integers (optionally useful)
    pk_round_in_range = [n for n in pk_round.tolist() if x_min <= n <= x_max]
    pk_round_in_range = sorted(set(pk_round_in_range))

    # ---- reporting (tail-only)
    print("=" * 60)
    print(f"STANDARD: [{x_min}, {x_max}] | points={num_points} | gammas={len(gammas)} | delta_k={delta_k}")
    print(f"gamma_source={gamma_source} | siegel_phase={use_siegel_phase} | cvxopt={use_cvxopt}")
    print("-" * 60)
    print(f"peak-rounded ints (unique): {len(pk_round_in_range)} | tail: {fmt_tail(pk_round_in_range, print_last)}")
    print(f"score candidates (light-filter): {len(final_candidates)} | tail: {fmt_tail(final_candidates, print_last)}")
    print(f"verified primes (echo-filtered): {len(final_verified_primes)} | tail: {fmt_tail(final_verified_primes, print_last)}")
    print(f"runtime: {time.time() - t0:.3f}s")
    print("=" * 60)

    # ---- save outputs
    if save_primes:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = out_prefix or f"coreframe_{gamma_source}_x{int(x_min)}_{int(x_max)}_{stamp}"
        np.save(p / f"{prefix}_gammas.npy", np.array(gammas, dtype=float))
        np.save(p / f"{prefix}_weights.npy", np.array(weights, dtype=float))
        np.save(p / f"{prefix}_ints.npy", ints.astype(float))
        np.save(p / f"{prefix}_psi_int.npy", psi_int.astype(float))
        np.save(p / f"{prefix}_score.npy", score.astype(float))
        np.save(p / f"{prefix}_cands.npy", np.array(final_candidates, dtype=int))
        np.save(p / f"{prefix}_primes.npy", np.array(final_verified_primes, dtype=int))

    # ---- plot
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(x, y, linewidth=1.0)
        plt.scatter(pk_x, y[pk_idx], s=10)
        for n in final_verified_primes:
            plt.axvline(n, linewidth=0.7, alpha=0.5)
        plt.title("CORE-FRAME field (normalized) + peaks + verified primes (vertical)")
        plt.xlabel("x")
        plt.ylabel("psi(x) normalized")
        plt.tight_layout()
        plt.show()

    return {
        "gammas": gammas,
        "weights": weights,
        "x": x,
        "psi": psi,
        "y": y,
        "peaks_x": pk_x,
        "peaks_round": pk_round_in_range,
        "ints": ints,
        "psi_int": psi_int,
        "score": score,
        "candidates": final_candidates,
        "verified_primes": final_verified_primes,
    }


# ============================================================
# 8) HIGH-N WINDOW MODE (N..N+W)
# ============================================================

def core_field_window(
    N: int,
    W: int,
    gammas: list[float],
    *,
    delta_k: float = 0.001,
    use_siegel_phase: bool = True,
    center_log: str = "xmin",  # "mean" | "xmin"
    weights: np.ndarray | None = None,
    chunk_gammas: int = 256,
) -> np.ndarray:
    """
    Compute psi(N+k) for k=0..W without representing N+k as float.
    Uses log(N+k) = logN + log1p(k/N) (k is small, N huge).
    """
    N = int(N)
    W = int(W)
    if N <= 0 or W < 0:
        raise ValueError("N must be > 0 and W must be >= 0")

    # k are small and representable exactly in float64 up to 2^53
    """k = np.arange(0, W + 1, dtype=np.float64)
    invN = 1.0 / float(N)
    logN = float(mp.log(mp.mpf(N)))  # accurate logN

    logx = logN + np.log1p(k * invN)
    logc = float(np.mean(logx)) if center_log == "mean" else float(np.min(logx))

    # sqrt(N+k) = sqrt(N) * sqrt(1+k/N)
    sqrtN = np.sqrt(float(N))
    sqrtx = sqrtN * np.sqrt(1.0 + k * invN)"""

    k = np.arange(0, W + 1, dtype=np.float64)

    logN = float(mp.log(mp.mpf(N)))  # accurate logN

    try:
        fN = float(N)
        invN = 1.0 / fN
        sqrtN = np.sqrt(fN)
    except OverflowError:
        # N too large for float -> avoid crash
        invN = 0.0
        sqrtN = 1.0

    logx = logN + np.log1p(k * invN)
    logc = float(np.mean(logx)) if center_log == "mean" else float(np.min(logx))

    sqrtx = sqrtN * np.sqrt(1.0 + k * invN)

    gam = np.asarray(gammas, dtype=float)
    J = gam.size

    if weights is None:
        w = np.ones(J, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size < J:
            w = np.resize(w, J)

    if use_siegel_phase:
        thetas = np.array([float(mp.siegeltheta(mp.mpf(g))) for g in gam], dtype=float)
    else:
        thetas = np.zeros(J, dtype=float)

    rho = np.sqrt(0.25 + gam * gam)
    field = np.zeros(W + 1, dtype=float)

    chunk_gammas = max(1, int(chunk_gammas))
    for s in range(0, J, chunk_gammas):
        e = min(J, s + chunk_gammas)
        g = gam[s:e]
        th = thetas[s:e]
        ww = w[s:e]
        rr = rho[s:e]

        phase = g[:, None] * logx[None, :] - th[:, None]
        sinc_arg = delta_k * (g / (2 * np.pi))[:, None] * (logx - logc)[None, :]
        sinc = np.sinc(sinc_arg)
        amp = sqrtx[None, :] / rr[:, None]

        field += np.sum((ww[:, None] * amp) * sinc * np.cos(phase), axis=0)

    return field


def run_core_frame_window(
    *,
    N: int = 10**17,
    W: int = 2_000_000,
    n_zeros: int = 25,
    gamma_source: str = "zetazero",  # "zetazero" | "hardyZ"
    hardy_tmax_start: float = 50.0,
    hardy_dt: float = 0.005,
    hardy_tol: float = 1e-12,
    delta_k: float = 0.001,
    center_log: str = "xmin",
    use_siegel_phase: bool = True,
    use_cvxopt: bool = True,
    chunk_gammas: int = 256,
    score_top_k: int = 5000,
    detrend_window: int = 11,
    echo_filter: bool = True,
    print_last: int = 50,
    save: bool = True,
    save_dir: str = "./out_primes",
    out_prefix: str | None = None,
    x_min: int = 2,
    x_max: int = 60,
):
    """
    Window-mode runner:
      - computes psi on offsets k=0..W around huge N
      - scores offsets (peakness/residual/derivatives)
      - selects top-K candidate offsets
      - verifies primality via deterministic Miller–Rabin (u64)
    """
    t0 = time.time()

    # ---- gammas
    if gamma_source == "hardyZ":
        gammas = gammas_from_hardy_scan(
            n_zeros=n_zeros,
            tmax_start=hardy_tmax_start,
            dt=hardy_dt,
            tol=hardy_tol,
        )
    else:
        gammas = gammas_from_zetazero(n_zeros)

    weights = get_optimal_weights_cvxopt(gammas) if use_cvxopt else np.ones(len(gammas), dtype=float)

    psi = core_field_window(
        N, W, gammas,
        delta_k=delta_k,
        use_siegel_phase=use_siegel_phase,
        center_log=center_log,
        weights=weights,
        chunk_gammas=chunk_gammas,
    )

    offs = np.arange(0, W + 1, dtype=float)
    score = integer_score(offs, psi, detrend_window=detrend_window)

    # top-K offsets by score
    idx = np.argsort(score)[::-1][:int(score_top_k)]
    cand_offsets = idx.astype(int).tolist()
    candidates = [int(N + k) for k in cand_offsets]

    verified_primes: list[int] = []
    detected_primes: list[int] = []

    for n in candidates:
        if echo_filter:
            is_echo = False
            for p in detected_primes:
                if p != n and n % p == 0:
                    is_echo = True
                    break
            if is_echo:
                continue

        if is_prime_u64(n):
            verified_primes.append(n)
            detected_primes.append(n)

    # ---- reporting
    print("=" * 60)
    print(f"gammas={len(gammas)} | delta_k={delta_k} | topK={score_top_k}")
    print(f"gamma_source={gamma_source} | siegel_phase={use_siegel_phase} | cvxopt={use_cvxopt}")
    print(f"prime hits in topK: {len(verified_primes)} ")
    print(f"runtime: {time.time() - t0:.3f}s")
    print(f"WINDOW: [{N}, {N + W}]")
    print("-" * 60)
    #print(f"candidates tail {print_last}: {fmt_tail(candidates, print_last)}")
    #print(f"prime hits in topK: {len(verified_primes)} | tail {print_last}: {fmt_tail(verified_primes, print_last)}")
    print(verified_primes)

    #print("=" * 60)

    # ---- save outputs
    if save:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = out_prefix or f"coreframe_window_{gamma_source}_N{N}_W{W}_{stamp}"
        prefix = out_prefix or f"coreframe_{gamma_source}_x{int(x_min)}_{int(x_max)}_{stamp}"
        np.save(p / f"{prefix}_gammas.npy", np.array(gammas, dtype=float))
        np.save(p / f"{prefix}_weights.npy", np.array(weights, dtype=float))
        np.save(p / f"{prefix}_psi.npy", psi.astype(float))
        np.save(p / f"{prefix}_score.npy", score.astype(float))
        np.save(p / f"{prefix}_cand_offsets.npy", np.array(cand_offsets, dtype=int))
        #np.save(p / f"{prefix}_candidates.npy", np.array(candidates, dtype=np.int64))
        #np.save(p / f"{prefix}_primes.npy", np.array(verified_primes, dtype=np.int64))
        # --- BIG INT SAFE SAVE (N can exceed int64) ---
        # store as strings to avoid numpy int64 overflow
        np.save(p / f"{prefix}_candidates.npy", np.array([str(x) for x in candidates], dtype=str))
        np.save(p / f"{prefix}_primes.npy", np.array([str(x) for x in verified_primes], dtype=str))

    return {
        "N": int(N),
        "W": int(W),
        "gammas": gammas,
        "weights": weights,
        "psi": psi,
        "score": score,
        "cand_offsets": cand_offsets,
        "candidates": candidates,
        "verified_primes": verified_primes,
    }


# ============================================================
# 9) MAIN (DEMO DEFAULTS)
# ============================================================

if __name__ == "__main__":
    # 1) show float64 failure around 1e17
    #demo_float_failure(10**17)

    # 2) quick sanity standard small-range run
    #    (toggle plot=False if running headless)
    """run_core_frame(
        x_min=2,
        x_max=120,
        num_points=60000,
        n_zeros=25,
        gamma_source="zetazero",
        delta_k=0.010,
        score_top_k=80,
        plot=True,
        save_primes=True,
        save_dir="./out_primes",
    )"""

    # 3) high-N window test (start with 1e15 if you want faster iteration)
    #run_core_frame_window(N=10**15, W=2_000_000, n_zeros=25, gamma_source="zetazero", score_top_k=5000)
    run_core_frame_window(N=10**1000, W=1_000_000, n_zeros=5, gamma_source="zetazero", score_top_k=1_000_001)


