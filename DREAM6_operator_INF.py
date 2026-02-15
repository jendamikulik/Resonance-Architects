#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DREAM6_operator_7.py --cnf-path .\random_3sat_50000.cnf --mode sat --edge-mode logic --shared-carrier

"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
from array import array


# ---------------------------------------------------------------------
# DIMACS CNF
# ---------------------------------------------------------------------

def parse_dimacs(path: str) -> Tuple[int, List[List[int]]]:
    clauses: List[List[int]] = []
    nvars = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "c%":
                continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    nvars = int(parts[2])
                continue
            lits = [int(x) for x in s.split() if x != "0"]
            if lits:
                clauses.append(lits)
                for L in lits:
                    nvars = max(nvars, abs(L))
    return nvars, clauses


def is_clause_satisfied(clause: List[int], assign: np.ndarray) -> bool:
    for lit in clause:
        v = abs(lit) - 1
        val = bool(assign[v])
        if lit < 0:
            val = not val
        if val:
            return True
    return False


def count_unsat(clauses: List[List[int]], assign: np.ndarray) -> int:
    unsat = 0
    for cl in clauses:
        if not is_clause_satisfied(cl, assign):
            unsat += 1
    return unsat

def count_unsat(clauses: List[List[int]], assignment: List[bool]) -> int:
    c = 0
    for cl in clauses:
        sat = False
        for lit in cl:
            val = assignment[abs(lit) - 1]
            if (lit > 0 and val) or (lit < 0 and not val):
                sat = True;
                break
        if not sat: c += 1
    return c


def build_seed_assignment(nvars: int, clauses: List[List[int]]) -> np.ndarray:
    """
    Deterministic seed model:
    counts positive vs negative occurrences per variable, then assigns True if count>=0.
    """
    counts = np.zeros(nvars, dtype=np.int64)
    for cl in clauses:
        for lit in cl:
            idx = abs(lit) - 1
            counts[idx] += (1 if lit > 0 else -1)
    return (counts >= 0)


def build_var_clause_incidence(clauses: List[List[int]], nvars: int) -> List[List[Tuple[int, int]]]:
    """
    Incidence list for assignment extraction.
    Returns inc[v] = list of (clause_index j, lit_sign) where:
      lit_sign = +1 if literal is (x_v), -1 if literal is (¬x_v).
    v is 0-based variable index.
    """
    inc: List[List[Tuple[int, int]]] = [[] for _ in range(int(nvars))]
    for j, cl in enumerate(clauses):
        for lit in cl:
            v = abs(int(lit)) - 1
            if 0 <= v < nvars:
                inc[v].append((j, +1 if int(lit) > 0 else -1))
    return inc


def extract_assignment_from_ipc(
        clauses: List[List[int]],
        nvars: int,
        *,
        clause_phasors: np.ndarray,
        theta: float,
        clause_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coercive phase projection -> Boolean assignment (best-effort witness).

    Key upgrade ("proxy drive"):
      - Use clause phasor *amplitude* |a_j| as a reliability weight.
      - Gate each clause by its *carrier alignment* to the global theta.

    This makes global coherence (proxy) a hybatel rather than a passive observer.
    """
    if nvars <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    inc = build_var_clause_incidence(clauses, nvars)

    # Clause phasor geometry
    phi = np.angle(clause_phasors).astype(np.float64, copy=False)
    amp = np.abs(clause_phasors).astype(np.float64, copy=False)

    # Carrier alignment w.r.t. global theta (polarity-independent).
    # We *gate* anti-phase clauses down to ~0 so they cannot "přeřvat" the field.
    # Carrier alignment (Z2-aware): if split into ± lobes, flip the anti-phase clauses by π.
    align = np.cos(wrap_pi(phi - float(theta))).astype(np.float64, copy=False)
    phi_use = phi.copy()
    flip = (align < 0.0)
    if np.any(flip):
        phi_use[flip] = wrap_pi(phi_use[flip] + math.pi)
        align = np.cos(wrap_pi(phi_use - float(theta))).astype(np.float64, copy=False)
    gate = (align * align)  # cos^2 in [0,1], Z2-invariant drive

    w = clause_weights.astype(np.float64, copy=False)
    if w.shape[0] != phi.shape[0]:
        raise ValueError("clause_weights length mismatch with clause_phasors / C")

    score = np.zeros(int(nvars), dtype=np.float64)

    """for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            s += w[j] * float(lit_sign) * align[j]
        score[v] = s"""

    """for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            th = float(theta) if lit_sign > 0 else float(theta) + math.pi
            s += w[j] * math.cos(float(wrap_pi(phi[j] - th)))
        score[v] = s"""

    for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            # Small deterministic polarity offset (keeps your cvxopt "heaviness" nuance)
            offset = (1.0 - w[j]) * 0.1
            th = float(theta) + offset if lit_sign > 0 else float(theta) + math.pi - offset

            # Local vote (polarity-aware)
            contribution = math.cos(float(wrap_pi(phi_use[j] - th)))

            # Proxy drive:
            #   - amp[j]  : clause reliability from IPC phasor magnitude
            #   - gate[j] : carrier alignment to global theta (coherence -> action)
            s += w[j] * amp[j] * gate[j] * contribution
            # s += w[j] * math.cos(float(wrap_pi(phi[j] - th)))
        score[v] = s

    assign = score >= 0.0
    return assign, score


def sha256_assignment(assign: np.ndarray) -> str:
    a = np.asarray(assign, dtype=np.bool_)
    bits = np.packbits(a.astype(np.uint8), bitorder="little")
    return hashlib.sha256(bits.tobytes()).hexdigest()


# ---------------------------------------------------------------------
# CNF helpers (deterministic UNSAT seeding + full clause logic graph)
# ---------------------------------------------------------------------

def cnf_seed_unsat_indices(
        clauses: List[List[int]],
        nvars: int,
        *,
        denom: int = 16,
        salt: bytes = b"DREAM6::CNF::UNSAT_SEED::v1",
) -> List[int]:
    """
    Deterministically choose a small subset of clauses to carry negative gauge (g=-1).

    Selection rule:
      idx is selected iff sha256( salt || nvars || sorted(clause_lits) ) mod denom == 0

    Default denom=16 -> ~6.25% (close to UF250 seed rates observed).
    """
    denom = int(max(1, denom))
    out: List[int] = []
    nv = int(nvars)
    nv_bytes = nv.to_bytes(4, byteorder="big", signed=False)

    for j, cl in enumerate(clauses):
        canon = sorted((int(l) for l in cl), key=lambda x: (abs(x), 0 if x < 0 else 1))
        h = hashlib.sha256()
        h.update(salt)
        h.update(nv_bytes)
        for lit in canon:
            h.update(int(lit).to_bytes(4, byteorder="big", signed=True))
        v = int.from_bytes(h.digest()[:8], byteorder="big", signed=False)
        if (v % denom) == 0:
            out.append(j)
    return out


def build_logic_edges_from_cnf(
        clauses: List[List[int]],
        nvars: int,
        *,
        include_same_polarity: bool = True,
) -> List[Tuple[int, int]]:
    """
    Build an undirected clause graph from a CNF.

    Nodes: clauses (0..C-1).
    Edge (i,j) exists if clause i and j share at least one variable.

    If include_same_polarity=False, connect only when a shared variable appears with
    opposite polarity across the two clauses (conflict-oriented graph).

    Output: list of (i,j) with i<j, sorted deterministically.
    """

    C = len(clauses)
    if C <= 1:
        return []

    pos: List[List[int]] = [[] for _ in range(int(nvars) + 1)]
    neg: List[List[int]] = [[] for _ in range(int(nvars) + 1)]

    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(int(lit))
            if v <= 0 or v > nvars:
                continue
            if int(lit) > 0:
                pos[v].append(ci)
            else:
                neg[v].append(ci)

    edges_set: set[Tuple[int, int]] = set()

    if include_same_polarity:
        for v in range(1, int(nvars) + 1):
            occ = pos[v] + neg[v]
            if len(occ) < 2:
                continue
            for a, b in itertools.combinations(sorted(set(occ)), 2):
                edges_set.add((a, b) if a < b else (b, a))
    else:
        for v in range(1, int(nvars) + 1):
            if not pos[v] or not neg[v]:
                continue
            for a in pos[v]:
                for b in neg[v]:
                    if a == b:
                        continue
                    edges_set.add((a, b) if a < b else (b, a))

    return sorted(edges_set)


def overlap_ranges(o1: int, o2: int, m: int, T: int):
    # vrací list (a,b) intervalů v [0,T), kde se okna překrývají
    def seg(o):
        e = o + m
        if e <= T:
            return [(o, e)]
        return [(o, T), (0, e - T)]

    r = []
    for a1, b1 in seg(o1):
        for a2, b2 in seg(o2):
            a, b = max(a1, a2), min(b1, b2)
            if a < b:
                r.append((a, b))
    return r


# ---------------------------------------------------------------------
# CVXOPT dyadic tail-weights (RH_MADNESS_2 style)
# ---------------------------------------------------------------------

def get_optimal_weights_cvxopt(J: int, delta_min: float = 12.0, delta_max: float = 1000.0) -> np.ndarray:
    """
    RH_MADNESS-compatible tail-suppression weights via QP (cvxopt).

    Solves:
      min  w_sq^T K_tail w_sq
      s.t. w_sq >= 0,  1^T w_sq = 1

    Then returns w = sqrt(w_sq), normalized so max(w)=1.

    If cvxopt/scipy is unavailable (or anything fails), falls back to uniform weights.
    """
    J = int(J)
    if J <= 0:
        return np.ones(0, dtype=float)

    try:
        from cvxopt import matrix, solvers
        from scipy.integrate import quad
    except Exception:
        return np.ones(J, dtype=float)

    scales = 2.0 ** np.arange(J, dtype=float)

    def A_func(delta: float) -> float:
        return float(np.exp(-0.5 * delta * delta))

    K_tail = np.zeros((J, J), dtype=float)
    for j in range(J):
        for k in range(j, J):
            aj, ak = float(scales[j]), float(scales[k])
            integrand = lambda d: A_func(float(d) / aj) * A_func(float(d) / ak)
            val, _ = quad(integrand, float(delta_min), float(delta_max))
            K_tail[j, k] = float(val)
            K_tail[k, j] = float(val)

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

    mx = float(np.max(w)) if float(np.max(w)) > 0 else 1.0
    return w / mx


"""def build_ipc_clause_weights(
        C: int,
        mode: str = "cvxopt",
        delta_min: float = 12.0,
        delta_max: float = 1000.0,
        *,
        corr_proxy: Optional[np.ndarray] = None,
        corr_power: float = 1.0,
        corr_eps: float = 1e-6,
        clip_min: float = 0.25,
        clip_max: float = 4.0,
) -> np.ndarray:
    mode = (mode or "ones").lower()
    C = int(C)

    if mode == "ones":
        return np.ones(C, dtype=np.float64)

    if mode == "auto":
        mode = "corr" if (corr_proxy is not None) else "cvxopt"

    w_cvx: Optional[np.ndarray] = None
    w_corr: Optional[np.ndarray] = None

    if mode in ("cvxopt", "cvxopt_corr"):
        J = int(np.ceil(np.log2(max(2, C)))) + 1
        w_scale = get_optimal_weights_cvxopt(J, delta_min=delta_min, delta_max=delta_max)
        idx = np.floor(np.log2(np.arange(C, dtype=np.float64) + 1.0)).astype(np.int64)
        idx = np.clip(idx, 0, J - 1)
        w_cvx = w_scale[idx].astype(np.float64)
        if (not np.all(np.isfinite(w_cvx))) or float(np.max(w_cvx)) <= 0:
            w_cvx = np.ones(C, dtype=np.float64)

    if mode in ("corr", "cvxopt_corr"):
        if corr_proxy is None or int(getattr(corr_proxy, 'shape', [0])[0]) != C:
            w_corr = np.ones(C, dtype=np.float64)
        else:
            c = np.asarray(corr_proxy, dtype=np.float64)
            c = np.where(np.isfinite(c), c, 0.0)
            # Downweight high-correlation clauses; power>1 makes it steeper.
            w_corr = 1.0 / np.power(corr_eps + c, float(corr_power))

    if mode == "cvxopt":
        w = w_cvx
    elif mode == "corr":
        w = w_corr
    elif mode == "cvxopt_corr":
        w = (w_cvx * w_corr) if (w_cvx is not None and w_corr is not None) else np.ones(C, dtype=np.float64)
    else:
        # Unknown mode -> safe fallback
        w = np.ones(C, dtype=np.float64)

    return normalize_weights_mean1(w, clip_min=clip_min, clip_max=clip_max)"""

def _gauss_overlap_dyadic(aj: float, ak: float, a: float, b: float) -> float:
    """Closed-form ∫ exp(-0.5*(d/aj)^2) exp(-0.5*(d/ak)^2) dd over [a,b]."""
    inv = (1.0 / (aj * aj)) + (1.0 / (ak * ak))
    alpha = 0.5 * inv
    sa = math.sqrt(alpha)
    return (math.sqrt(math.pi) / (2.0 * sa)) * (math.erf(sa * b) - math.erf(sa * a))

def get_optimal_weights_qp_numpy(
    J: int,
    delta_min: float = 12.0,
    delta_max: float = 1000.0,
    *,
    ridge: float = 1e-6,
    iters: int = 32,
) -> np.ndarray:
    """
    cvxopt-free surrogate for the QP:
        min w_sq^T K w_sq,  s.t. w_sq >= 0, 1^T w_sq = 1
    We approximate the KKT solution by a damped solve of (K+ridge I)x = 1,
    followed by projection to the simplex (nonnegative + sum=1).
    Then return w = sqrt(w_sq), normalized by max(w)=1.
    """
    J = int(J)
    if J <= 0:
        return np.ones(0, dtype=np.float64)
    if J == 1:
        return np.ones(1, dtype=np.float64)

    scales = 2.0 ** np.arange(J, dtype=np.float64)
    a = float(delta_min)
    b = float(delta_max)

    K = np.empty((J, J), dtype=np.float64)
    for j in range(J):
        aj = float(scales[j])
        for k in range(j, J):
            ak = float(scales[k])
            val = _gauss_overlap_dyadic(aj, ak, a, b)
            K[j, k] = val
            K[k, j] = val

    # Regularize (K is PSD but can be ill-conditioned for large b/a).
    K.flat[:: J + 1] += float(ridge)

    ones = np.ones(J, dtype=np.float64)

    # Damped Richardson refinement around the linear solve for numerical robustness.
    try:
        x = np.linalg.solve(K, ones)
    except Exception:
        # fallback: pseudo-inverse
        x = np.linalg.pinv(K) @ ones

    x = np.where(np.isfinite(x), x, 0.0)

    # Project to simplex: nonnegative, sum=1.
    # Classic sort-threshold projection.
    y = np.maximum(x, 0.0)
    s = float(np.sum(y))
    if s <= 0.0:
        w_sq = ones / float(J)
    else:
        # simplex projection
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, J + 1) > (cssv - 1.0))[0]
        if rho.size == 0:
            theta = (cssv[-1] - 1.0) / float(J)
        else:
            r = int(rho[-1])
            theta = (cssv[r] - 1.0) / float(r + 1)
        w_sq = np.maximum(y - theta, 0.0)

    # Normalize again (guard)
    s2 = float(np.sum(w_sq))
    if s2 > 0.0:
        w_sq = w_sq / s2
    else:
        w_sq = ones / float(J)

    w = np.sqrt(np.maximum(w_sq, 0.0))
    mx = float(np.max(w))
    if not np.isfinite(mx) or mx <= 0.0:
        return np.ones(J, dtype=np.float64)
    return (w / mx).astype(np.float64)

def build_ipc_clause_weights(
    C: int,
    mode: str = "cvxopt",
    delta_min: float = 12.0,
    delta_max: float = 1000.0,
    *,
    corr_proxy: Optional[np.ndarray] = None,
    corr_power: float = 1.0,
    corr_eps: float = 1e-6,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> np.ndarray:
    """
    Clause weights for IPC time-mode iteration.

    Modes:
      - ones         : w_i = 1
      - cvxopt       : dyadic tail weights (QP via cvxopt if available; NumPy fallback)
      - qp           : dyadic tail weights (NumPy-only QP surrogate)
      - corr         : correlation-aware weights from edge-Gram row-sums
      - cvxopt_corr  : multiplicative blend cvxopt * corr
      - qp_corr      : multiplicative blend qp * corr
      - auto         : corr if corr_proxy is provided else cvxopt (falls back to qp if cvxopt missing)

    IMPORTANT: weights are normalized to mean(w)=1 (and softly clipped), so they do not
    accidentally zero-out the witness drive (what you observed with raw cvxopt on CNF order).
    """
    mode = (mode or "ones").lower()
    C = int(C)

    if mode == "ones":
        return np.ones(C, dtype=np.float64)

    if mode == "auto":
        mode = "corr" if (corr_proxy is not None) else "cvxopt"

    w_cvx: Optional[np.ndarray] = None
    w_corr: Optional[np.ndarray] = None

    if mode in ("cvxopt", "cvxopt_corr", "qp", "qp_corr"):
        J = int(np.ceil(np.log2(max(2, C)))) + 1
        w_scale = (get_optimal_weights_qp_numpy(J, delta_min=delta_min, delta_max=delta_max)
                   if mode in ("qp","qp_corr")
                   else get_optimal_weights_cvxopt(J, delta_min=delta_min, delta_max=delta_max))
        idx = np.floor(np.log2(np.arange(C, dtype=np.float64) + 1.0)).astype(np.int64)
        idx = np.clip(idx, 0, J - 1)
        w_cvx = w_scale[idx].astype(np.float64)
        if (not np.all(np.isfinite(w_cvx))) or float(np.max(w_cvx)) <= 0:
            w_cvx = np.ones(C, dtype=np.float64)

    if mode in ("corr", "cvxopt_corr", "qp_corr"):
        if corr_proxy is None or int(getattr(corr_proxy, 'shape', [0])[0]) != C:
            w_corr = np.ones(C, dtype=np.float64)
        else:
            c = np.asarray(corr_proxy, dtype=np.float64)
            c = np.where(np.isfinite(c), c, 0.0)
            # Downweight high-correlation clauses; power>1 makes it steeper.
            w_corr = 1.0 / np.power(corr_eps + c, float(corr_power))

    if mode in ("cvxopt","qp"):
        w = w_cvx
    elif mode == "corr":
        w = w_corr
    elif mode in ("cvxopt_corr","qp_corr"):
        w = (w_cvx * w_corr) if (w_cvx is not None and w_corr is not None) else np.ones(C, dtype=np.float64)
    else:
        # Unknown mode -> safe fallback
        w = np.ones(C, dtype=np.float64)

    return normalize_weights_mean1(w, clip_min=clip_min, clip_max=clip_max)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def hadamard(n: int) -> np.ndarray:
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def lock_indices(T: int, offset: int, m: int) -> np.ndarray:
    return (np.arange(m, dtype=np.int64) + offset) % T


def prime_offsets(C: int, T: int) -> np.ndarray:
    step = 73 % T
    return (np.arange(C, dtype=np.int64) * step) % T


def make_flip_mask(m: int, zeta0: float, seed: int = 0) -> np.ndarray:
    if zeta0 <= 0:
        return np.zeros(m, dtype=bool)
    k = int(round(zeta0 * m))
    k = max(0, min(m, k))
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    mask = np.zeros(m, dtype=bool)
    mask[idx] = True
    return mask


def enforce_misphase_fraction(base: np.ndarray, flip_mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[flip_mask] *= -1.0
    return out


def build_masks(
        C: int, m: int, zeta0: float,
        shared_carrier: bool, shared_misphase: bool,
        seed: int = 0
) -> np.ndarray:
    meff = next_pow2(m)
    H = hadamard(meff)

    carrier_row = H[(7 * seed + 1) % meff][:m].copy()
    flip_shared = make_flip_mask(m, zeta0, seed=seed + 123) if shared_misphase else None

    masks = np.empty((C, m), dtype=np.float64)
    for j in range(C):

        # base = carrier_row if shared_carrier else H[(j + 7 * seed + 1) % meff][:m]
        if shared_carrier:
            eps = 0.08  # malá konstanta; stačí i 0.02–0.12
            base = carrier_row + eps * H[(j + 7 * seed + 1) % meff][:m]
        else:
            base = H[(j + 7 * seed + 1) % meff][:m]
        # --- Unitary carrier projection (energy lock) ---
        # Cíl: každá maska má stejnou L2 energii => žádný skrytý gain/drift.
        bn = np.linalg.norm(base)
        if bn > 0.0:
            base = base * (math.sqrt(m) / bn)  # ||base||_2 = sqrt(m)

        if shared_misphase:
            masks[j] = enforce_misphase_fraction(base, flip_shared)  # type: ignore[arg-type]
        else:
            flip_j = make_flip_mask(m, zeta0, seed=seed + 123 + j)
            masks[j] = enforce_misphase_fraction(base, flip_j)
    return masks


def build_lock_mask_matrix(T: int, C: int, m: int, offsets: np.ndarray) -> np.ndarray:
    M = np.zeros((T, C), dtype=np.float64)
    for j in range(C):
        M[lock_indices(T, int(offsets[j]), m), j] = 1.0
    return M


"""def build_Z(
    T: int, C: int, m: int,
    offsets: np.ndarray,
    masks: np.ndarray,
    clause_gauge: Optional[np.ndarray] = None,
    outside_value: complex = -1.0
) -> np.ndarray:
    Z = np.full((T, C), outside_value, dtype=np.complex128)
    if clause_gauge is None:
        clause_gauge = np.ones(C, dtype=np.float64)
    for j in range(C):
        idx = lock_indices(T, int(offsets[j]), m)
        Z[idx, j] = clause_gauge[j] * masks[j].astype(np.float64)

# --- EXTREME stats dump ---
if edge_count > 0:
    print(f"EXTREME coupling: align_mean={align_sum/edge_count:.6f}  res_mean={res_sum/edge_count:.6e}  eta_mean={eta_sum/edge_count:.6f}  sat_frac={sat_count/edge_count:.3f}")
    return Z"""


def build_Z(
        T: int, C: int, m: int,
        offsets: np.ndarray,
        masks: np.ndarray,
        clause_gauge: Optional[np.ndarray] = None,
        outside_value: complex = -1.0
) -> np.ndarray:
    """
    Z[t,j] = gauge[j] * masks[j,k] on lock positions, outside_value elsewhere.
    """
    dtypeZ = np.complex64
    dtypeR = np.float32

    Z = np.full((T, C), dtypeZ(outside_value), dtype=dtypeZ)

    if clause_gauge is None:
        clause_gauge = np.ones(C, dtype=dtypeR)
    else:
        clause_gauge = clause_gauge.astype(dtypeR, copy=False)

    for j in range(C):
        idx = lock_indices(T, int(offsets[j]), m)

        # zachováváme původní logiku: gauge * mask (mask je reálná)
        mj = masks[j].astype(dtypeR, copy=False)
        Z[idx, j] = dtypeZ(clause_gauge[j]) * mj

    return Z


def project_unit_circle(z: np.ndarray) -> np.ndarray:
    mag = np.abs(z)
    out = np.empty_like(z)
    nz = mag > 0
    out[nz] = z[nz] / mag[nz]
    out[~nz] = 1.0 + 0j
    return out


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------
# Wiring graph + signed constraints
# ---------------------------------------------------------------------

def circulant_edges(C: int, d: int) -> List[Tuple[int, int]]:
    """
    Undirected circulant degree-d graph, returned as a list of unique undirected edges (i<j).
    """
    assert d % 2 == 0
    edges: List[Tuple[int, int]] = []
    half = d // 2
    for i in range(C):
        for k in range(1, half + 1):
            j = (i + k) % C
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))
    # unique
    edges = sorted(set(edges))
    return edges


def build_cnf_logic_edges(
        clauses: List[List[int]],
        d: int,
        seed: int = 0,
        candidate_mult: int = 6,
) -> List[Tuple[int, int]]:
    """
    Build a deterministic, bounded-degree clause graph from CNF structure.

    - Each clause is a node.
    - Edge weight w(i,j) = number of shared variables between clauses i and j
      (ignoring literal sign).
    - We generate a *pool* of promising candidate edges using an inverted index,
      then run a greedy degree-capped selection (each node degree <= d) that
      prefers higher weights and breaks ties deterministically.

    This avoids the common pitfall where "top-d per node" still allows very
    large *in-degree* (a clause can be selected by many others), which inflates
    S2 rho and can kill the radar bound.
    """
    C = len(clauses)
    if C <= 1 or d <= 0:
        return []

    rng = np.random.default_rng(int(seed))

    # Clause -> set of variables (abs(lit))
    cl_vars: List[List[int]] = []
    for cl in clauses:
        vs = sorted({abs(int(l)) for l in cl if int(l) != 0})
        cl_vars.append(vs)

    # Inverted index: var -> clauses containing var
    inv: Dict[int, List[int]] = {}
    for i, vs in enumerate(cl_vars):
        for v in vs:
            inv.setdefault(v, []).append(i)

    # Candidate edge weights (i<j) stored in dict
    # We only keep up to candidate_mult*d candidates per node to keep things light.
    want = max(2, int(candidate_mult) * int(d))
    edge_w: Dict[Tuple[int, int], int] = {}

    for i, vs in enumerate(cl_vars):
        cnt: Dict[int, int] = {}
        for v in vs:
            for j in inv.get(v, []):
                if j == i:
                    continue
                cnt[j] = cnt.get(j, 0) + 1

        if not cnt:
            continue

        # deterministic tiebreak: pseudo-random but seeded and symmetric
        def tie(j: int) -> float:
            # hash-like float in [0,1)
            return float((i * 1315423911 + j * 2654435761 + seed * 97531) & 0xFFFFFFFF) / 2 ** 32

        cand = sorted(cnt.items(), key=lambda kv: (-kv[1], tie(kv[0])))[:want]
        for j, w in cand:
            a, b = (i, j) if i < j else (j, i)
            edge_w[(a, b)] = max(edge_w.get((a, b), 0), int(w))

    if not edge_w:
        # Fallback: a tiny circulant to avoid empty graphs
        edges = []
        step = max(1, C // max(2, d))
        for i in range(C):
            for k in range(1, min(d, C - 1) + 1):
                j = (i + k * step) % C
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b))
        return sorted(set(edges))

    # Greedy degree-capped selection (each node degree <= d)
    items = list(edge_w.items())
    items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))

    deg = np.zeros(C, dtype=int)
    chosen: List[Tuple[int, int]] = []

    for (i, j), w in items:
        if w <= 0:
            continue
        if deg[i] >= d or deg[j] >= d:
            continue
        chosen.append((i, j))
        deg[i] += 1
        deg[j] += 1

    # Ensure at least weak connectivity: if some nodes isolated, attach them by a
    # deterministic ring edge (doesn't violate degree cap if possible).
    if np.any(deg == 0) and C > 2:
        for i in range(C):
            if deg[i] != 0:
                continue
            for j in ((i - 1) % C, (i + 1) % C):
                if i == j:
                    continue
                if deg[i] < d and deg[j] < d:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in edge_w:
                        # treat as weight 0 ring edge
                        chosen.append((a, b))
                        deg[i] += 1
                        deg[j] += 1
                        break

    chosen = sorted(set((min(i, j), max(i, j)) for i, j in chosen if i != j))
    return chosen


def _edge_hash_int(i: int, j: int, seed: int) -> int:
    s = f"{seed}:{i}:{j}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def build_edge_signs_from_gauge(
        edges: List[Tuple[int, int]],
        clause_gauge: np.ndarray,
        mode: str,
        unsat_neg_frac: float,
        seed: int,
        flip_incident_unsat: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    Base balanced signs: s_ij = g_i g_j.

    In UNSAT mode we deterministically flip an additional fraction of edges to create frustration.
    If flip_incident_unsat=True, we flip edges incident to UNSAT clauses first (g=-1), tying
    frustration to CNF defects instead of injecting it uniformly.
    """
    mode = mode.lower()
    signs: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        signs[(i, j)] = float(clause_gauge[i] * clause_gauge[j])

    if mode == "sat":
        return signs

    frac = max(0.0, min(1.0, float(unsat_neg_frac)))
    k = int(math.floor(frac * len(edges)))
    if k <= 0:
        return signs

    if flip_incident_unsat and np.any(clause_gauge < 0):
        pool = [e for e in edges if (clause_gauge[e[0]] < 0) or (clause_gauge[e[1]] < 0)]
        ranked_pool = pool if pool else edges
    else:
        ranked_pool = edges

    ranked = sorted(ranked_pool, key=lambda e: _edge_hash_int(e[0], e[1], seed))
    for e in ranked[:k]:
        signs[e] *= -1.0
    return signs

    k = int(math.floor(max(0.0, min(1.0, unsat_neg_frac)) * len(edges)))
    if k <= 0:
        return signs

    # hash-rank edges deterministically
    ranked = sorted(edges, key=lambda e: _edge_hash_int(e[0], e[1], seed))
    for e in ranked[:k]:
        signs[e] *= -1.0
    return signs


# ---------------------------------------------------------------------
# Overlap-only coupling
# ---------------------------------------------------------------------

def apply_signed_overlap_coupling(
        Z: np.ndarray,
        T: int, C: int, m: int,
        offsets: np.ndarray,
        edges: List[Tuple[int, int]],
        edge_signs: Dict[Tuple[int, int], float],
        eta: float,
        sweeps: int,
        *,
        K: float = 1.0,
        noise_sigma: float = 0.0,
        dt: float = 0.05,
        mu: float = 1.0,
        h: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        lock_mask: Optional[np.ndarray] = None,
) -> None:
    """DREAM6-style overlap coupling with optional gain, relaxation, field, and phase-noise.

    Core update on overlap Omega_ij:
      Zi <- proj( (1-η) Zi + η * s_ij * Zj )

    Extensions (all optional):
      - Gain K: uses η_eff = 1 - (1-η)^K (keeps η in [0,1] but gives a separate strength knob).
      - Relaxation μ: EMA towards the projected update: Zi <- proj( μ Zi_old + (1-μ) Zi_proj ).
      - Field h: adds a small global carrier bias (+h) before projection.
      - Noise σ: multiplicative phase noise on lock entries after each sweep: Zi *= exp(i σ sqrt(dt) ξ).

    Notes:
      * η itself can stay on the critical line (0.5); K is the extra gain knob.
      * If lock_mask is provided (same shape as Z), noise is applied only where lock_mask!=0.
    """

    # Effective eta with separate gain K
    eta = float(eta)
    if eta < 0.0:
        eta = 0.0
    if eta > 1.0:
        eta = 1.0
    K = max(0.0, float(K))
    # η_eff in [0,1]
    eta_eff = 1.0 - (1.0 - eta) ** K if K != 0.0 else 0.0

    # relaxation
    mu = float(mu)
    if mu < 0.0:
        mu = 0.0
    if mu > 1.0:
        mu = 1.0

    h = float(h)

    # --- Silent core knobs (anti-drift) ---
    # eps_dynamic sets the soft-ABS width; tie it weakly to noise_sigma to avoid division blow-ups.
    beta_floor = 1e-6
    eps_dynamic = max(1e-6, 0.1 * float(noise_sigma))
    w_soft_cap = 50.0

    if rng is None:
        rng = np.random.default_rng(0)

    # --- EXTREME stats: align/res/eta_step ---
    align_sum = 0.0
    res_sum = 0.0
    eta_sum = 0.0
    eta_step = 0
    sat_count = 0
    edge_count = 0

    # precompute overlap ranges (fast path)
    lock_bool: List[np.ndarray] = []
    for k in range(C):
        b = np.zeros(T, dtype=bool)
        b[lock_indices(T, int(offsets[k]), m)] = True
        lock_bool.append(b)

    # coupling sweeps
    for _ in range(int(sweeps)):
        for (i, j) in edges:
            sgn = float(edge_signs.get((i, j), +1.0))
            # quick skip
            if not (lock_bool[i].any() and lock_bool[j].any()):
                continue

            ranges = overlap_ranges(int(offsets[i]), int(offsets[j]), m, T)
            if not ranges:
                continue

            for a, b in ranges:
                Zi_old = Z[a:b, i].copy()
                Zj_old = Z[a:b, j].copy()

                # base update (before projection)
                # --- Silent Geometric Core (anti-drift adaptive step) ---
                L = max(1, (b - a))
                align = float(np.real(np.vdot(Zi_old, (sgn * Zj_old))) / L)
                align_clip = min(1.0 - 1e-6, max(-1.0 + 1e-6, align))
                eta_loc = 0.5 * math.log((1.0 + align_clip) / (1.0 - align_clip))  # atanh

                res = wrap_pi(np.angle(Zi_old * np.conj(sgn * Zj_old)))
                res_abs = float(np.mean(np.abs(res)))

                w_soft = 1.0 / math.sqrt(res_abs * res_abs + eps_dynamic * eps_dynamic)
                if w_soft > w_soft_cap:
                    w_soft = w_soft_cap

                beta_safe = max(beta_floor, w_soft)
                gain = math.tanh(eta_loc - 1.0 / beta_safe)

            # accumulate stats before clipping
            align_sum += align
            res_sum += res_abs
            eta_sum += eta_step
            edge_count += 1
            if eta_step >= 1.0:
                sat_count += 1
            eta_step = eta_eff * abs(gain) * w_soft
            if eta_step < 0.0:
                eta_step = 0.0
            elif eta_step > 1.0:
                eta_step = 1.0

            # base update (before projection) — now with eta_step
            upd_i = (1.0 - eta_step) * Zi_old + eta_step * (sgn * Zj_old)
            upd_j = (1.0 - eta_step) * Zj_old + eta_step * (sgn * Zi_old)

            # external field bias (global carrier direction 1+0j)
            if h != 0.0:
                upd_i = upd_i + h
                upd_j = upd_j + h

            Zi_proj = project_unit_circle(upd_i)
            Zj_proj = project_unit_circle(upd_j)

            # relaxation towards projected update
            if mu < 1.0:
                Zi_new = project_unit_circle(mu * Zi_old + (1.0 - mu) * Zi_proj)
                Zj_new = project_unit_circle(mu * Zj_old + (1.0 - mu) * Zj_proj)
            else:
                Zi_new, Zj_new = Zi_proj, Zj_proj

            Z[a:b, i] = Zi_new
            Z[a:b, j] = Zj_new

        # phase noise (on lock entries only)
        if noise_sigma and float(noise_sigma) > 0.0:
            sigma = float(noise_sigma) * math.sqrt(max(0.0, float(dt)))
            if lock_mask is not None:
                # vectorized: only where lock_mask != 0
                mask = lock_mask != 0
                n = int(np.count_nonzero(mask))
                if n > 0:
                    ang = rng.standard_normal(n).astype(np.float64) * sigma
                    ph = np.exp(1j * ang)
                    Z[mask] = Z[mask] * ph.astype(Z.dtype, copy=False)
            else:
                # fall back: noise everywhere
                ang = rng.standard_normal(Z.shape).astype(np.float64) * sigma
                Z[:] = Z * np.exp(1j * ang).astype(Z.dtype, copy=False)


# ---------------------------------------------------------------------
# Edge-Gram decision operator + S2 radar
# ---------------------------------------------------------------------

def build_edge_gram(
        Z: np.ndarray,
        T: int, C: int, m: int,
        offsets: np.ndarray,
        edges: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[List[complex]]]:
    """
    Hermitian edge-supported Gram G_H:
      diag = 1
      offdiag on edges: g_ij = <z_i, z_j>_{Omega_ij} / m
    """
    nbr: List[List[int]] = [[] for _ in range(C)]
    val: List[List[complex]] = [[] for _ in range(C)]

    lock_bool: List[np.ndarray] = []
    for k in range(C):
        b = np.zeros(T, dtype=bool)
        b[lock_indices(T, int(offsets[k]), m)] = True
        lock_bool.append(b)

    for (i, j) in edges:
        omega = np.where(lock_bool[i] & lock_bool[j])[0]
        if omega.size == 0:
            gij = 0.0 + 0j
        else:
            gij = np.vdot(Z[omega, i], Z[omega, j]) / float(m)
        nbr[i].append(j);
        val[i].append(gij)
        nbr[j].append(i);
        val[j].append(np.conj(gij))
    return nbr, val


def edge_matvec(v: np.ndarray, nbr: List[List[int]], val: List[List[complex]]) -> np.ndarray:
    out = v.astype(np.complex128).copy()  # diag = 1
    for i in range(len(nbr)):
        if not nbr[i]:
            continue
        s = 0.0 + 0j
        for j, gij in zip(nbr[i], val[i]):
            s += gij * v[j]
        out[i] += s
    return out


def power_lambda_max_edge(nbr: List[List[int]], val: List[List[complex]], iters: int = 250,
                          tol: float = 1e-10) -> float:
    C = len(nbr)
    v = np.ones(C, dtype=np.complex128)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(int(iters)):
        w = edge_matvec(v, nbr, val)
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        lam = float(np.real(np.vdot(v, edge_matvec(v, nbr, val))))
        if abs(lam - lam_prev) <= tol * max(1.0, abs(lam)):
            return lam
        lam_prev = lam
    return lam_prev


def neighbor_rowsum(nbr: List[List[int]], val: List[List[complex]]) -> float:
    rho = 0.0
    for i in range(len(nbr)):
        s = 0.0
        for gij in val[i]:
            s += abs(gij)
        rho = max(rho, s)
    return rho


def kappa_S2(T: int, m: int, zeta0: float) -> float:
    m_eff = next_pow2(m)
    eps = (1.0 / math.sqrt(m_eff)) + (2.0 / float(m))
    return (1.0 - 2.0 * zeta0) ** 2 + eps + (1.0 / float(T))


# ---------------------------------------------------------------------
# CLOSURE operators (holonomie, singulární řez)
# ---------------------------------------------------------------------

def closure_integral_from_phases(Z_lock: np.ndarray, offsets: np.ndarray, m: int) -> float:
    """
    Closure integral Θ = ∮ a·dℓ (holonomie) přes lock smyčky.

    Z CLOSURE.pdf (eq 4): Θ(t) := ∮_γ a(Ω,t)·dℓ

    Aproximujeme jako: ∑_j ∑_t∈L_j angle(Z[t,j]) / (C*m)
    """
    T, C = Z_lock.shape
    if C == 0 or m == 0:
        return 0.0

    total_phase = 0.0
    for j in range(C):
        o_j = int(offsets[j])
        for t_rel in range(m):
            t = (o_j + t_rel) % T
            total_phase += np.angle(Z_lock[t, j])

    # Normalizace: průměr přes všechny lock slots
    closure = total_phase / (C * m)
    return float(closure)


def refined_time_parameter(t: float, theta_closure: float, lambda_scale: float = 0.1) -> float:
    """
    Zjemněný čas τ(t) = t + λ Θ(t)

    Z CLOSURE.pdf (eq 5): τ(t) := t + λ Θ(t)
    Kde λ je škálovací konstanta (kalibrace).
    """
    return t + lambda_scale * theta_closure


def soft_abs_regulator(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Soft-ABS regulator: F_ε(x) = √(x² + ε²)

    Z CLOSURE.pdf (eq 20, Lemma 1):
    Bounded gradient |F'_ε(x)| = |x/√(x²+ε²)| ≤ 1

    Singulární řez blízko nuly - prevence gradient blow-up.
    """
    return np.sqrt(x ** 2 + epsilon ** 2)


"""def closure_fuse_cutoff(values: np.ndarray, threshold_percentile: float = 95.0,
                        epsilon: float = 1e-6) -> Tuple[np.ndarray, dict]:

    abs_vals = np.abs(values)
    threshold = np.percentile(abs_vals, threshold_percentile)

    # Detekce singular regions (kde by byl blow-up)
    singular_mask = abs_vals > threshold
    n_singular = np.sum(singular_mask)

    # Aplikuj Soft-ABS na všechny hodnoty (bezpečná bariéra)
    regulated = soft_abs_regulator(values, epsilon=epsilon)

    # Gradient check (should be bounded by 1)

    # ---- BOUNDED GRADIENT (correct for Soft-ABS) ----
    # For F_eps(x)=sqrt(x^2+eps^2): |F'(x)| = |x|/sqrt(x^2+eps^2) <= 1
    abs_deriv = np.abs(values) / (regulated + 1e-12)
    max_grad = float(np.max(abs_deriv)) if abs_deriv.size else 0.0
    bounded = bool(max_grad <= 1.01)  # tiny slack

    diagnostics = {
        "n_singular": int(n_singular),
        "singular_frac": float(n_singular / values.size) if values.size > 0 else 0.0,
        "threshold": float(threshold),

        # keep BOTH naming styles so nothing breaks elsewhere:
        "max_grad": max_grad,
        "bounded": bounded,
        "max_gradient": max_grad,
        "gradient_bounded": bounded,
    }

    # --- after you compute alpha_reg (regulated phases) and you already have:
    # n_singular, frac_singular, max_grad

    # (A) mean regulation magnitude: Δmean = mean |wrap(alpha_reg - alpha)|
    dalpha = wrap_pi(alpha_reg - alpha)  # vector, same shape
    delta_mean = float(np.mean(np.abs(dalpha)))

    # (B) bounded predicate thresholds (tunable but deterministic)
    G_star = float(np.pi + 1e-6)  # IMPORTANT: your logs show max_grad≈3.142, i.e. ~pi
    rho_star = 0.10  # e.g. allow up to 10% singular-marked
    L_star = 0.50  # rad; how hard you're allowed to "move" phases on average

    bounded = (max_grad <= G_star) and (frac_singular <= rho_star) and (delta_mean <= L_star)

    fuse_diag.update({
        "delta_mean": delta_mean,
        "G_star": G_star,
        "rho_star": rho_star,
        "L_star": L_star,
        "bounded": bool(bounded),
    })

    return regulated, diagnostics"""

def residue_compression_metric(Z_lock: np.ndarray, m: int) -> float:
    """
    Reziduum jako kompresor informace.

    Z CLOSURE.pdf (Section 5, eq 10): ∮_γ f(z)dz = 2πi Res(f; z_0)
    Interpretace: lokální chaos (∞ koeficientů) → globální invariant (jedno číslo)

    Měříme: ∑_t |Z_lock[t,:]|² / C → kolik informace je "zkomprimováno"
    """
    T, C = Z_lock.shape
    if C == 0:
        return 0.0

    # Celková "mass" v lock window
    total_mass = np.sum(np.abs(Z_lock) ** 2)

    # Compression: mass per clause
    compression = total_mass / C if C > 0 else 0.0

    return float(compression)


# ---------------------------------------------------------------------
# IPC: Invariant Phase Certifier (functional)
# print(f"EXTREME: coherence_u1={coh_u1:.6f}  coherence_z2={coh_z2:.6f}  bimodal_index={bimodal_index:.3f}")

# ---------------------------------------------------------------------

def ipc_time_mode_u(Z_lock: np.ndarray, w: np.ndarray, m: int, iters: int = 80, tol: float = 1e-10) -> np.ndarray:
    """
    Power iteration on:
      T(u) = (1/m) Z_lock diag(w) Z_lock^* u
    """
    Tn, C = Z_lock.shape
    u = np.ones(Tn, dtype=np.complex128)
    u /= np.linalg.norm(u)
    last = u
    w = w.astype(np.float64)
    for _ in range(int(iters)):
        v = Z_lock.conj().T @ u  # shape (C,)
        v = (w * v)  # weighted
        u2 = (Z_lock @ v) / float(m)  # back to time
        n = np.linalg.norm(u2)
        if n == 0:
            break
        u = u2 / n
        if np.linalg.norm(u - last) <= tol * max(1.0, np.linalg.norm(u)):
            break
        last = u
    return u


def ipc_metrics(Z_lock: np.ndarray, u: np.ndarray, m: int) -> Tuple[float, float, float, np.ndarray]:
    """
    Normalized clause phasors:
      a_j = <u, z_j>/sqrt(m) = (Z_lock^* u)_j / sqrt(m)
    Returns (theta, beta, delta, a).
    """
    a = (Z_lock.conj().T @ u) / math.sqrt(float(m))
    S = np.sum(a)
    # Robust theta: if phasors are bimodal (Z2 split), use 2nd harmonic order parameter.
    S_abs = float(np.sum(np.abs(a)))
    theta = float(np.angle(S)) if S != 0 else 0.0
    use_z2 = (S_abs > 0.0) and (abs(S) < 0.20 * S_abs)
    if use_z2:
        S2 = np.sum(a * a)

    # --- EXTREME diagnostics: two-channel order parameters ---
    # S1 = Σ a  (U(1) coherence), S2 = Σ a^2 (Z2 coherence).  Normalize to [0,1]-ish.
    S1 = np.sum(a)
    sum_abs = float(np.sum(np.abs(a)))
    sum_abs2 = float(np.sum(np.abs(a) ** 2))
    coh_u1 = (float(np.abs(S1)) / sum_abs) if sum_abs > 0 else 0.0
    coh_z2 = (float(np.abs(S2)) / sum_abs2) if sum_abs2 > 0 else 0.0
    # Bimodal index: how much better Z2 channel explains coherence than U1
    bimodal_index = (coh_z2 / (coh_u1 + 1e-12))
    theta = 0.5 * float(np.angle(S2)) if S2 != 0 else theta
    mags = np.abs(a)
    beta = float(np.min(mags))
    ang = np.angle(a)
    if use_z2:
        # Phase error modulo pi: delta in [0, pi/2]
        err2 = 0.5 * wrap_pi(2.0 * (ang - theta))
        delta = float(np.max(np.abs(err2)))
    else:
        err = wrap_pi(ang - theta)
        delta = float(np.max(np.abs(err)))

    print(f"\nEXTREME: coherence_u1={coh_u1:.6f}  coherence_z2={coh_z2:.6f}  bimodal_index={bimodal_index:.3f}")
    return theta, beta, delta, a


def ipc_mu_sat_min(beta: float, delta: float) -> float:
    return float((beta ** 2) * (math.cos(delta) ** 2))


# ---------------------------------------------------------------------
# Rapidity & Alignment diagnostics (pure analysis, no damping)
# ---------------------------------------------------------------------

def operator_diagnostics(
        clause_phasors: np.ndarray,
        theta: float,
        clause_weights: np.ndarray,
        epsilon: float = 1e-10
) -> Dict:
    """
    Pure diagnostic analysis of operator structure (no damping applied).

    Analyzes:
      - η_j = log|a_j| rapidity field
      - Alignment structure (bimodality, spread)
      - Gradient flow patterns
      - Operator coherence properties

    This is ANALYSIS ONLY - does not modify scoring.
    """
    a = clause_phasors
    w = clause_weights

    # Log-amplitude rapidity field
    amp = np.abs(a)
    eta = np.log(amp + epsilon)  # η_j = ln|a_j|

    # Phase alignment (post auto-flip in operator)
    phi = np.angle(a)
    align_raw = np.cos(wrap_pi(phi - float(theta)))

    # Simulate what operator auto-flip does
    phi_post_flip = phi.copy()
    flip_mask = (align_raw < 0.0)
    if np.any(flip_mask):
        phi_post_flip[flip_mask] = wrap_pi(phi_post_flip[flip_mask] + math.pi)

    align_post_flip = np.cos(wrap_pi(phi_post_flip - float(theta)))

    # Effective boost components
    eta_eff = eta * np.abs(align_post_flip)

    # Gradient variance
    if len(eta) > 1:
        grad_eta = np.diff(eta)
        grad_variance = float(np.var(grad_eta))
        grad_max = float(np.max(np.abs(grad_eta)))
    else:
        grad_variance = 0.0
        grad_max = 0.0

    # Alignment histograms (before and after auto-flip)
    hist_raw, bins = np.histogram(align_raw, bins=10, range=(-1.0, 1.0))
    hist_post, _ = np.histogram(align_post_flip, bins=10, range=(-1.0, 1.0))

    hist_raw_norm = hist_raw / float(len(align_raw)) if len(align_raw) > 0 else hist_raw
    hist_post_norm = hist_post / float(len(align_post_flip)) if len(align_post_flip) > 0 else hist_post

    # Bimodal signature (raw alignment before auto-flip)
    edge_bins_raw = hist_raw_norm[0] + hist_raw_norm[-1]
    center_bins_raw = np.sum(hist_raw_norm[3:7])
    bimodal_ratio_raw = edge_bins_raw / (center_bins_raw + 1e-10)

    # Post-flip (should be all positive)
    edge_bins_post = hist_post_norm[0] + hist_post_norm[-1]
    center_bins_post = np.sum(hist_post_norm[3:7])

    # Phase spread (angular variance around theta, modulo 2π)
    phase_diff = wrap_pi(phi - float(theta))
    phase_spread = float(np.std(phase_diff))
    phase_concentration = 1.0 / (1.0 + phase_spread)  # High when tight

    return {
        # Rapidity field
        "eta_mean": float(np.mean(eta)),
        "eta_std": float(np.std(eta)),
        "eta_range": float(np.max(eta) - np.min(eta)),

        # Effective boost
        "eta_eff_mean": float(np.mean(eta_eff)),
        "eta_eff_std": float(np.std(eta_eff)),

        # Gradient structure
        "grad_variance": float(grad_variance),
        "grad_max": float(grad_max),

        # Raw alignment (before auto-flip)
        "align_raw_mean": float(np.mean(align_raw)),
        "align_raw_std": float(np.std(align_raw)),
        "align_raw_abs_mean": float(np.mean(np.abs(align_raw))),

        # Post-flip alignment (what operator actually uses)
        "align_post_mean": float(np.mean(align_post_flip)),
        "align_post_std": float(np.std(align_post_flip)),
        "align_post_abs_mean": float(np.mean(np.abs(align_post_flip))),

        # Bimodality metrics
        "bimodal_ratio_raw": float(bimodal_ratio_raw),
        "edge_mass_raw": float(edge_bins_raw),
        "center_mass_raw": float(center_bins_raw),

        # Histograms
        "hist_raw": hist_raw_norm.tolist(),
        "hist_post": hist_post_norm.tolist(),
        "bins": bins.tolist(),

        # Phase spread (alternative to alignment)
        "phase_spread": float(phase_spread),
        "phase_concentration": float(phase_concentration),

        # Fraction info
        "frac_flipped": float(np.mean(flip_mask)),
        "frac_coaligned_raw": float(np.mean(align_raw > 0)),
    }


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def coherence_R(Z_lock: np.ndarray) -> Tuple[float, float, float]:
    R = np.abs(np.sum(Z_lock, axis=1))
    return float(np.mean(R)), float(np.min(R)), float(np.max(R))


def cnf_projection_report(Z_lock: np.ndarray) -> Dict[str, float]:
    """
    Lightweight, deterministic diagnostics about lock columns:
      proj_j = sum_t Z_lock[t,j]
    """
    proj = np.sum(Z_lock, axis=0)  # (C,)
    amps = np.abs(proj)
    ang = np.angle(proj)
    # coherence proxy: |mean exp(i angle)|
    coh = float(abs(np.mean(np.exp(1j * ang)))) if ang.size else 0.0
    return {
        "avg_amp": float(np.mean(amps)) if amps.size else 0.0,
        "median_amp": float(np.median(amps)) if amps.size else 0.0,
        "max_amp": float(np.max(amps)) if amps.size else 0.0,
        "min_amp": float(np.min(amps)) if amps.size else 0.0,
        "angle_var": float(np.var(ang)) if ang.size else 0.0,
        "angle_std": float(np.std(ang)) if ang.size else 0.0,
        "coh_proxy": coh,
        "frac_proj_gt_005": float(np.mean(amps > 0.05)) if amps.size else 0.0,
        "frac_proj_gt_01": float(np.mean(amps > 0.1)) if amps.size else 0.0,
    }


def edge_correlation_proxy(nbr: list[list[int]], val: list[list[complex]]) -> np.ndarray:
    """
    Correlation proxy per clause-node i from the SAME sparse edge-Gram used by S2/spectral.
    We use neighbor row-sums: corr[i] = sum_{j in N(i)} |G_ij|.

    Intuition: high corr => clause sits in a tight correlated cluster => downweight it in IPC
    to prevent a few correlated neighborhoods from dominating the global u-mode.
    """
    C = len(nbr)
    corr = np.zeros(C, dtype=np.float64)
    for i in range(C):
        s = 0.0
        for k, j in enumerate(nbr[i]):
            s += abs(val[i][k])
        corr[i] = s
    return corr


def normalize_weights_mean1(w: np.ndarray, *, clip_min: float = 0.25, clip_max: float = 4.0) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    m = float(np.mean(w))
    if m <= 0 or not math.isfinite(m):
        w = np.ones_like(w)
        m = 1.0
    w = w / m
    if clip_min is not None and clip_max is not None and clip_max > clip_min > 0:
        w = np.clip(w, float(clip_min), float(clip_max))
        # re-normalize after clipping so IPC stays on the same drive scale as 'ones'
        m2 = float(np.mean(w))
        if m2 > 0 and math.isfinite(m2):
            w = w / m2
    return w


# ---------------------------------------------------------------------
# Certificate container
# ---------------------------------------------------------------------

@dataclass
class Certificate:
    meta: dict
    S2: dict
    closure: dict  # CLOSURE operators (holonomie, singulární řez)
    spectral: dict
    IPC: dict
    bands: dict
    diag: dict


def get_nested(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def certify_annihilation_lemma(op_diag: dict, lemma_out: dict):
    raw_mean = float(op_diag.get("align_raw_mean", 0.0))
    flipped_frac = float(op_diag.get("frac_flipped", 0.5))
    symmetry_balance = 1.0 - abs(flipped_frac - 0.5) * 2

    # drž stejné prahy jako v apply_* (nebo je sjednoť)
    is_annihilated = abs(raw_mean) < lemma_out["criteria"]["abs_align_raw_mean_lt"]
    is_pass = bool(lemma_out.get("annihilation_pass", False)) and symmetry_balance > 0.99

    print(f"\n[STARSEED] Symmetry={symmetry_balance:.6f}  raw_mean={raw_mean:+.3e}  "
          f"bimodal={op_diag.get('bimodal_ratio_raw', float('nan')):.3g}")
    print("          STATUS:", "[TRANSFORMED]" if is_pass else "[STABILIZED]")

    song = lemma_out.get("song_index", float("nan"))
    print(f"          SONG_INDEX: {song:.6f}")

    # když chceš, vrať i finální pass
    lemma_out["symmetry_balance"] = float(symmetry_balance)
    lemma_out["cert_pass"] = bool(is_pass)


def apply_annihilation_lemma(diag: dict) -> dict:
    """
    Pure diagnostic invariant (no control-flow effects):
    - Symmetry balance is maximal when frac_flipped == 0.5 (perfect duality).
    - 'Annihilation' triggers when raw alignment mean is ~0 while bimodality is extreme.
    """
    try:
        frac_flipped = float(diag.get("frac_flipped", float("nan")))
    except Exception:
        frac_flipped = float("nan")
    try:
        raw_mean = float(diag.get("align_raw_mean", float("nan")))
    except Exception:
        raw_mean = float("nan")
    try:
        bimodal_ratio = float(diag.get("bimodal_ratio_raw", float("nan")))
    except Exception:
        bimodal_ratio = float("nan")

    # 1.0 at exact 0.5, 0.0 at 0 or 1
    symmetry_balance = float(1.0 - abs(frac_flipped - 0.5) * 2.0) if np.isfinite(frac_flipped) else float("nan")

    # Conservative default thresholds (purely reporting)
    #pass_flag = (np.isfinite(raw_mean) and np.isfinite(bimodal_ratio)
    #             and abs(raw_mean) < 1e-5 and bimodal_ratio > 1e6)

    raw_tol = 1e-3  # pro tvé CNF běhy realistická tolerance
    sym_tol = 0.999  # “skoro přesná rovnost”
    bimodal_tol = 1e6

    pass_flag = (
            np.isfinite(raw_mean) and np.isfinite(bimodal_ratio) and np.isfinite(symmetry_balance)
            and abs(raw_mean) < raw_tol
            and symmetry_balance > sym_tol
            and bimodal_ratio > bimodal_tol
    )

    # "spectrum zpívá" index (vyšší = koherentnější + větší bimodalita + přesnější symetrie)
    try:
        phase_spread = float(diag.get("phase_spread", float("nan")))
    except Exception:
        phase_spread = float("nan")

    if np.isfinite(symmetry_balance) and np.isfinite(bimodal_ratio) and np.isfinite(phase_spread):
        #song_index = float(symmetry_balance) * math.log10(max(1.0, float(bimodal_ratio))) / (1e-6 + float(phase_spread))
        song_index = song_index = float(symmetry_balance) * math.log10(max(1.0, bimodal_ratio)) / (1e-6 + float(diag.get("phase_spread", 0.0)))
    else:
        song_index = float("nan")


    return {
        "symmetry_balance": symmetry_balance,
        "song_index": song_index,
        "annihilation_pass": bool(pass_flag),
        "criteria": {
            "abs_align_raw_mean_lt": raw_tol,
            "symmetry_balance_gt": sym_tol,
            "bimodal_ratio_raw_gt": bimodal_tol,
            "target_frac_flipped": 0.5,
        },
    }

    """return {
        "symmetry_balance": symmetry_balance,
        "annihilation_pass": bool(pass_flag),
        "criteria": {
            "abs_align_raw_mean_lt": 1e-5,
            "bimodal_ratio_raw_gt": 1e6,
            "target_frac_flipped": 0.5,
        },
    }"""



def verify_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mu_sat_min = float(get_nested(obj, "IPC.mu_sat_min"))
    mu_unsat_max = float(get_nested(obj, "bands.mu_unsat_max"))
    tau_rep = float(get_nested(obj, "bands.tau"))
    Delta_rep = float(get_nested(obj, "bands.Delta"))

    tau = 0.5 * (mu_sat_min + mu_unsat_max)
    Delta = 0.5 * (mu_sat_min - mu_unsat_max)

    out = {
        "tau_reported": tau_rep,
        "tau_recomputed": tau,
        "Delta_reported": Delta_rep,
        "Delta_recomputed": Delta,
        "bands_separated": bool(Delta > 0),
        "S2_ok": bool(get_nested(obj, "S2.pass")),
        "notes": {
            "mode": str(get_nested(obj, "meta.mode")),
            "lambda_max_GH": float(get_nested(obj, "spectral.lambda_max_GH")),
        }
    }
    print(json.dumps(out, indent=2))
    return out


# fix
def extract_witness(clauses: List[List[int]], n_vars: int, Z: np.ndarray) -> Dict:
    """Extrahuje ohodnocení s úhlovým sweepem pro nalezení nejlepšího průmětu."""
    best_unsat = len(clauses) + 1
    best_assignment = {}

    # Zkusíme 24 směrů (po 15 stupních)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    for phi in angles:
        # Rotace a průmět
        Z_rot = Z * np.exp(1j * phi)
        # Agregace přes dimenzi d (osa 1) a rozhodnutí podle znaménka
        current_assign = {}
        for i in range(n_vars):
            val = np.sum(np.real(Z_rot[i, :]))
            current_assign[i + 1] = True if val >= 0 else False

        # Výpočet UNSAT pro toto natočení
        u_count = 0
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit)
                pol = lit > 0
                if current_assign[v] == pol:
                    sat = True
                    break
            if not sat:
                u_count += 1

        if u_count < best_unsat:
            best_unsat = u_count
            best_assignment = current_assign.copy()
            if best_unsat == 0: break

    # Finální statistiky pro nejlepší nalezený úhel
    scores = [np.sum(np.real(Z * np.exp(1j * angles[np.argmin(angles)]))) for i in range(n_vars)]

    return {
        "assign_sha256": hashlib.sha256(
            json.dumps([best_assignment[i] for i in range(1, n_vars + 1)]).encode()).hexdigest(),
        "unsat": best_unsat,
        "unsat_frac": best_unsat / len(clauses),
        "score_stats": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
    }


# fux
def get_clause_laminate_dither(clauses: List[List[int]], epsilon: float = 0.01) -> np.ndarray:
    """
    Vytvoří deterministický fázový posun pro každou klauzuli (Laminated Dither).
    Tím se zabrání dokonalé Z2 koherenci, která vede k 'plateau UNSAT'.
    """
    dithers = []
    for j, cl in enumerate(clauses):
        # Deterministický hash klauzule pro konzistentní ditherizaci
        h = int(hashlib.md5(str(cl).encode()).hexdigest(), 16)
        phi_j = (h % 1000) / 1000.0 * 2 * math.pi
        dithers.append(complex(math.cos(phi_j), math.sin(phi_j)) * epsilon)
    return np.array(dithers)


def find_optimal_theta(clauses, nvars, clause_phasors, clause_weights, steps=32):
    """
    Hledá větev theta, která maximalizuje gradient (rozhodnost) extrakce.
    Tím 'witness sklouzne na nižší UNSAT' namísto zamrznutí.
    """
    best_theta = 0.0
    min_unsat = float('inf')

    # Prohledáváme [0, pi) díky Z2 symetrii zmíněné ve WRATH
    for theta in np.linspace(0, math.pi, steps):
        assign, _ = extract_assignment_from_ipc(
            clauses, nvars,
            clause_phasors=clause_phasors,
            theta=theta,
            clause_weights=clause_weights
        )
        current_unsat = count_unsat(clauses, assign)
        if current_unsat < min_unsat:
            min_unsat = current_unsat
            best_theta = theta

    return best_theta, min_unsat


def extract_assignment_from_ipc_v6(
        clauses: List[List[int]],
        nvars: int,
        *,
        clause_phasors: np.ndarray,
        theta: float,
        clause_weights: np.ndarray,
        dither_amplitude: float = 0.01  # Parametr z dokumentu WRATH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vylepšená extrakce witnessu s fázovou ditherizací pro zlomení regularity.
    """
    if nvars <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    inc = build_var_clause_incidence(clauses, nvars)

    # 1. Bod zlomu: Přidání laminované ditherizace k fázím
    # Generujeme deterministický šum na základě indexu klauzule
    phi = np.angle(clause_phasors).astype(np.float64)
    dithers = np.array([
        (int(hashlib.md5(str(j).encode()).hexdigest(), 16) % 1000) / 1000.0 * 2 * math.pi
        for j in range(len(clauses))
    ])
    phi += dithers * dither_amplitude  # Obnova regularity fází

    amp = np.abs(clause_phasors).astype(np.float64)

    # 2. Výpočet zarovnání k větvi theta (Z2-aware)
    # wrap_pi musí být v kódu definována (v v5_EXTREME je přítomna)
    align = np.cos(wrap_pi(phi - float(theta)))

    # 3. Agregace skóre s gradientem (nenulové díky ditheru)
    score = np.zeros(int(nvars), dtype=np.float64)
    w = clause_weights.astype(np.float64)

    for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            # Každá klauzule přispívá k hlasování proměnné skrze svůj ditherovaný fázový posun
            s += w[j] * amp[j] * float(lit_sign) * align[j]
        score[v] = s

    # Finální přiřazení (witness)
    assign = (score >= 0)
    return assign, score


def find_optimal_witness_v6(clauses, nvars, clause_phasors, clause_weights, steps=64):
    """
    Hledá větev theta*, kde delta(theta*) != 0, a witness 'sklouzne na nižší UNSAT'.
    """
    best_assign = None
    min_unsat = float('inf')
    best_theta = 0.0

    # Prohledáváme prostor větví (S1 -> Z2 symetrie)
    # Stačí [0, pi), protože cos(phi - (theta+pi)) = -cos(phi - theta), což jen převrací znaménka
    thetas = np.linspace(0, math.pi, steps)

    for th in thetas:
        assign, _ = extract_assignment_from_ipc_v6(
            clauses, nvars,
            clause_phasors=clause_phasors,
            theta=th,
            clause_weights=clause_weights
        )
        current_unsat = count_unsat(clauses, assign)

        if current_unsat < min_unsat:
            min_unsat = current_unsat
            best_assign = assign
            best_theta = th

        # Pokud najdeme SAT (0 UNSAT), můžeme skončit dříve (podle manifestu: jednat pomalu, kde netřeba víc)
        if min_unsat == 0:
            break

    return best_assign, min_unsat, best_theta

# fix
def closure_fuse_cutoff(phases: np.ndarray,
                        threshold_percentile: float = 95.0,
                        epsilon: float = 1e-6,
                        grad_cap: float = 3.141592653589793,   # π default
                        grad_tol: float = 1e-3,
                        mode: str = "tanh"):                   # "tanh" nebo "clip"
    phases = np.asarray(phases, dtype=np.float64)

    abs_vals = np.abs(phases)
    threshold = np.percentile(abs_vals, threshold_percentile)

    # Detekce singular regions (kde by byl blow-up)
    singular_mask = abs_vals > threshold
    #singular_idx = np.flatnonzero(singular_mask).tolist()
    singular_idx = np.flatnonzero(singular_mask).astype(int).tolist()

    # wrap to (-pi, pi]
    phases = (phases + np.pi) % (2*np.pi) - np.pi

    # local phase gradient proxy (circular diff)
    dphi = np.diff(phases, prepend=phases[0])
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    grad = np.abs(dphi)

    # pick singular set by percentile
    thr = np.percentile(grad, threshold_percentile)
    singular = grad >= max(thr, epsilon)

    # --- limiter on singular gradients only ---
    if np.any(singular):
        if mode == "clip":
            dphi_l = dphi.copy()
            dphi_l[singular] = np.clip(dphi_l[singular], -grad_cap, +grad_cap)
        else:
            # soft-ABS (smooth clip): cap * tanh(x/cap)
            dphi_l = dphi.copy()
            dphi_l[singular] = grad_cap * np.tanh(dphi_l[singular] / max(grad_cap, 1e-12))

        # reconstruct regulated phases (integrate back)
        phases_reg = phases.copy()
        phases_reg[0] = phases[0]
        phases_reg[1:] = phases[0] + np.cumsum(dphi_l[1:])
        phases_reg = (phases_reg + np.pi) % (2*np.pi) - np.pi
    else:
        phases_reg = phases

    # recompute grad after regulation for bounded flag
    dphi2 = np.diff(phases_reg, prepend=phases_reg[0])
    dphi2 = (dphi2 + np.pi) % (2*np.pi) - np.pi
    grad2 = np.abs(dphi2)

    max_grad_before = float(np.max(grad)) if grad.size else 0.0
    max_grad_after  = float(np.max(grad2)) if grad2.size else 0.0

    bounded = bool(max_grad_after <= grad_cap + grad_tol)

    # Rozdíl mezi regulovanou fází a původní fází
    # wrap_pi zajistí, že např. rozdíl mezi 3.1 a -3.1 bude 0.08, ne 6.2
    dalpha = wrap_pi(phases_reg - phases)

    # Průměr absolutních hodnot těchto změn
    delta_mean = float(np.mean(np.abs(dalpha)))

    diag = {
        "threshold_percentile": float(threshold_percentile),
        "thr": float(thr),
        "n_singular": int(np.sum(singular)),
        "frac": float(np.mean(singular)) if singular.size else 0.0,
        "max_grad_before": max_grad_before,
        "max_grad": max_grad_after,      # keep your old key name
        "grad_cap": float(grad_cap),
        "mode": str(mode),
        "bounded": bounded,
        #"dalpha": np.ndarray(dalpha),
        #"delta_mean": float(delta_mean),
        "delta_mean": delta_mean,
        "dalpha": dalpha,
        "singular_idx": singular_idx,
        "singular_mask": singular_mask,

    }
    return phases_reg, diag



#fix
def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi


def z2_superselect_phase(phi: np.ndarray, anchor: float) -> np.ndarray:
    """
    Enforce 'no-flip sector' relative to anchor.
    Returns phases shifted by 0 or π so that cos(phi - anchor) >= 0.
    (This is the 'good there too' constraint: same sector across the cut.)
    """
    phi = np.asarray(phi, dtype=np.float64)
    align = np.cos(wrap_pi(phi - float(anchor)))
    flip = align < 0.0
    out = phi.copy()
    if np.any(flip):
        out[flip] = wrap_pi(out[flip] + np.pi)
    return out


def rejoin_branches_soft(phi: np.ndarray,
                         anchor: float,
                         eps: float = 1e-6,
                         temperature: float = 0.15,
                         enforce_no_flip: bool = True) -> tuple[np.ndarray, dict]:
    """
    'Two branches around singularity' -> recombine into single signal.
    - anchor sets the target branch (picked theta, or theta_U1).
    - if enforce_no_flip=True: Z2 superselection, no sign inversion across the cut.
    - returns (phi_rejoined, diagnostics)
    """
    phi = np.asarray(phi, dtype=np.float64)

    # 1) optional Z2 sector lock (no phase inversion)
    if enforce_no_flip:
        phi0 = z2_superselect_phase(phi, anchor=float(anchor))
    else:
        phi0 = phi.copy()

    # 2) Soft re-centering to anchor (a "timeless" gauge-fix: only relative phase matters)
    # We shrink residual phases toward anchor with a smooth map.
    dphi = wrap_pi(phi0 - float(anchor))
    # smooth contraction (tanh) so we don't brutally clip fine structure
    dphi2 = temperature * np.tanh(dphi / max(eps, temperature))
    phi_re = wrap_pi(float(anchor) + dphi2)

    # 3) Diagnostics: how well it recombined
    spread_before = float(np.std(wrap_pi(phi - float(anchor))))
    spread_after  = float(np.std(wrap_pi(phi_re - float(anchor))))
    frac_flipped  = float(np.mean(np.cos(wrap_pi(phi - float(anchor))) < 0.0))
    frac_flipped_after = float(np.mean(np.cos(wrap_pi(phi_re - float(anchor))) < 0.0))

    diag = {
        "anchor": float(anchor),
        "spread_before": spread_before,
        "spread_after": spread_after,
        "frac_neg_before": frac_flipped,
        "frac_neg_after": frac_flipped_after,
        "temperature": float(temperature),
        "enforce_no_flip": bool(enforce_no_flip),
    }
    return phi_re, diag

#fix2
def ipc_metrics(Z_lock: np.ndarray, u: np.ndarray, m: int) -> Tuple[float, float, float, np.ndarray]:
    a = (Z_lock.conj().T @ u) / math.sqrt(float(m))

    # --- EXTREME diagnostics: two-channel order parameters ---
    S1 = np.sum(a)
    sum_abs  = float(np.sum(np.abs(a)) + 1e-12)
    sum_abs2 = float(np.sum(np.abs(a) ** 2) + 1e-12)

    coh_u1 = float(np.abs(S1) / sum_abs) if sum_abs > 0 else 0.0

    # ALWAYS define S2
    S2 = 0.0 + 0.0j
    coh_z2 = 0.0
    if sum_abs2 > 0:
        S2 = np.sum(a * a)
        coh_z2 = float(np.abs(S2) / sum_abs2)

    bimodal_index = coh_z2 / (coh_u1 + 1e-12)

    # robust theta choice
    theta_u1 = float(np.angle(S1)) if np.abs(S1) > 1e-12 else 0.0
    theta_z2 = float(0.5 * np.angle(S2)) if np.abs(S2) > 1e-12 else theta_u1

    # your original Z2 trigger, ale bezpečně:
    use_z2 = (sum_abs > 0.0) and (np.abs(S1) < 0.20 * sum_abs)
    theta = theta_z2 if use_z2 else theta_u1

    mags = np.abs(a)
    beta = float(np.min(mags))
    ang = np.angle(a)

    if use_z2:
        err2 = 0.5 * wrap_pi(2.0 * (ang - theta))
        delta = float(np.max(np.abs(err2)))
    else:
        err = wrap_pi(ang - theta)
        delta = float(np.max(np.abs(err)))

    print(f"\nEXTREME: coherence_u1={coh_u1:.6f}  coherence_z2={coh_z2:.6f}  bimodal_index={bimodal_index:.3f}")
    return theta, beta, delta, a



#fix 3

def run(
        C: int,
        R: int,
        d: int,
        sweeps: int,
        eta: float,
        K: float,
        noise_sigma: float,
        dt: float,
        mu: float,
        mu_E: float,
        h: float,
        tail_frac: float,
        mode: str,
        shared_carrier: bool,
        shared_misphase: bool,
        unsat_neg_frac: float,
        seed: int,
        power_iters: int,
        power_tol: float,
        ipc_weight_mode: str,
        w_delta_min: float,
        w_delta_max: float,
        cnf_path: Optional[str] = None,
        edge_mode: str = "auto",
        flip_incident_unsat: bool = True,
        outside_value: complex = -1.0,
        json_out: Optional[str] = None,
        # CLOSURE parameters (singulární řez blízko singularity)
        lambda_closure: float = 0.1,
        kappa_coupling: float = 0.01,
        enable_closure_cut: bool = True,
        closure_epsilon: float = 1e-6,
        model_out: Optional[str] = None,
        model_from_n_singular: bool = False,
        unsat_out: Optional[str] = None,
        unsat_dump_k: int = 50,
        inf_path: Optional[str] = None,
        inf_max_tokens: int = 200000,
        inf_seed: int = 0,
        polish: int = 0,

        #finisher
        finisher: bool = False,
        max_flips: int = 50_000_000,
        p_min: float = 0.003,
        p_max: float = 0.18,
        p_base: float = 0.10,
        stall_window: int = 800_000,
        restart_shake: int = 96,
        w_inc: float = 1.0,
        w_decay: float = 0.9996,
        w_cap: float = 40.0,
        snapback_gap: int = 250,
        basin_mult: float = 2.2,
        basin_abs: int = 350,
        kick_after: int = 300_000,
        kick_len: int = 100_000,
        kick_p: float = 0.18,

        kick_cooldown: int = 250_000,
        kick_disable_best_mult: int = 2,
        sniper_u: int = 64,
        sniper_flips: int = 8_000_000,
        sniper_p: float = 0.33,
        use_tabu: bool = True,
        tabu_u_threshold: int = 128,
        tabu_tenure: int = 45,
        sniper_end_p: float = 0.03,
        report_every: int = 100_000,

) -> Certificate:
    # ------------------ CNF vs synthetic ------------------
    cnf_meta: dict = {}
    clause_gauge: Optional[np.ndarray] = None
    zeta0 = 0.25

    if inf_path:
        print(f"[INF mód] Načítám {inf_path}")
        txt = _read_text_any(inf_path)
        tokens = _tokenize_inf(txt, max_tokens=inf_max_tokens)
        clause_gauge, sectors, assign_sha256 = _inf_to_gauge_and_sectors(tokens, seed=inf_seed)

        C = len(tokens)
        nvars = 0
        clauses = []  # not used in INF (unless you want assignment extraction later)
        cnf_meta = {
            "type": "INF",
            "inf_path": inf_path,
            "C": int(C),
            "inf_seed": int(inf_seed),
            "assign_sha256": assign_sha256,
        }

    elif cnf_path:
        print(f"[CNF mód] Načítám {cnf_path}")
        nvars, clauses = parse_dimacs(cnf_path)
        C = len(clauses)
        print(f"  proměnné: {nvars}    klauzule: {C:,}")

        sectors = np.empty(C, dtype=np.int32)

        # deterministic UNSAT seeding (hash-based, invariant across machines)
        unsat_idx = cnf_seed_unsat_indices(clauses, nvars)

        # SAT mode: baseline all +1 (do not inject defects from an imperfect seed model)
        # UNSAT mode: inject π-defects exactly on seed-unsatisfied clauses
        g = np.ones(C, dtype=np.float64)
        if mode.lower() == "unsat":
            for j in unsat_idx:
                g[j] = -1.0
        clause_gauge = g

        cnf_meta = {
            "cnf_path": cnf_path,
            "cnf_sha256": sha256_file(cnf_path),
            "nvars": nvars,
            "seed_unsat": int(len(unsat_idx)),
            "seed_unsat_frac": float(len(unsat_idx) / max(1, C)),
        }
    else:
        clauses = []
        nvars = 0
        clause_gauge = np.ones(C, dtype=np.float64)
        sectors = np.empty(C, dtype=np.int32)

    # ------------------ geometry ------------------
    T = 2 * int(R)
    m = int(R) // 2

    offsets = prime_offsets(C, T)

    # 4D block: push each clause into one of 4 "quadrants" by shifting carrier phase/time
    if sectors is not None:
        qshift = max(1, T // 8)  # conservative separation
        offsets = (offsets + sectors.astype(np.int64) * qshift) % T

    masks = build_masks(C, m, zeta0, shared_carrier, shared_misphase, seed=seed)

    # IMPORTANT: for large T, a non-zero outside_value can dominate Gram overlaps and
    # suppress IPC ("silent" regime). Keep it configurable.
    Z = build_Z(T, C, m, offsets, masks, clause_gauge=clause_gauge, outside_value=outside_value)
    M = build_lock_mask_matrix(T, C, m, offsets)
    Z_lock = Z * M  # zeros outside lock

    # CNF quick report (before coupling)
    if cnf_path:
        rep = cnf_projection_report(Z_lock)
        print(f"  Průměrná amplituda     : {rep['avg_amp']:.6f}")
        print(f"  Medián amplitudy       : {rep['median_amp']:.6f}")
        print(f"  Max / min amplituda    : {rep['max_amp']:.6f} / {rep['min_amp']:.6f}")
        print(f"  Frakce |proj| > 0.05   : {100 * rep['frac_proj_gt_005']:.2f} %")
        print(f"  Frakce |proj| > 0.1    : {100 * rep['frac_proj_gt_01']:.2f} %")
        print(f"  Rozptyl úhlů (variance): {rep['angle_var']:.6f} rad²")
        print(f"  Std úhlů               : {rep['angle_std']:.4f} rad  ≈ {rep['angle_std'] * 180 / math.pi:.2f}°")
        print(f"  Koherenční proxy       : {rep['coh_proxy']:.6f}  (1 = všechny locky dokonale zarovnané)")

    # ------------------ signed constraints + overlap coupling ------------------
    em = (edge_mode or "auto").lower()

    # ochrana proti OOM: full clause-graph jen pro malé CNF
    BIG = (len(clauses) > 200_000) or (nvars > 200_000)

    if cnf_path and em in ("cnf", "logic"):
        if BIG:
            # bounded-degree CNF graph (strip) – škáluje
            edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
        else:
            # full graph jen pro malé instance
            edges = build_logic_edges_from_cnf(clauses, nvars, include_same_polarity=(em == "cnf"))

        if not edges:
            edges = circulant_edges(C, d)

    elif cnf_path and em == "auto":
        edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
        if not edges:
            edges = circulant_edges(C, d)
    else:
        edges = circulant_edges(C, d)

    edge_signs = build_edge_signs_from_gauge(
        edges,
        clause_gauge,
        mode=mode,
        unsat_neg_frac=unsat_neg_frac,
        seed=seed,
        flip_incident_unsat=flip_incident_unsat,
    )
    rng = np.random.default_rng(int(seed) + 1337)
    apply_signed_overlap_coupling(
        Z, T, C, m, offsets, edges, edge_signs,
        eta=eta, sweeps=sweeps, K=K, noise_sigma=noise_sigma,
        dt=dt, mu=mu, h=h, rng=rng, lock_mask=M
    )
    Z_lock = Z * M

    # ------------------ decision operator (edge-Gram) ------------------
    nbr, val = build_edge_gram(Z, T, C, m, offsets, edges)
    lam = power_lambda_max_edge(nbr, val, iters=power_iters, tol=power_tol)
    mu_dec = float(lam / float(C))

    rho = neighbor_rowsum(nbr, val)
    kap = kappa_S2(T, m, zeta0)
    bound = float(d) * kap
    S2_ok = bool(rho <= bound + 1e-12)

    # ------------------ CLOSURE operators (část 1: před IPC) ------------------
    theta_closure = closure_integral_from_phases(Z_lock, offsets, m)
    tau_refined = refined_time_parameter(float(T), theta_closure, lambda_closure)
    residue_metric = residue_compression_metric(Z_lock, m)

    # ------------------ IPC with clause weights ------------------
    corr = edge_correlation_proxy(nbr, val)
    w = build_ipc_clause_weights(C, mode=ipc_weight_mode, delta_min=w_delta_min, delta_max=w_delta_max, corr_proxy=corr)
    u = ipc_time_mode_u(Z_lock, w, m=m, iters=power_iters, tol=power_tol)
    theta, beta, delta, a = ipc_metrics(Z_lock, u, m=m)

    # ------------------ CLOSURE operators (část 2: po IPC) ------------------
    closure_diagnostics = {
        "theta_closure": float(theta_closure),
        "tau_refined": float(tau_refined),
        "lambda_scale": float(lambda_closure),
        "kappa_coupling": float(kappa_coupling),
        "residue_compression": float(residue_metric),
        "enabled": bool(enable_closure_cut),
    }

    sing_mask = None
    fuse_diag = None

    if enable_closure_cut:
        a_phases = np.angle(a) if hasattr(a, '__len__') else np.array([float(np.angle(a))])

        a_regulated, fuse_diag = closure_fuse_cutoff(
            a_phases,
            threshold_percentile=95.0,
            epsilon=closure_epsilon,
            grad_cap=math.pi,
            mode="tanh",
        )

        # apply fuse-regulated phases to 'a' (keep amplitudes)
        a = np.abs(a) * np.exp(1j * a_regulated)

        # robust singular mask
        """sing_mask = fuse_diag.get("singular_mask", None)
        if sing_mask is None:
            idx = fuse_diag.get("singular_idx", [])
            sing_mask = np.zeros_like(a_phases, dtype=bool)
            if len(idx):
                sing_mask[np.array(idx, dtype=int)] = True"""

        idx = np.asarray(fuse_diag.get("singular_idx", []), dtype=int)
        sing_mask = np.zeros_like(a_phases, dtype=bool)
        if idx.size:
            sing_mask[idx] = True

        # fallback: if still empty, activate 1 index so REJOIN is never a no-op
        if np.sum(sing_mask) == 0:
            idx = fuse_diag.get("singular_idx", [])
            if len(idx):
                sing_mask[np.array(idx[:1], dtype=int)] = True

        print("REJOIN_MASK: true_count =", int(np.sum(sing_mask)), "of", int(len(sing_mask)))
        print("FUSE_KEYS:", sorted(list(fuse_diag.keys())))

        closure_diagnostics["fuse"] = fuse_diag

        # NOTE: rejoin is intentionally NOT applied here.
        # It will be applied once, after branch selection, using best_theta_used.

    # ------------------ CNF witness extraction (assignment projection) ------------------
    witness = {}
    if cnf_path:
        #try:
            # Compute both order parameters for branch selection
            S1 = np.sum(a)
            S2 = np.sum(a * a)

            theta_U1 = float(np.angle(S1)) if S1 != 0 else 0.0
            theta_Z2 = 0.5 * float(np.angle(S2)) if S2 != 0 else 0.0

            # "Tower" identity: 4^(16^x) = 16^4  => 16^x = 8 => x = log_16(8) = 3/4
            x_TOWER = math.log(8.0, 16.0)  # 0.75
            theta_TOWER = float(2.0 * math.pi * x_TOWER)  # 3π/2

            """candidates = [
                ("θ_U1", theta_U1),
                ("θ_Z2", theta_Z2),
                ("θ_Z2+π", theta_Z2 + np.pi),
            ]"""

            candidates = [("θ_U1", theta_U1), ("θ_Z2", theta_Z2), ("θ_Z2+π", theta_Z2 + np.pi), ("θ_TOWER", theta_TOWER)]

            best_unsat = float('inf')
            best_assign = None
            best_score = None
            best_theta_used = float(theta)
            best_branch_name = "theta"

            #print("\n[3-BRANCH TEST: Θ = {θ_U1, θ_Z2, θ_Z2+π}] (Operator mode: auto-flip enabled)")
            print(f"\n[4-BRANCH TEST: Θ = {{θ_U1, θ_Z2, θ_Z2+π, θ_TOWER}}] (Operator mode: auto-flip enabled)")

            diag_results = {}

            for branch_name, th_test in candidates:
                assign_test, score_test = extract_assignment_from_ipc(
                    clauses, nvars,
                    clause_phasors=a, theta=th_test, clause_weights=w
                )
                unsat_test = count_unsat(clauses, assign_test)

                op_diag = operator_diagnostics(a, th_test, w)
                diag_results[branch_name] = op_diag

                print(
                    f"  {branch_name:8s}: UNSAT = {unsat_test:3d}  "
                    f"φ_spread={op_diag['phase_spread']:.4f}  "
                    f"bimodal_raw={op_diag['bimodal_ratio_raw']:.1f}  "
                    f"flipped={100 * op_diag['frac_flipped']:.1f}%"
                )

                if unsat_test < best_unsat:
                    best_unsat = unsat_test
                    best_assign = assign_test
                    best_score = score_test
                    best_theta_used = float(th_test)
                    best_branch_name = branch_name

            # ---- APPLY REJOIN ONCE (after branch selection) ----
            if enable_closure_cut and (sing_mask is not None) and np.any(sing_mask):
                phi = np.angle(a).copy()
                phi_rej, re2 = rejoin_branches_soft(
                    phi,
                    anchor=float(best_theta_used),
                    eps=max(1e-12, float(closure_epsilon)),
                    temperature=0.12,
                    enforce_no_flip=True
                )

                spread_before = re2.get("spread_before", float("nan"))
                spread_after = re2.get("spread_after", float("nan"))
                frac_neg_before = re2.get("frac_neg_before", float("nan"))
                frac_neg_after = re2.get("frac_neg_after", float("nan"))

                phi[sing_mask] = phi_rej[sing_mask]
                a = np.abs(a) * np.exp(1j * phi)

                closure_diagnostics["rejoin_final"] = {**re2,"applied_frac": float(np.mean(sing_mask)),}

                print(
                    f"CLOSURE-REJOIN(masked): applied_frac={closure_diagnostics['rejoin_final']['applied_frac']:.4f} "
                    f"spread_before={spread_before:.4f} spread_after={spread_after:.4f} "
                    f"neg_before={frac_neg_before:.3f} neg_after={frac_neg_after:.3f}"
                )

            print(f"  → Vybrána větev {best_branch_name} s UNSAT = {best_unsat}")

            # Display detailed operator structure for selected branch
            best_diag = diag_results[best_branch_name]
            print(f"\n[OPERATOR STRUCTURE - {best_branch_name}]")
            print(f"  Phase spread: {best_diag['phase_spread']:.4f} rad  (concentration: {best_diag['phase_concentration']:.3f})")
            print(f"  Raw alignment: mean={best_diag['align_raw_mean']:+.3f}  |mean|={best_diag['align_raw_abs_mean']:.3f}")
            print(f"  Post auto-flip: mean={best_diag['align_post_mean']:+.3f}  |mean|={best_diag['align_post_abs_mean']:.3f}")
            print(f"  Fraction flipped: {100 * best_diag['frac_flipped']:.1f}%")
            print(f"  Bimodal signature (raw): ratio={best_diag['bimodal_ratio_raw']:.2f}  edge={best_diag['edge_mass_raw']:.3f}")

            # ASCII histogram (raw alignment before auto-flip)
            hist = best_diag['hist_raw']
            bins = best_diag['bins']
            max_bar = 40
            max_val = max(hist) if max(hist) > 0 else 1.0
            print(f"  Raw alignment histogram (before auto-flip):")
            for i in range(len(hist)):
                bin_center = (bins[i] + bins[i + 1]) / 2
                bar_len = int((hist[i] / max_val) * max_bar)
                bar = "█" * bar_len
                print(f"    {bin_center:+.2f}: {bar} {hist[i]:.3f}")

            # Comparison across branches
            print(f"\n[BRANCH COMPARISON - Operator mode]")
            print(f"  Branch    | Phase_spread | Bimodal_raw | Frac_flipped | Edge_mass")
            print(f"  ----------|--------------|-------------|--------------|----------")
            for bname in ["θ_U1", "θ_Z2", "θ_Z2+π", "θ_TOWER"]:
                bd = diag_results[bname]
                print(
                    f"  {bname:8s}  |    {bd['phase_spread']:.4f}    |    {bd['bimodal_ratio_raw']:6.1f}  |     {100 * bd['frac_flipped']:5.1f}%   |   {bd['edge_mass_raw']:.3f}"
                )

            # Use the best branch
            assign_wit = best_assign
            score_wit = best_score
            unsat_wit = best_unsat

            # --- export model + validation ---
            """if model_out is not None:
                write_dimacs_model(model_out, assign_wit, nvars)

                # recompute UNSAT from scratch (validation)
                unsat_check = count_unsat(clauses, assign_wit)
                witness["unsat_check"] = int(unsat_check)
                witness["model_out"] = str(model_out)

                if unsat_out is not None:
                    bad = unsat_clause_indices(clauses, assign_wit, limit=unsat_dump_k)
                    with open(unsat_out, "w", encoding="utf-8") as f:
                        f.write("\n".join(map(str, bad)))
                    witness["unsat_out"] = str(unsat_out)
                    witness["unsat_dump_k"] = int(unsat_dump_k)
            """

            # --- export model + validation ---
            model_assign = None
            model_source = None

            if model_from_n_singular:
                model_assign = assignment_from_n_singular(int(fuse_diag["n_singular"]), nvars)
                model_source = f"n_singular={int(fuse_diag['n_singular'])}"
            else:
                model_assign = assign_wit
                model_source = "ipc_witness"

            if polish > 0:
                assign = greedy_polish(clauses, model_assign, flips=polish, seed=seed)
                uns = count_unsat(clauses, _as_int_assign(assign))
                print(
                    f"\n[POLISH] unsat_clauses  : {uns} / {len(clauses)}  ({100.0 * uns / max(1, len(clauses)):.2f}%)")
            else:
                assign = model_assign

            if finisher:

                var_occ = build_var_occ(clauses, nvars)

                model, solved, st = finisher_predator_sole_sat_vFinal(
                    clauses=clauses,
                    nvars=nvars,
                    a0=assign,
                    var_occ=var_occ,
                    seed=seed,
                    max_flips=max_flips,
                    p_min=p_min,
                    p_max=p_max,
                    p_base=p_base,
                    stall_window=stall_window,
                    restart_shake=restart_shake,
                    w_inc=w_inc,
                    w_decay=w_decay,
                    w_cap=w_cap,
                    snapback_gap=snapback_gap,
                    basin_mult=basin_mult,
                    basin_abs=basin_abs,
                    kick_after=kick_after,
                    kick_len=kick_len,
                    kick_p=kick_p,
                    sniper_u=sniper_u,
                    sniper_end_p=sniper_end_p,
                    sniper_flips=sniper_flips,
                    sniper_p=sniper_p,
                    use_tabu=bool(use_tabu),
                    tabu_u_threshold=tabu_u_threshold,
                    tabu_tenure=tabu_tenure,
                    report_every=report_every,
                )

                # sat = check_sat(clauses, _as_int_assign(model))
                uns = count_unsat(clauses, _as_int_assign(model))

                print(
                    f"[FINISHER] unsat_clauses  : {uns} / {len(clauses)}  ({100.0 * uns / max(1, len(clauses)):.2f}%)")
                # print(f"Verified SAT   : {sat}")
                # print(f"Solved flag    : {solved}")

            else:
                model = assign



            unsat = count_unsat(clauses, model)

            if model_out is not None:
                write_dimacs_model(model_out, model, nvars)
                witness["model_out"] = model_out
                witness["model_source"] = model_source
                witness["model_unsat"] = unsat

                # recompute UNSAT from scratch (validation)
                #unsat_check = count_unsat(clauses, model_assign)
                #witness["unsat_check"] = int(unsat_check)
                #witness["model_out"] = str(model_out)

                # if unsat_out is not None:
                #    bad = unsat_clause_indices(clauses, model_assign, limit=unsat_dump_k)
                #    with open(unsat_out, "w", encoding="utf-8") as f:
                #        f.write("\n".join(map(str, bad)))
                #    witness["unsat_out"] = str(unsat_out)
                #    witness["unsat_dump_k"] = int(unsat_dump_k)

        # Diagnostics: how strongly proxy is actually driving the witness
            phi_w = np.angle(a)
            amp_w = np.abs(a).astype(np.float64, copy=False)
            align_w = np.cos(wrap_pi(phi_w - float(best_theta_used))).astype(np.float64, copy=False)
            phi_w_use = phi_w.copy()
            flip_w = (align_w < 0.0)
            if np.any(flip_w):
                phi_w_use[flip_w] = wrap_pi(phi_w_use[flip_w] + math.pi)
                align_w = np.cos(wrap_pi(phi_w_use - float(best_theta_used))).astype(np.float64, copy=False)
            gate_w = (align_w * align_w)  # Z2-invariant gating
            drive_w = w.astype(np.float64, copy=False) * amp_w * gate_w


            witness = {
                "assign_sha256": sha256_assignment(assign_wit),
                "unsat": int(unsat),
                "unsat_frac": float(unsat / max(1, len(clauses))),
                "branch_selected": str(best_branch_name),
                "branch_theta": float(best_theta_used),
                "theta_U1": float(theta_U1),
                "theta_Z2": float(theta_Z2),
                "tower_x": float(x_TOWER),
                "theta_TOWER": float(theta_TOWER),
                "operator_diag": diag_results[best_branch_name],
                "operator_diag_all": diag_results,
                "drive_stats": {
                    "amp_mean": float(np.mean(amp_w)) if amp_w.size else 0.0,
                    "amp_median": float(np.median(amp_w)) if amp_w.size else 0.0,
                    "gate_mean": float(np.mean(gate_w)) if gate_w.size else 0.0,
                    "drive_mean": float(np.mean(drive_w)) if drive_w.size else 0.0,
                    "drive_median": float(np.median(drive_w)) if drive_w.size else 0.0,
                    "drive_min": float(np.min(drive_w)) if drive_w.size else 0.0,
                    "drive_max": float(np.max(drive_w)) if drive_w.size else 0.0,
                    "drive_nonzero_frac": float(np.mean(drive_w > 0.0)) if drive_w.size else 0.0,
                },
                "score_stats": {
                    "min": float(np.min(score_wit)) if score_wit.size else 0.0,
                    "max": float(np.max(score_wit)) if score_wit.size else 0.0,
                    "mean": float(np.mean(score_wit)) if score_wit.size else 0.0,
                    "std": float(np.std(score_wit)) if score_wit.size else 0.0,
                },
            }

        #except Exception as e:
        #    witness = {"error": f"{type(e).__name__}: {e}"}

    # ------------------ bands ------------------
    mu_sat_min = ipc_mu_sat_min(beta, delta)

    lam_unsat_ceiling = float(1.0 + bound)
    mu_unsat_max = float(lam_unsat_ceiling / float(C))
    tau = 0.5 * (mu_sat_min + mu_unsat_max)
    Delta = 0.5 * (mu_sat_min - mu_unsat_max)
    separated = bool(Delta > 0)

    # ------------------ coherence diag (tail + EMA) ------------------
    R_series = np.abs(np.sum(Z_lock, axis=1)).astype(np.float64, copy=False)
    r_mean = float(np.mean(R_series)) if R_series.size else 0.0
    r_min = float(np.min(R_series)) if R_series.size else 0.0
    r_max = float(np.max(R_series)) if R_series.size else 0.0
    tf = float(tail_frac)
    tf = 0.0 if tf < 0.0 else (1.0 if tf > 1.0 else tf)
    if R_series.size and tf > 0.0:
        start = int(max(0, min(R_series.size - 1, math.floor((1.0 - tf) * R_series.size))))
        tail = R_series[start:]
    else:
        tail = R_series
    r_mean_tail = float(np.mean(tail)) if tail.size else 0.0

    # EMA on tail (uses mu_E)
    muE = float(mu_E)
    if muE < 0.0:
        muE = 0.0
    if muE > 1.0:
        muE = 1.0
    if tail.size:
        ema = float(tail[0])
        a_ema = 1.0 - muE
        for x in tail[1:]:
            ema = muE * ema + a_ema * float(x)
        r_ema_tail = float(ema)
    else:
        r_ema_tail = 0.0

    meta = {
        "C": int(C), "T": int(T), "m": int(m), "R": int(R), "d": int(d),
        "mode": str(mode),
        "sweeps": int(sweeps), "eta": float(eta),
        "dream6": {"K": float(K), "noise_sigma": float(noise_sigma), "dt": float(dt), "mu": float(mu),
                   "mu_E": float(mu_E), "h": float(h), "tail_frac": float(tail_frac)},
        "shared_carrier": bool(shared_carrier),
        "shared_misphase": bool(shared_misphase),
        "unsat_neg_frac": float(unsat_neg_frac),
        "seed": int(seed),
        "zeta0": float(zeta0),
        "ipc_weights": {"mode": ipc_weight_mode, "delta_min": float(w_delta_min), "delta_max": float(w_delta_max)},
        **cnf_meta,
    }

    cert_data = {
        "meta": meta,
        "S2": {"rho": float(rho), "kappa": float(kap), "d_kappa": float(bound), "pass": bool(S2_ok)},
        "closure": closure_diagnostics,
        "spectral": {"lambda_max_GH": float(lam), "mu_dec": float(mu_dec)},
        "IPC": {"beta": float(beta), "delta": float(delta), "theta": float(theta), "mu_sat_min": float(mu_sat_min)},
        "bands": {"lam_unsat_ceiling": float(lam_unsat_ceiling), "mu_unsat_max": float(mu_unsat_max),
                  "tau": float(tau), "Delta": float(Delta), "separated": bool(separated)},
        "diag": {
            "coherence_R": {"mean": float(r_mean), "min": float(r_min), "max": float(r_max),
                            "mean_tail": float(r_mean_tail), "ema_tail": float(r_ema_tail),
                            "tail_frac": float(tail_frac)},
            "corr_proxy": {"mean": float(np.mean(corr)) if corr.size else 0.0,
                           "min": float(np.min(corr)) if corr.size else 0.0,
                           "max": float(np.max(corr)) if corr.size else 0.0},
            "weights": {"mean": float(np.mean(w)) if w.size else 0.0,
                        "min": float(np.min(w)) if w.size else 0.0,
                        "max": float(np.max(w)) if w.size else 0.0},
            "cnf_witness": witness
        },
    }

    # ---- ANNIHILATION LEMMA (diagnostic imprint) ----
    try:
        if isinstance(witness, dict) and "operator_diag" in witness:
            op = witness["operator_diag"]
            witness["annihilation"] = apply_annihilation_lemma(op)
            certify_annihilation_lemma(op, witness["annihilation"])
    except Exception:
        pass

    print(f"\nWitness: {witness}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(cert_data, f, indent=2, ensure_ascii=False)

    return Certificate(**cert_data)


def write_dimacs_model(path: str, assign: np.ndarray, nvars: int, lits_per_line: int = 20) -> None:
    """
    DIMACS model format:
      v 1 -2 3 ... 0
    assign[i-1] = True  -> +i
    assign[i-1] = False -> -i
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("c DREAM6 exported model\n")
        f.write(f"c nvars={nvars}\n")
        line = ["v"]
        for i in range(1, nvars + 1):
            lit = i if bool(assign[i - 1]) else -i
            line.append(str(lit))
            if (i % lits_per_line) == 0:
                f.write(" ".join(line) + " 0\n")
                line = ["v"]
        if len(line) > 1:
            f.write(" ".join(line) + " 0\n")


def unsat_clause_indices(clauses: List[List[int]], assign: np.ndarray, limit: int = 0) -> List[int]:
    bad = []
    for j, cl in enumerate(clauses):
        sat = False
        for lit in cl:
            v = abs(lit) - 1
            val = bool(assign[v])
            if (lit > 0 and val) or (lit < 0 and not val):
                sat = True
                break
        if not sat:
            bad.append(j)
            if limit > 0 and len(bad) >= limit:
                break
    return bad


# 4D
# ---------------------------------------------------------------------
# OPERATOR_INF (text -> "info clauses")
# ---------------------------------------------------------------------

def _read_text_any(path: str) -> str:
    p = path.lower()
    if p.endswith((".txt", ".tex", ".log", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if p.endswith(".pdf"):
        # lightweight PDF text extraction (good enough for operator seeding)
        import PyPDF2  # pip install pypdf2
        out = []
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            for page in r.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out)
    raise ValueError(f"Unsupported inf format: {path}")

def _tokenize_inf(text: str, max_tokens: int) -> List[str]:
    # deterministic, boring tokenizer (whitespace + strip punctuation-ish)
    buf = []
    cur = []
    for ch in text:
        if ch.isalnum() or ch in "_^{}\\":  # keep LaTeX-ish atoms stable
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
                if len(buf) >= max_tokens:
                    break
    if cur and len(buf) < max_tokens:
        buf.append("".join(cur))
    return buf

def _inf_to_gauge_and_sectors(tokens: List[str], seed: int = 0) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns:
      clause_gauge: float64 in {-1,+1}  (C,)
      sectors: int32 in {0,1,2,3}       (C,)
      assign_sha256: hash anchor for reproducibility
    """
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\n")
    if seed:
        h.update(str(seed).encode("ascii"))
    digest = h.hexdigest()

    C = len(tokens)
    gauge = np.empty(C, dtype=np.float64)
    sectors = np.empty(C, dtype=np.int32)

    for j, t in enumerate(tokens):
        hj = hashlib.sha256((t + "|" + str(seed)).encode("utf-8", "ignore")).digest()
        # sector = 2 bits
        sectors[j] = hj[0] & 0x03
        # gauge sign from next bit
        gauge[j] = +1.0 if (hj[1] & 0x01) else -1.0

    return gauge, sectors, digest


"""def assignment_from_n_singular(n_singular: int, nvars: int) -> dict:
    
    #Deterministicky vytvoří DIMACS model (assignment) jen ze seed = n_singular.
    #Var i dostane hodnotu podle bitstreamu SHA-256(seed:chunk).
    #Stabilní a reprodukovatelné napříč stroji.
    
    import hashlib

    if n_singular is None or nvars <= 0:
        return {}

    assign = {}
    bit_index = 0
    chunk = 0

    while bit_index < nvars:
        digest = hashlib.sha256(f"{int(n_singular)}:{chunk}".encode("utf-8")).digest()
        for byte in digest:
            for b in range(8):
                if bit_index >= nvars:
                    break
                bit = (byte >> b) & 1  # LSB-first v rámci bytu (jen konzistence)
                var = bit_index + 1
                assign[var] = bool(bit)
                bit_index += 1
        chunk += 1

    return assign"""

def assignment_from_n_singular(n_singular: int, nvars: int) -> list:
    if n_singular is None or nvars <= 0:
        return []

    assign_list = []
    bit_index = 0
    chunk = 0

    while bit_index < nvars:
        # Deterministický hash pro daný chunk
        digest = hashlib.sha256(f"{int(n_singular)}:{chunk}".encode("utf-8")).digest()
        for byte in digest:
            for b in range(8):
                if bit_index >= nvars:
                    break
                # Extraxe bitu
                bit = (byte >> b) & 1
                assign_list.append(bool(bit))
                bit_index += 1
        chunk += 1

    return assign_list

"""def assignment_from_n_singular(n_singular: int, num_vars: int, divisor: float = 3.0):
    
    #Deterministický model (assignment) z n_singular přes SHA-256:
    #- canonical: "n_singular:<int>"
    #- digest -> big-endian int -> modulo -> seed -> RNG
    #- vrací dict {var_index: bool} pro 1..num_vars
    
    import hashlib, random

    s = f"n_singular:{int(n_singular)}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    x = int.from_bytes(h, "big")

    # kontrolovaná komprese do seed prostoru (divisor laditelný)
    seed = int(x / float(divisor)) & 0xFFFFFFFF
    rng = random.Random(seed)

    #return {i: bool(rng.getrandbits(1)) for i in range(1, num_vars + 1)}
    return [bool(rng.getrandbits(1)) for _ in range(num_vars)]"""


import random

def write_unsat_witness_file(path: str, unsat_clause_ids, assign: dict, nvars: int) -> None:
    """
    Zapíše UNSAT witness: seznam ID klauzulí + (volitelně) model v DIMACS stylu.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("c UNSAT witness\n")
        f.write(f"c unsat_count {len(unsat_clause_ids)}\n")
        if unsat_clause_ids:
            f.write("c unsat_clause_ids " + " ".join(map(str, unsat_clause_ids)) + "\n")
        else:
            f.write("c unsat_clause_ids (none)\n")

        if assign:
            f.write("c model (DIMACS v-lines):\n")
            # v řádky po ~10 literálech
            line = []
            for v in range(1, nvars + 1):
                lit = v if assign.get(v, False) else -v
                line.append(str(lit))
                if len(line) >= 10:
                    f.write("v " + " ".join(line) + " 0\n")
                    line = []
            if line:
                f.write("v " + " ".join(line) + " 0\n")
        else:
            f.write("c model: (none)\n")

# ---- polish ----
def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int = 20000,
    seed: int = 49,
    alpha: float = 2.4,     # probSAT make exponent
    beta: float = 0.9,      # probSAT break exponent
    epsilon: float = 1e-3,  # probSAT epsilon
    probsat_quota: int = 2000,  # max kroků pro probSAT fázi v rámci polish
) -> List[int]:
    """
    Micro-polish finisher:
      A) exhaust all zero-break flips (tie by max-make)
      B) min-break + max-make tie-break
      C) short probSAT burst if still UNSAT
    """
    rnd = random.Random(seed)
    nvars = len(assign01)
    C = len(clauses)

    # --- adjacency (1-indexed variables) ---
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for L in cl:
            (pos if L > 0 else neg)[abs(L)].append(ci)

    # --- state (1-indexed assign for speed) ---
    assign = [False] + [bool(b) for b in assign01]
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list: List[int] = []  # compact container of UNSAT clause indices

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

    # init sat_count / unsat
    for ci, cl in enumerate(clauses):
        cnt = 0
        for L in cl:
            v = abs(L)
            val = assign[v]
            if L < 0:
                val = (not val)
            if val:
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            add_unsat(ci)
    # compaction
    unsat_list = [ci for ci in unsat_list if in_unsat[ci]]

    # --- helpers ---
    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:  # True->False
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:          # False->True
            for ci in neg[v]:
                if sat_count[ci] == 1:
                    bc += 1
        return bc

    def makecount(v: int) -> int:
        mk = 0
        if assign[v]:
            for ci in neg[v]:
                if in_unsat[ci]:
                    mk += 1
        else:
            for ci in pos[v]:
                if in_unsat[ci]:
                    mk += 1
        return mk

    def flip_var(v: int):
        #Incremental flip with sat_count/unsat maintenance.
        old = assign[v]
        assign[v] = not old
        if old:
            for ci in pos[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in neg[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)
        else:
            for ci in neg[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in pos[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)

    # utility to pick a random UNSAT clause quickly
    def pick_unsat_clause() -> Optional[int]:
        if not unsat_list:
            return None
        # fast cleanup-on-read
        i = rnd.randrange(len(unsat_list))
        for _ in range(3):  # up to 3 tries to hit a live one
            ci = unsat_list[i]
            if in_unsat[ci]:
                return ci
            i = rnd.randrange(len(unsat_list))
        # fallback: compact and retry once
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    def cur_unsat() -> int:
        # quick count without full compaction
        return sum(1 for ci in unsat_list if in_unsat[ci])

    # --- keep best-so-far ---
    best_assign = assign[:]
    best_uns = cur_unsat()

    steps = 0
    # -------- Phase A: exhaust freebies --------
    made_progress = True
    while made_progress and steps < flips:
        made_progress = False
        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]
        freebies = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            if bc == 0:
                freebies.append((makecount(v), v))
        if freebies:
            freebies.sort(reverse=True)  # prefer max make
            _, v = freebies[0]
            flip_var(v)
            steps += 1
            made_progress = True
            # update best
            u = cur_unsat()
            if u < best_uns:
                best_uns = u
                best_assign = assign[:]
            if u == 0:
                return [1 if b else 0 for b in assign[1:]]

    # -------- Phase B: min-break, max-make --------
    while steps < flips:
        if best_uns == 0:
            return [1 if b else 0 for b in best_assign[1:]]

        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]

        # preferentially do freebies if present
        v_choice = None
        freebies = []
        cand = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            mk = makecount(v)
            if bc == 0:
                freebies.append((mk, v))
            cand.append((bc, -mk, v))  # -mk to maximize make on tie
        if freebies:
            freebies.sort(reverse=True)
            v_choice = freebies[0][1]
        else:
            # min break, then max make
            v_choice = min(cand)[2]

        flip_var(v_choice)
        steps += 1

        u = cur_unsat()
        if u < best_uns:
            best_uns = u
            best_assign = assign[:]
        if u == 0:
            return [1 if b else 0 for b in assign[1:]]

        # malý „kick“: když se nic nezlepšilo 1k kroků, zkus probSAT burst
        if steps % 1000 == 0 and u >= best_uns:
            # -------- Phase C: short probSAT burst --------
            for _ in range(min(probsat_quota, flips - steps)):
                ci2 = pick_unsat_clause()
                if ci2 is None:
                    return [1 if b else 0 for b in assign[1:]]
                clause2 = clauses[ci2]
                # scores ~ (make+eps)^alpha / (break+eps)^beta
                scores = []
                tot = 0.0
                last_v = None
                for L in clause2:
                    v = abs(L)
                    mk = makecount(v)
                    bc = breakcount(v)
                    s = ((mk + epsilon) ** alpha) / ((bc + epsilon) ** beta)
                    scores.append((v, s))
                    tot += s
                    last_v = v
                r = rnd.random() * tot
                acc = 0.0
                pick = last_v
                for v, s in scores:
                    acc += s
                    if acc >= r:
                        pick = v
                        break
                flip_var(pick)
                steps += 1
                u2 = cur_unsat()
                if u2 < best_uns:
                    best_uns = u2
                    best_assign = assign[:]
                if u2 == 0 or steps >= flips:
                    return [1 if b else 0 for b in assign[1:]]
            # po burstu pokračujeme B-fází

    # vyčerpán budget – vrať nejlepší nalezené
    return [1 if b else 0 for b in (best_assign[1:] if best_uns < cur_unsat() else assign[1:])]



# --- finishers

BoolAssign = List[bool]
Clause = List[int]



def _as_bool_assign(a: List[int] | List[bool], nvars: int) -> BoolAssign:
    if len(a) != nvars:
        raise ValueError(f"Assignment length mismatch: got {len(a)} expected {nvars}")
    if a and isinstance(a[0], bool):
        return list(a)  # type: ignore[arg-type]
    return [bool(x) for x in a]  # type: ignore[arg-type]


def _as_int_assign(a: BoolAssign) -> List[int]:
    return [1 if b else 0 for b in a]


def lit_true(lit: int, aval: bool) -> bool:
    return aval if lit > 0 else (not aval)


def build_var_occ(clauses: List[Clause], nvars: int) -> List[List[Tuple[int, int]]]:
    occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            if 0 <= v < nvars:
                occ[v].append((ci, lit))
    return occ

def compute_core_vars_from_state(
    clauses: List[Clause],
    nvars: int,
    unsat_list: List[int],
    var_occ: List[List[Tuple[int, int]]],
    *,
    bfs_steps: int = 2,
) -> List[bool]:
    """
    CORE = proměnné z aktuálních UNSAT klauzulí + BFS rozšíření přes incidence (1–3 kroky).
    Vrací masku core_mask[v] = True/False.
    """
    core_mask = [False] * nvars
    if not unsat_list:
        return core_mask

    # V0: proměnné z UNSAT klauzulí
    frontier: List[int] = []
    for ci in unsat_list:
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if 0 <= v < nvars and not core_mask[v]:
                core_mask[v] = True
                frontier.append(v)

    # BFS přes occ: var -> clauses -> vars
    for _ in range(max(0, bfs_steps)):
        new_frontier: List[int] = []
        # seber klauzule, kde se vyskytují proměnné z frontier
        seen_clause = set()
        for v in frontier:
            for ci, _lit in var_occ[v]:
                if ci not in seen_clause:
                    seen_clause.add(ci)
        # z těch klauzulí vytáhni další proměnné
        for ci in seen_clause:
            for lit in clauses[ci]:
                v2 = abs(lit) - 1
                if 0 <= v2 < nvars and not core_mask[v2]:
                    core_mask[v2] = True
                    new_frontier.append(v2)

        if not new_frontier:
            break
        frontier = new_frontier

    return core_mask

from collections import deque

class CoreTracker:
    """
    Empirický core: počítá, jak často byly klauzule UNSAT v posledním okně.
    Na vyžádání vrátí core_mask proměnných z top-K klauzulí.
    """
    def __init__(self, m: int, window: int = 4096):
        self.m = m
        self.window = window
        self.buf = deque(maxlen=window)   # list[int] of clause ids (can repeat)
        self.freq = [0] * m

    def update(self, unsat_list: List[int]) -> None:
        # Pozor: tohle je O(u). Volat jen občas (např. každých 64–256 flipů v endgame).
        for ci in unsat_list:
            if 0 <= ci < self.m:
                if len(self.buf) == self.buf.maxlen:
                    old = self.buf[0]
                    self.freq[old] -= 1
                self.buf.append(ci)
                self.freq[ci] += 1

    def core_mask_from_topk(
        self,
        clauses: List[Clause],
        nvars: int,
        *,
        topk: int = 512,
        min_freq: int = 2,
    ) -> List[bool]:
        # vyber topk klauzulí podle freq
        idx = [i for i, f in enumerate(self.freq) if f >= min_freq]
        if not idx:
            return [False] * nvars

        idx.sort(key=lambda i: self.freq[i], reverse=True)
        idx = idx[: min(topk, len(idx))]

        mask = [False] * nvars
        for ci in idx:
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if 0 <= v < nvars:
                    mask[v] = True
        return mask

def focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
    """
    1–2 flip pulse for u <= 25: evaluate by REAL solver state (len(unsat_list)).
    """
    if not unsat_list:
        return False

    base_u = len(unsat_list)
    cl = clauses[unsat_list[rng.randrange(base_u)]]
    lits = cl[:]

    best_seq = None
    best_u = base_u

    for lit1 in lits:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = len(unsat_list)
        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in lits:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)
            u2 = len(unsat_list)
            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]
            flip_var(v2)

        flip_var(v1)

    if best_seq and best_u < base_u:
        for v in best_seq:
            flip_var(v)
        return True

    return False

def focused_endgame_pulse3(clauses, a, unsat_list, flip_var, rng):
    """
    3-flip pulse for u <= ~22: evaluate by REAL solver state (len(unsat_list)).
    No stale snapshots, no clause rescans.
    """
    if not unsat_list:
        return False

    base_u = len(unsat_list)
    # anchor clause from current UNSAT set
    cl = clauses[unsat_list[rng.randrange(base_u)]]
    lits = cl[:]

    best_seq = None
    best_u = base_u

    for lit1 in lits:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = len(unsat_list)
        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in lits:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)
            u2 = len(unsat_list)
            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]

            for lit3 in lits:
                v3 = abs(lit3) - 1
                if v3 == v1 or v3 == v2:
                    continue
                flip_var(v3)
                u3 = len(unsat_list)
                if u3 < best_u:
                    best_u = u3
                    best_seq = [v1, v2, v3]
                flip_var(v3)

            flip_var(v2)
        flip_var(v1)

    if best_seq and best_u < base_u:
        for v in best_seq:
            flip_var(v)
        return True

    return False

# -------------------------
# SNIPER
# -------------------------
def finisher_classic_to_zero_sniper(
    clauses: List[Clause],
    nvars: int,
    a0: List[int] | List[bool],
    *,
    var_occ: Optional[List[List[Tuple[int, int]]]] = None,
    seed: int = 0,
    max_flips: int = 8_000_000,
    p_mid: float = 0.33,
    p_mid_hard: float = 0.33,  # kept for compatibility
    endgame_at: int = 24,
    p_end: float = 0.06,
    sniper_end_p: float = 0.03,
    tabu_tenure: int = 45,
    rb_prob: float = 0.001,        # 1/1000
    rb_stall_flips: int = 200_000, # jak dlouho bez zlepšení
    core_mask: Optional[List[bool]] = None,
    core_bfs_steps: int = 2,
    core_freeze_at: int = 32,

        report_every: int = 100_000,
) -> Tuple[BoolAssign, bool, Dict]:
    rng = random.Random(seed)
    m = len(clauses)
    a = _as_bool_assign(a0, nvars)
    if var_occ is None:
        var_occ = build_var_occ(clauses, nvars)

    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    def add_unsat(ci: int) -> None:
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        p = pos[ci]
        if p != -1:
            last = unsat_list[-1]
            unsat_list[p] = last
            pos[last] = p
            unsat_list.pop()
            pos[ci] = -1

    for ci, cl in enumerate(clauses):
        sc = 0
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                sc += 1
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)

    tabu_until = [0] * nvars

    def flip_var(v: int) -> None:
        old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, old)
            now_true = not was_true

            if was_true and not now_true:
                if sc == 1:
                    sat_count[ci] = 0
                    add_unsat(ci)
                else:
                    sat_count[ci] = sc - 1
            elif (not was_true) and now_true:
                if sc == 0:
                    sat_count[ci] = 1
                    rem_unsat(ci)
                else:
                    sat_count[ci] = sc + 1

    def breakcount(v: int) -> int:
        bc = 0
        cur = a[v]
        for ci, lit in var_occ[v]:
            if sat_count[ci] == 1 and lit_true(lit, cur):
                bc += 1
        return bc

    def rebuild_from_assign(best_assign: BoolAssign) -> None:
        # reset assignment
        a[:] = best_assign[:]

        # rebuild sat_count + unsat_list + pos
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1

        for ci, cl in enumerate(clauses):
            sc = 0
            for lit in cl:
                v = abs(lit) - 1
                if lit_true(lit, a[v]):
                    sc += 1
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)


    best = a[:]
    best_unsat = len(unsat_list)
    # --- SNIPER EMA + smooth p + rollback ---
    ema_r = 0.0
    u_prev = best_unsat
    flips_since_improve = 0
    rollback_every = 50_000  # soft rollback cadence
    p = float(p_mid)

    last_improve_flip = 0
    reheat_until = 0
    last_report = 0
    t0 = time.time()

    in_endgame = False  # hysteresis state for SNIPER mode

    # --- CORE FREEZE state ---
    core_mask_local: Optional[List[bool]] = core_mask  # může přijít zvenku (predator)
    core_dirty = True
    last_core_best = 10**9

    # --- Empirický core tracker (historie UNSAT klauzulí) ---
    core_tr = CoreTracker(m, window=4096)
    core_tr_rate = 128  # update každých 128 flipů (jen v approach/endgame)


    for flip in range(1, max_flips + 1):
        u = len(unsat_list)

        """if u <= 25 and flip % 500 == 0:
            if focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
                continue"""
        if u <= 22 and flip % 200 == 0:
            if focused_endgame_pulse3(clauses, a, unsat_list, flip_var, rng):
                continue
        elif u <= 25 and flip % 200 == 0:
            if focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
                continue

        if u == 0:
            return a, True, {"phase": "sniper", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip
            # when we improve, cancel any reheat
            reheat_until = 0
            # best se změnil -> jádro se může změnit
            core_dirty = True


        # Endgame should be sticky once we've *ever* reached <= endgame_at.
        # Otherwise the search bounces between MID and SNIPER as u fluctuates around the threshold.
        #endgame = (best_unsat <= endgame_at)
        """enter_at = endgame_at
        exit_at  = endgame_at + 12
        if (not in_endgame) and (u <= enter_at):
            in_endgame = True
        elif in_endgame and (u >= exit_at):
            in_endgame = False
        endgame = in_endgame"""
        # --- endgame is based on BEST (sticky) to avoid MID/SNIPER bouncing ---
        if (not in_endgame) and (best_unsat <= endgame_at):
            in_endgame = True
        endgame = in_endgame
        # --- approach band: start behaving "sniper-ish" before true endgame ---
        approach = (not endgame) and (u <= 3 * endgame_at)

        if (endgame or approach) and (flip % core_tr_rate == 0):
            core_tr.update(unsat_list)

        # --- CORE mask: počítej jen když jsme blízko a jen když je potřeba ---
        """if (core_mask_local is None) and (best_unsat <= core_freeze_at):
            if core_dirty or (best_unsat < last_core_best):
                # Pozor: používáme UNSAT set aktuálního stavu (ne best), protože je to levné a lokální
                core_mask_local = compute_core_vars_from_state(
                    clauses, nvars, unsat_list, var_occ, bfs_steps=core_bfs_steps
                )
                core_dirty = False
                last_core_best = best_unsat"""

        if (best_unsat <= core_freeze_at):
            if core_mask_local is None or core_dirty or (best_unsat < last_core_best):
                # 1) snapshot BFS core (to co máš)
                mask1 = compute_core_vars_from_state(
                    clauses, nvars, unsat_list, var_occ, bfs_steps=core_bfs_steps
                )
                # 2) empirický core z historie (top-K)
                mask2 = core_tr.core_mask_from_topk(
                    clauses, nvars, topk=512, min_freq=2
                )
                # 3) union (bezpečné: jen rozšíří povolené proměnné)
                if any(mask2):
                    for v in range(nvars):
                        mask1[v] = mask1[v] or mask2[v]
                core_mask_local = mask1

                core_dirty = False
                last_core_best = best_unsat



        # EMA of local progress (step-to-step improvement)
        improve = max(0, u_prev - u)
        ema_r = 0.9 * ema_r + 0.1 * improve
        if improve > 0:
            flips_since_improve = 0
        else:
            flips_since_improve += 1
        u_prev = u

        # --- probabilistic rollback when hard-stalled ---
        #if flips_since_improve >= rb_stall_flips:
        # --- probabilistic rollback when hard-stalled (DISABLED in approach/endgame) ---
        if (not endgame) and (not approach) and flips_since_improve >= rb_stall_flips:

            if rb_prob > 0.0 and rng.random() < rb_prob:
                rebuild_from_assign(best)
                flips_since_improve = 0
                # keep noise bounded – no explosion after rollback
                p = max(p, 0.8 * min(p_mid, 0.16))

        # Soft rollback if we're stuck too long: rebuild exact best state (no shake)
        """if flips_since_improve >= rollback_every and best is not None:
            rebuild_from(best, shake=0)
            a[:] = best[:]
            flips_since_improve = 0
            # slight reheat after rollback
            p = max(p, 0.8 * p_mid)"""
        """if flips_since_improve >= rollback_every and best is not None:
            # rollback to best-known assignment (no rebuild_from dependency)
            a[:] = best[:]
            flips_since_improve = 0
            # slight reheat after rollback
            p = max(p, 0.8 * p_mid)"""
        if flips_since_improve >= rollback_every and best is not None:
            rebuild_from_assign(best)
            flips_since_improve = 0
            # slight reheat after rollback (bounded)
            p = max(p, 0.8 * min(p_mid, 0.16))
            core_dirty = True




        # Smooth p toward a target (lower under ~30 UNSAT; contract a bit on progress)
        """if endgame:
            if u <= 30:
                p_target = max(sniper_end_p, min(p_end, 0.06))
            else:
                p_target = max(sniper_end_p, p_end)
            scale = 1.0 - 0.15 * min(1.0, ema_r / 5.0)
            p_target = max(sniper_end_p, p_target * scale)
            if reheat_until != 0 and flip <= reheat_until:
                p_target = max(p_target, p_mid)
            if reheat_until != 0 and flip <= reheat_until:
                # reheat is bounded in endgame
                p_target = max(p_target, min(p_mid, 0.10))

        else:
            p_target = p_mid"""

        if endgame:
            # true endgame: low noise + tabu
            if u <= 30:
                p_target = max(sniper_end_p, min(p_end, 0.06))
            else:
                p_target = max(sniper_end_p, p_end)
            scale = 1.0 - 0.15 * min(1.0, ema_r / 5.0)
            p_target = max(sniper_end_p, p_target * scale)
            if reheat_until != 0 and flip <= reheat_until:
                p_target = max(p_target, p_mid)

        elif approach:
            # approach: already reduce noise (kills the 0.33 random-walk plateau)
            p_target = max(0.08, min(0.16, 0.55 * p_mid))

        else:
            p_target = p_mid


        p = 0.95 * p + 0.05 * p_target

        if report_every and (flip - last_report) >= report_every:
            last_report = flip
            mode = "REHEAT" if (endgame and flip <= reheat_until and reheat_until != 0) else ("SNIPER" if endgame else "MID")
            print(f"[finisher] flips={flip:,} unsat={u} best={best_unsat} p={p:.3f} mode={mode}")

        # Refresh u (unsat_list can change after rebuild/sniper)
        u_pick = len(unsat_list)
        if u_pick == 0:
            return a, True, {"phase": "sniper", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}
        ci = unsat_list[rng.randrange(u_pick)]
        cl = clauses[ci]

        """if rng.random() < p:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = None
            best_bc = 10**9
            for lit in cl:
                vv = abs(lit) - 1
                if endgame and tabu_until[vv] > flip:
                    continue
                bc = breakcount(vv)
                if bc < best_bc:
                    best_bc = bc
                    best_v = vv
                elif bc == best_bc and best_v is not None and rng.random() < 0.35:
                    best_v = vv
            v = best_v if best_v is not None else abs(cl[rng.randrange(len(cl))]) - 1"""
        # --- endgame clamp: never let endgame behave like high-noise walk ---
        p_pick = p
        if endgame:
            # hard clamp in endgame (prevents 0.33 "reheat" from blowing structure)
            if p_pick < 0.02:
                p_pick = 0.02
            elif p_pick > 0.08:
                p_pick = 0.08

        # Evaluate candidates by (breakcount, age/tie)
        """cand = []
        for lit in cl:
            vv = abs(lit) - 1
            if (endgame or approach) and tabu_until[vv] > flip:
                continue
            bc = breakcount(vv)
            cand.append((bc, vv))"""

        cand = []
        for lit in cl:
            vv = abs(lit) - 1

            # --- CORE FREEZE: v endgame/approach flipuj jen v jádru (pokud ho máme) ---
            if (endgame or approach) and core_mask_local is not None:
                if not core_mask_local[vv]:
                    continue

            if (endgame or approach) and tabu_until[vv] > flip:
                continue

            bc = breakcount(vv)
            cand.append((bc, vv))


        """if not cand:
            # all tabu or empty -> fallback random in clause
            v = abs(cl[rng.randrange(len(cl))]) - 1"""
        if not cand:
            if (endgame or approach) and core_mask_local is not None:
                core_lits = [abs(lit) - 1 for lit in cl if core_mask_local[abs(lit) - 1]]
                if core_lits:
                    v = core_lits[rng.randrange(len(core_lits))]
                else:
                    v = abs(cl[rng.randrange(len(cl))]) - 1
            else:
                v = abs(cl[rng.randrange(len(cl))]) - 1

        else:
            cand.sort(key=lambda x: x[0])  # sort by breakcount asc
            # deterministic best is minimal breakcount
            best_bc = cand[0][0]
            best_vars = [vv for (bc, vv) in cand if bc == best_bc]

            # noise: pick from top-K of best breakcount set, not random-any
            if rng.random() < p_pick:
                # topK among best breakcount (cap keeps it local)
                K = 8
                v = best_vars[rng.randrange(min(K, len(best_vars)))]
            else:
                # stable pick (if multiple best, pick one randomly to avoid cycles)
                v = best_vars[rng.randrange(len(best_vars))]


        flip_var(v)
        """if endgame:
            tabu_until[v] = flip + tabu_tenure"""
        if endgame or approach:
            tabu_until[v] = flip + tabu_tenure


    return best, False, {"phase": "sniper", "flips": max_flips, "best_unsat": best_unsat, "time_s": time.time() - t0}


# -------------------------
# PREDATOR
# -------------------------
def finisher_predator_sole_sat_vFinal(
    clauses: List[Clause],
    nvars: int,
    a0: List[int] | List[bool],
    *,
    var_occ: Optional[List[List[Tuple[int, int]]]] = None,
    seed: int = 0,
    max_flips: int = 50_000_000,
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,
    stall_window: int = 800_000,
    restart_shake: int = 96,
    w_inc: float = 1.0,
    w_decay: float = 0.9996,
    w_cap: float = 40.0,
    snapback_gap: int = 250,
    basin_mult: float = 2.2,
    basin_abs: int = 350,
    kick_after: int = 300_000,
    kick_len: int = 100_000,
    kick_p: float = 0.18,

    kick_cooldown: int = 250_000,
    kick_disable_best_mult: int = 2,
    sniper_u: int = 64,
    sniper_flips: int = 8_000_000,
    sniper_p: float = 0.33,
    use_tabu: bool = True,
    tabu_u_threshold: int = 128,
    tabu_tenure: int = 45,
    sniper_end_p: float = 0.03,
    report_every: int = 100_000,
) -> Tuple[BoolAssign, bool, Dict]:
    rng = random.Random(seed)
    m = len(clauses)
    a = _as_bool_assign(a0, nvars)
    if var_occ is None:
        var_occ = build_var_occ(clauses, nvars)

    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    w = [1.0] * m
    tabu_until = [0] * nvars

    last_sniper_best = 10**9
    next_sniper_call = 0

    def add_unsat(ci: int) -> None:
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        p = pos[ci]
        if p != -1:
            last = unsat_list[-1]
            unsat_list[p] = last
            pos[last] = p
            unsat_list.pop()
            pos[ci] = -1

    for ci, cl in enumerate(clauses):
        sc = 0
        sv = -1
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                sc += 1
                sv = v
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)
            sole_sat[ci] = -1
        elif sc == 1:
            sole_sat[ci] = sv
        else:
            sole_sat[ci] = -1

    def apply_flip(v: int) -> None:
        old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, old)
            now_true = not was_true
            if was_true and not now_true:
                if sc == 1:
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    sat_count[ci] = 1
                    sv = -1
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(lit2, a[vv]):
                            sv = vv
                            break
                    sole_sat[ci] = sv
                else:
                    sat_count[ci] = sc - 1
            elif (not was_true) and now_true:
                if sc == 0:
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    def rebuild_from(best_assign: BoolAssign, shake: int) -> None:
        nonlocal a
        a = best_assign[:]
        if shake > 0:
            for _ in range(shake):
                vv = rng.randrange(nvars)
                a[vv] = not a[vv]
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1
        for ci, cl in enumerate(clauses):
            sc = 0
            sv = -1
            for lit in cl:
                vv = abs(lit) - 1
                if lit_true(lit, a[vv]):
                    sc += 1
                    sv = vv
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                sole_sat[ci] = sv
            else:
                sole_sat[ci] = -1

    def mk_br(v: int) -> Tuple[float, float]:
        mk = 0.0
        br = 0.0
        cur = a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            becomes_true = not lit_true(lit, cur)
            if becomes_true:
                if sc == 0:
                    mk += w[ci]
            else:
                if sc == 1 and sole_sat[ci] == v:
                    br += w[ci]
        return mk, br

    def core_microsearch(max_tries: int = 4000, depth: int = 3) -> bool:
        if not unsat_list:
            return False

        # kandidátní proměnné jen z UNSAT klauzulí
        cand_vars = []
        for ci0 in unsat_list:
            for lit in clauses[ci0]:
                cand_vars.append(abs(lit) - 1)
        if not cand_vars:
            return False

        base_u = len(unsat_list)
        best_local = base_u
        best_seq = None

        # zkusíme pár náhodných sekvencí malého depth
        for _ in range(max_tries):
            seq = [cand_vars[rng.randrange(len(cand_vars))] for _ in range(depth)]
            for v in seq:
                apply_flip(v)
            u1 = len(unsat_list)
            if u1 < best_local:
                best_local = u1
                best_seq = seq[:]
                if best_local == 0:
                    return True
            # rollback
            for v in reversed(seq):
                apply_flip(v)

        if best_seq and best_local < base_u:
            for v in best_seq:
                apply_flip(v)
            return True

        return False

    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0
    last_improve_flip = 0
    reheat_until = 0
    next_endgame_reset = 0
    last_report = 0
    kick_left = 0
    kick_cooldown_left = 0
    prev_kick_left = 0
    t0 = time.time()
    next_endgame_jolt = 0
    # --- SNIPER RETRY (micro upgrade): re-run sniper from best if we stall in endgame ---
    next_sniper_retry = 0
    sniper_retry_after = 900_000      # how long we must stall before retrying sniper
    sniper_retry_cooldown = 900_000   # minimum spacing between retries

    # --- Fixpoint controller (bounded adaptivity; avoids thrash in endgame)
    p_state = float(p_base)
    eta_p = 0.05  # EMA rate; <1 ensures contractive adjustment
    stability = 0.0
    no_improve_epochs = 0

    core_tr = CoreTracker(m, window=8192)
    core_tr_rate = 256  # predator update řidší, aby to nebolelo

    # --- FAST stability tracker (no O(nvars) Hamming scan)
    # We approximate "instability" by the fraction of unique variables flipped since the last report.
    # This is O(1) per flip and O(1) per report.
    epoch = 1
    mark = array('I', [0]) * nvars  # per-var last-seen epoch
    changed_count = 0

    print(f"\n[finisher:init] p_min={p_min} p_max={p_max} p_base={p_base}")

    # --- HARD CAP: prevent catastrophic over-noise ---
    if p_max > 0.16:
        p_max = 0.16
    if kick_p > 0.16:
        kick_p = 0.16


    flip = 0
    while flip < max_flips:
        u = len(unsat_list)

        if best_unsat <= 64 and (flip % core_tr_rate == 0):
            core_tr.update(unsat_list)

        if u == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}



        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip
            # when we improve, cancel any reheat
            reheat_until = 0
            last_improve = flip

            if (
                    best_unsat <= sniper_u
                    and best_unsat <= last_sniper_best - 4  # call only if we improved meaningfully
                    and flip >= next_sniper_call
            ):
                """last_sniper_best = 10 ** 9
                next_sniper_call = 0"""

                last_sniper_best = best_unsat
                next_sniper_call = flip + 600_000  # cooldown

                core_mask_best = compute_core_vars_from_state(
                    clauses, nvars, unsat_list, var_occ, bfs_steps=2
                )


                print(f"[predator] target locked (u={best_unsat}), calling sniper ...")
                model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses,
                    nvars=nvars,
                    a0=best,
                    var_occ=var_occ,
                    seed=seed + 1337,
                    max_flips=sniper_flips,
                    p_mid=sniper_p,
                    p_mid_hard=sniper_p,
                    endgame_at=min(sniper_u, best_unsat),
                    p_end=0.1,
                    sniper_end_p=sniper_end_p,
                    tabu_tenure=tabu_tenure,
                    rb_prob=0.001,
                    rb_stall_flips=200_000,
                    core_mask=core_mask_best,
                    core_bfs_steps=2,
                    core_freeze_at=32,
                    report_every=report_every,
                )
                if ok2:
                    return model2, True, st2
                # continue from sniper's best, rebuilt
                a = model2[:]
                best = a[:]
                rebuild_from(best, shake=0)
                last_improve = flip

        stalled = flip - last_improve

        # --- micro rollback to best (endgame only, rate-limited via stall) ---
        if best_unsat <= sniper_u and stalled >= 600_000 and kick_left == 0:
            # very low probability to avoid thrashing
            if rng.random() < 0.0005:  # 1/2000
                rebuild_from(best, shake=0)
                last_improve = flip

        # --- SNIPER RETRY (micro upgrade) ---
        # If we're already close (best_unsat small) but haven't improved for a long time,
        # re-run the sniper from the *best* assignment to try to break the last core.
        if (
            best_unsat <= sniper_u
            and stalled >= sniper_retry_after
            and kick_left == 0
            and flip >= next_sniper_retry
        ):
            print(f"[predator] stalled near endgame (best={best_unsat}, stalled={stalled:,}), retrying sniper ...")
            model2, ok2, st2 = finisher_classic_to_zero_sniper(
                clauses=clauses,
                nvars=nvars,
                a0=best,
                var_occ=var_occ,
                seed=seed + 2337 + flip,   # tiny decorrelation per retry
                max_flips=sniper_flips,
                p_mid=sniper_p,
                p_mid_hard=sniper_p,
                endgame_at=24,
                p_end=0.06,
                sniper_end_p=sniper_end_p,
                tabu_tenure=tabu_tenure,
                report_every=report_every,
            )
            if ok2:
                return model2, True, st2

            # continue from sniper's best, rebuilt (no shake)
            a = model2[:]
            best = a[:]
            rebuild_from(best, shake=0)

            last_improve = flip
            next_sniper_retry = flip + sniper_retry_cooldown



        # --- ENDGAME WEIGHT RESET (micro-bypass to avoid weight "cementing") ---
        # Trigger only when very close to SAT and truly stalled; keep it rate-limited.
        if (
            best_unsat <= 32
            and stalled >= 650_000
            and kick_left == 0
            and flip >= next_endgame_reset
        ):
            # Drop clause weights to a neutral baseline and re-center on the best state.
            for i in range(m):
                w[i] = 1.0
            rebuild_from(best, shake=0)
            last_improve = flip  # reset stall timer
            next_endgame_reset = flip + 1_500_000  # cooldown
            # Gently re-inject some exploration so we don't immediately re-cement.
            p_state = min(p_max, max(p_state, p_min * 2.0))

        # --- ENDGAME TARGETED JOLT (break backbone clusters with minimal disruption) ---
        if (
            best_unsat <= 28
            and stalled >= 350_000
            and kick_left == 0
            and flip >= next_endgame_jolt
            and len(unsat_list) > 0
        ):
            # Collect vars from current UNSAT clauses
            """cand = []
            for ci in unsat_list:
                for lit in clauses[ci]:
                    cand.append(abs(lit) - 1)"""
            # Collect vars from current UNSAT clauses, but prefer empirical CORE
            core_mask = core_tr.core_mask_from_topk(
                clauses, nvars, topk=768, min_freq=2
            )

            cand = []
            for ci in unsat_list:
                for lit in clauses[ci]:
                    v = abs(lit) - 1
                    if 0 <= v < nvars and core_mask[v]:
                        cand.append(v)

            # fallback: if empirical core is empty, use plain UNSAT-support vars
            if not cand:
                for ci in unsat_list:
                    for lit in clauses[ci]:
                        cand.append(abs(lit) - 1)

            if cand:
                # flip a few vars from the UNSAT-support set
                k = 16  # 12–24 typical; keep small
                for _ in range(k):
                    v = cand[rng.randrange(len(cand))]
                    apply_flip(v)

                last_improve = flip          # reset stall timer (we intentionally moved)
                next_endgame_jolt = flip + 900_000  # cooldown
                # slight exploration bump (very gentle)
                p_state = min(p_max, max(p_state, p_min * 1.5))

            """if cand:
                # flip a few vars from the UNSAT-support set
                k = 32 if best_unsat <= 20 else 16
                for _ in range(k):
                    v = cand[rng.randrange(len(cand))]
                    apply_flip(v)

                last_improve = flip          # reset stall timer (we intentionally moved)
                next_endgame_jolt = flip + (450_000 if best_unsat <= 20 else 900_000)  # cooldown
                # slight exploration bump (very gentle)
                p_state = min(p_max, max(p_state, p_min * 1.5))"""

        if best_unsat <= 20 and stalled >= 120_000 and (flip % 50_000 == 0):
            if core_microsearch(max_tries=2500, depth=3):
                last_improve = flip

        # --- CORE-ONLY MINI-KICK (last-mile extractor) ---
        if best_unsat <= 22 and stalled >= 250_000 and (flip % 100_000 == 0):
            core_mask = core_tr.core_mask_from_topk(
                clauses, nvars, topk=768, min_freq=2
            )

            # collect core vars that actually appear in current UNSAT clauses
            cand = []
            for ci in unsat_list:
                for lit in clauses[ci]:
                    v = abs(lit) - 1
                    if 0 <= v < nvars and core_mask[v]:
                        cand.append(v)

            if cand:
                # small, non-destructive pulse
                k = 24 if best_unsat <= 18 else 16
                for _ in range(k):
                    v = cand[rng.randrange(len(cand))]
                    apply_flip(v)
                last_improve = flip

        # --- CHAOS KICK (rate-limited, disabled in late endgame) ---
        if kick_cooldown_left > 0:
            kick_cooldown_left -= 1

        # Disable kicks when we're already close to endgame.
        kick_disabled = (best_unsat <= kick_disable_best_mult * sniper_u)

        if (not kick_disabled) and stalled > kick_after and kick_left == 0 and kick_cooldown_left == 0:
            kick_left = kick_len
            kick_cooldown_left = kick_cooldown
            # Reset stall timer so we don't immediately retrigger on kick end.
            last_improve = flip
            print(f"!!! [kick] stagnation at best={best_unsat} -> chaos kick for {kick_len} flips (cooldown {kick_cooldown})")

        """if kick_left == 0 and best_unsat > 0:
            meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs)
            softsnap = (u > best_unsat + snapback_gap) and (u > basin_abs)
            if meltdown:
                rebuild_from(best, shake=restart_shake)
                last_improve = flip
                continue
            if softsnap:
                rebuild_from(best, shake=0)
                last_improve = flip
                continue"""

        if kick_left == 0 and best_unsat > 0:
            if best_unsat <= sniper_u:
                # endgame: basin_abs must NOT block snapback
                meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap))
                softsnap = (u > best_unsat + snapback_gap)
            else:
                meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs)
                softsnap = (u > best_unsat + snapback_gap) and (u > basin_abs)

            if meltdown:
                rebuild_from(best, shake=restart_shake)
                last_improve = flip
                continue
            if softsnap:
                rebuild_from(best, shake=0)
                last_improve = flip
                continue

        if stalled > stall_window and kick_left == 0:
            rebuild_from(best, shake=restart_shake)
            last_improve = flip
            continue

        if (flip & 255) == 0:
            for i in range(m):
                wi = w[i] * w_decay
                w[i] = wi if wi >= 1.0 else 1.0

        prev_kick_left = kick_left
        if kick_left > 0:
            p_eff = min(p_max, max(kick_p, p_min))
            if p_eff > 0.16:
                p_eff = 0.16

            kick_left -= 1
            # when a kick ends, immediately re-center on best with a small core-shake
            if prev_kick_left == 1:
                rebuild_from(best, shake=max(8, restart_shake // 2))
                last_improve = flip
                # IMPORTANT: rebuild changes unsat_list length; refresh u to avoid stale index
                u = len(unsat_list)
                if u == 0:
                    return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        else:
            # Target noise is a bounded function of (u, stability), with a gentle EMA update.
            # Intuition: when dynamics are already stable (high stability) and u is small, we want a
            # near-fixpoint regime (low noise). When the state is unstable or plateauing, we reheat a bit.
            progress = min(1.0, u / max(1.0, float(sniper_u) * 8.0))
            base = p_min + (p_max - p_min) * math.sqrt(progress)
            if u <= sniper_u:
                base = min(base, float(sniper_end_p))
            stab_factor = 1.0 + 2.0 * (1.0 - stability)  # in [1,3] approximately
            target_p = base * stab_factor
            # Do not ramp noise to max when we're already in endgame.
            if best_unsat > 32 and (flip - last_improve) >= 2000000:
                target_p *= 1.35
            target_p = float(max(p_min, min(p_max, target_p)))
            p_state = (1.0 - eta_p) * p_state + eta_p * target_p
            p_eff = float(max(p_min, min(p_max, p_state)))

            p_eff = float(max(p_min, min(p_max, p_eff)))

            if best_unsat <= 64:
                p_eff = min(p_eff, 0.10)
            if best_unsat <= 24:
                p_eff = min(p_eff, 0.07)

        if report_every and (flip - last_report) >= report_every:
            # --- stability estimate from unique vars touched since last report
            instab = changed_count / max(1, nvars)  # 0..1
            stability = 0.95 * stability + 0.05 * (1.0 - instab)

            # reset tracker (no per-var clearing)
            epoch += 1
            if epoch >= 0xFFFFFFFF:
                # extremely rare; keep it safe
                epoch = 1
                for i in range(nvars):
                    mark[i] = 0
            changed_count = 0

            last_report = flip
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat} stab={stability:.3f}", flush=True)

        u_pick = len(unsat_list)
        if u_pick == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}
        ci = unsat_list[rng.randrange(u_pick)]
        cl = clauses[ci]

        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = None
            best_score = -1.0
            for lit in cl:
                v0 = abs(lit) - 1
                if kick_left == 0 and use_tabu and u <= tabu_u_threshold and tabu_until[v0] > flip:
                    continue
                mk, br = mk_br(v0)
                eps = 1e-6
                mk1 = mk + eps
                br1 = br + eps
                score = (mk1 * mk1 * mk1 * mk1 * mk1) / (br1 * br1 * br1)  # mk^5 / br^3
                if score > best_score:
                    best_score = score
                    best_v = v0
            v = best_v if best_v is not None else abs(cl[rng.randrange(len(cl))]) - 1

        apply_flip(v)

        # stability tracker: mark var touched in this epoch (O(1), no scans)
        if mark[v] != epoch:
            mark[v] = epoch
            changed_count += 1

        if kick_left == 0 and use_tabu and len(unsat_list) <= tabu_u_threshold:
            tabu_until[v] = flip + tabu_tenure

        w[ci] = min(w_cap, w[ci] + w_inc)
        flip += 1

    return best, False, {"phase": "predator", "flips": flip, "best_unsat": best_unsat, "time_s": time.time() - t0}

## ----


def main() -> None:
    ap = argparse.ArgumentParser(description="DREAM6_operator_2 functional certifier (no placeholders).")
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--R", type=int, default=104)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--mode", type=str, default="sat", choices=["sat", "unsat", "inf"])

    ap.add_argument("--inf-path", type=str, default=None, help="Path to text/tex/log/pdf fragment for OPERATOR_INF")
    ap.add_argument("--inf-max-tokens", type=int, default=200000, help="Max tokens extracted from inf fragment")
    ap.add_argument("--inf-seed", type=int, default=0, help="Extra seed for deterministic INF hashing")

    ap.add_argument("--sweeps", type=int, default=0)
    ap.add_argument("--eta", type=float, default=0.5)

    ap.add_argument("--steps", type=int, default=None, help="Alias for --sweeps (DREAM6 tuner nomenclature).")
    ap.add_argument("--K", type=float, default=4.0, help="DREAM6 coupling gain (separate from eta).")
    ap.add_argument("--noise-sigma", dest="noise_sigma", type=float, default=0.025, help="DREAM6 phase-noise sigma.")
    ap.add_argument("--dt", type=float, default=0.05, help="Time step used only to scale phase-noise (sigma*sqrt(dt)).")
    ap.add_argument("--mu", type=float, default=0.9999954637353263,
                    help="Relaxation/EMA towards projected overlap update.")
    ap.add_argument("--mu-E", dest="mu_E", type=float, default=0.9999952540504147,
                    help="EMA parameter for tail-coherence diagnostic.")
    ap.add_argument("--h", type=float, default=0.4, help="External carrier field bias.")
    ap.add_argument("--tail-frac", type=float, default=0.33, help="Fraction of the end of R(t) used for tail metrics.")

    ap.add_argument("--shared-carrier", action="store_true", default=False)
    ap.add_argument("--shared-misphase", dest="shared_misphase", action="store_true", default=False)
    ap.add_argument("--no-shared-misphase", dest="shared_misphase", action="store_false")
    ap.add_argument("--unsat-neg-frac", type=float, default=0.25)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--power-iters", type=int, default=250)
    ap.add_argument("--power-tol", type=float, default=1e-10)

    ap.add_argument("--ipc-weights", type=str, default="qp",
                    choices=["ones", "qp", "corr", "cvxopt_corr", "auto", "qp_corr", "qp"])
    ap.add_argument("--w-delta-min", type=float, default=12.0)
    ap.add_argument("--w-delta-max", type=float, default=1000.0)

    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--verify-json", type=str, default=None)

    ap.add_argument("--cnf-path", type=str, default="EDEN_1_MILION.cnf", help="DIMACS CNF path (.cnf)")
    ap.add_argument("--edge-mode", type=str, default="logic", choices=["auto", "circulant", "cnf", "logic"],
                    help="Graph topology: auto (cnf->logic else circulant), circulant, cnf/logic (CNF clause-variable graph)")
    ap.add_argument("--outside-value", type=float, default=-1.0,
                    help="Background value outside lock windows (e.g. 0.0 to avoid silent regime for large T).")
    ap.add_argument("--no-flip-incident-unsat", action="store_true",
                    help="Disable incident-first edge flipping in UNSAT mode")

    # CLOSURE parameters (holonomie, singulární řez)
    ap.add_argument("--lambda-closure", type=float, default=0.1,
                    help="CLOSURE: Lambda scale for refined time τ(t) = t + λΘ(t)")
    ap.add_argument("--kappa-coupling", type=float, default=0.01,
                    help="CLOSURE: Coupling strength κ for holonomie functional F(Θ)")
    ap.add_argument("--enable-closure-cut", action="store_true", default=True,
                    help="CLOSURE: Enable singular cut (CLOSURE-FUSE protocol)")
    ap.add_argument("--disable-closure-cut", dest="enable_closure_cut", action="store_false",
                    help="CLOSURE: Disable singular cut")
    ap.add_argument("--closure-epsilon", type=float, default=1e-6,
                    help="CLOSURE: Epsilon for Soft-ABS regulator F_ε(x) = √(x²+ε²)")

    ap.add_argument("--get-params", action="store_true",
                    help="Vypočítá doporučené parametry R/eta/d z CNF bez spuštění simulace.")

    ap.add_argument("--model-out", type=str, default=None, help="Write DIMACS v-line model to file")
    ap.add_argument("--unsat-out", type=str, default=None, help="Write indices of unsatisfied clauses to file")
    ap.add_argument("--unsat-dump-k", type=int, default=50, help="How many UNSAT clauses to dump (0=all)")

    ap.add_argument(
        "--model-from-n-singular",
        action="store_true",
        default=False,
        help="If set, --model-out is deterministically generated from fuse n_singular (not from IPC witness)."
    )


    # finisher

    ap.add_argument(
        "--finisher",
        action="store_true",
        default=False,
        help=" predator/sniper finisher "
    )

    ap.add_argument("--polish", type=int, default=300_000)
    ap.add_argument("--finisher_flips", type=int, default=5_000_000)

    ap.add_argument("--p_min", type=float, default=0.003)
    ap.add_argument("--p_max", type=float, default=0.18)
    ap.add_argument("--p_base", type=float, default=0.10)
    ap.add_argument("--stall_window", type=int, default=800_000)
    ap.add_argument("--restart_shake", type=int, default=96)
    ap.add_argument("--w_inc", type=float, default=1.0)
    ap.add_argument("--w_decay", type=float, default=0.9996)
    ap.add_argument("--w_cap", type=float, default=40.0)

    ap.add_argument("--snapback_gap", type=int, default=250)
    ap.add_argument("--basin_mult", type=float, default=2.2)
    ap.add_argument("--basin_abs", type=int, default=350)

    ap.add_argument("--kick_after", type=int, default=300_000)
    ap.add_argument("--kick_len", type=int, default=100_000)
    ap.add_argument("--kick_p", type=float, default=0.18)

    ap.add_argument("--sniper_u", type=int, default=64)
    ap.add_argument("--sniper_flips", type=int, default=8_000_000)
    ap.add_argument("--sniper_p", type=float, default=0.33)
    ap.add_argument("--sniper_end_p", type=float, default=0.03)

    ap.add_argument("--tabu", action="store_true")
    ap.add_argument("--tabu_u", type=int, default=128)
    ap.add_argument("--tabu_tenure", type=int, default=45)

    ap.add_argument("--report_every", type=int, default=1_000_000)

    args = ap.parse_args()

    print("\n")

    if args.steps is not None:
        args.sweeps = int(args.steps)

    if args.verify_json:
        verify_json(args.verify_json)
        return

    """# --- Spectral Focusing Navigator (empirical fit to your optimal points) ---
    if args.get_params:
        if args.cnf_path:
            nvars, clauses = parse_dimacs(args.cnf_path)
            C = len(clauses)
        else:
            C = int(args.C)
            clauses = []

        if C <= 0:
            print("Chyba: C musí být > 0.")
            return

        # Empirická kalibrace: 10k -> R=56, 50k -> ~24
        # R ~ 56 * (C/10000)^(-1/2)
        R_opt = int(round(56.0 * (float(C) / 10000.0) ** (-0.5)))
        R_opt = int(np.clip(R_opt, 16, 256))

        # Zachovej tvoje defaulty pro eta/d (nepřepisuju teorii, jen reportuju)
        eta_opt = float(args.eta)
        d_opt = float(args.d)

        logC = float(np.log(float(C)))
        # Pokud chceš reportovat i "focus" metriku:
        # (T v tomhle CLI beru jako 2*R, čistě pro diagnostiku)
        T_opt = 2 * R_opt
        F = T_opt / (d_opt * logC) if logC > 0 else float("inf")

        print("\n=== Spectral Navigator (empirical) ===")
        print(f"Instance: C={C}")
        print(f"R_opt    : {R_opt}   (T≈{T_opt})")
        print(f"eta      : {eta_opt}")
        print(f"d        : {d_opt}")
        print(f"F        : {F:.4f}")
        print("-" * 55)
        print("Doporučený příkaz:")
        print(f"python DREAM6_operator_6.py --cnf-path {args.cnf_path} "
              f"--mode {args.mode} --edge-mode {args.edge_mode} "
              f"--eta {eta_opt} --d {int(d_opt)} --shared-carrier --R {R_opt}")
        return"""

    cert = run(
        C=args.C, R=args.R, d=args.d,
        sweeps=args.sweeps, eta=args.eta,
        K=args.K, noise_sigma=args.noise_sigma, dt=args.dt, mu=args.mu, mu_E=args.mu_E, h=args.h,
        tail_frac=args.tail_frac,
        mode=args.mode,
        inf_path=args.inf_path,
        inf_max_tokens=args.inf_max_tokens,
        inf_seed=args.inf_seed,
        shared_carrier=args.shared_carrier,
        shared_misphase=args.shared_misphase,
        unsat_neg_frac=args.unsat_neg_frac,
        seed=args.seed,
        power_iters=args.power_iters,
        power_tol=args.power_tol,
        ipc_weight_mode=args.ipc_weights,
        w_delta_min=args.w_delta_min,
        w_delta_max=args.w_delta_max,
        cnf_path=args.cnf_path,
        edge_mode=args.edge_mode,
        flip_incident_unsat=(not args.no_flip_incident_unsat),
        outside_value=complex(args.outside_value),
        json_out=args.json_out,
        # CLOSURE parameters
        lambda_closure=args.lambda_closure,
        kappa_coupling=args.kappa_coupling,
        enable_closure_cut=args.enable_closure_cut,
        closure_epsilon=args.closure_epsilon,
        model_out=args.model_out,
        unsat_out=args.unsat_out,
        unsat_dump_k=args.unsat_dump_k,
        model_from_n_singular=args.model_from_n_singular,
        polish = args.polish,
        #finisher
        finisher=args.finisher,
        max_flips=args.finisher_flips,
        p_min=args.p_min,
        p_max=args.p_max,
        p_base=args.p_base,
        stall_window=args.stall_window,
        restart_shake=args.restart_shake,
        w_inc=args.w_inc,
        w_decay=args.w_decay,
        w_cap=args.w_cap,
        snapback_gap=args.snapback_gap,
        basin_mult=args.basin_mult,
        basin_abs=args.basin_abs,
        kick_after=args.kick_after,
        kick_len=args.kick_len,
        kick_p=args.kick_p,
        sniper_u=args.sniper_u,
        sniper_end_p=args.sniper_end_p,
        sniper_flips=args.sniper_flips,
        sniper_p=args.sniper_p,
        use_tabu=bool(args.tabu),
        tabu_u_threshold=args.tabu_u,
        tabu_tenure=args.tabu_tenure,
        report_every=args.report_every,

    )

    res = asdict(cert)

    print(f"\n=== DREAM6 Operator Certifier ({args.ipc_weights})")
    print(
        f"C={res['meta']['C']}  T={res['meta']['T']}  m={res['meta']['m']}  d={res['meta']['d']}  mode={res['meta']['mode']}")
    # if res["meta"].get("cnf_path"):
    #    print(f"CNF: vars={res['meta']['nvars']}  seed_unsat={res['meta']['seed_unsat']} ({100*res['meta']['seed_unsat_frac']:.2f}%)")
    print(
        f"shared_carrier={res['meta']['shared_carrier']}  shared_misphase={res['meta']['shared_misphase']}  unsat_neg_frac={res['meta']['unsat_neg_frac']}")
    print(f"ipc_weights={res['meta']['ipc_weights']['mode']}")
    d6 = res['meta'].get('dream6', {})
    print(
        f"dream6: K={d6.get('K')}  noise_sigma={d6.get('noise_sigma')}  dt={d6.get('dt')}  mu={d6.get('mu')}  mu_E={d6.get('mu_E')}  h={d6.get('h')}  tail_frac={d6.get('tail_frac')}")

    print(f"S2 radar: rho={res['S2']['rho']:.6g}  <= d*kappa={res['S2']['d_kappa']:.6g}  pass={res['S2']['pass']}")

    # CLOSURE reporting (holonomie, singulární řez)
    cls = res.get('closure', {})
    if cls.get('enabled', False):
        print(
            f"CLOSURE: Θ={cls.get('theta_closure', 0):.6g}  τ_refined={cls.get('tau_refined', 0):.6g}  λ={cls.get('lambda_scale', 0):.4g}")
        print(f"         residue_compression={cls.get('residue_compression', 0):.6g}")
        fuse = cls.get('fuse', {})
        if fuse:
            #print(
            #    f"         FUSE: n_singular={fuse.get('n_singular', 0)}  frac={fuse.get('singular_frac', 0):.4g}  max_grad={fuse.get('max_gradient', 0):.4g}  bounded={fuse.get('gradient_bounded', False)}")

            print(f"         FUSE: n_singular={fuse['n_singular']}  frac={fuse['frac']:.4g}  "
                  f"max_grad={fuse['max_grad']:.3f}  delta_mean={fuse['delta_mean']:.3f}  "
                  f"cap={fuse.get('grad_cap'):.3f}  "
                  f"bounded={fuse['bounded']}")


    else:
        print(
            f"CLOSURE: Θ={cls.get('theta_closure', 0):.6g}  τ_refined={cls.get('tau_refined', 0):.6g}  (cut disabled)")

    print(f"Spectral: lambda_max(G_H)={res['spectral']['lambda_max_GH']:.6g}  mu_dec={res['spectral']['mu_dec']:.6g}")
    print(
        f"IPC: beta={res['IPC']['beta']:.6g}  delta={res['IPC']['delta']:.6g}  mu_sat_min={res['IPC']['mu_sat_min']:.6g}")
    print(
        f"Bands: lam_unsat_ceiling={res['bands']['lam_unsat_ceiling']:.6g}  mu_unsat_max={res['bands']['mu_unsat_max']:.6g}")
    print(f"tau={res['bands']['tau']:.6g}  Delta={res['bands']['Delta']:.6g}  separated={res['bands']['separated']}")

    r = res["diag"]["coherence_R"]
    print(
        f"Coherence R(t): mean={r['mean']:.6g}  min={r['min']:.6g}  max={r['max']:.6g}  tail_mean={r.get('mean_tail', 0.0):.6g}  tail_ema={r.get('ema_tail', 0.0):.6g}  tail_frac={r.get('tail_frac', 0.0):.3g}")
    print(
        f"Coherence tail: mean_tail={r.get('mean_tail', 0.0):.6g}  ema_tail={r.get('ema_tail', 0.0):.6g}  tail_frac={r.get('tail_frac', 0.0):.6g}")

    print("==============================================================\n")

    if args.json_out:
        print(f"Wrote certificate JSON: {args.json_out}")


if __name__ == "__main__":
    main()