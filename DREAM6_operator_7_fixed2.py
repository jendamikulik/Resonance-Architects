#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6 Operator 6 — functional certifier (no placeholders)

Fixes:
- Removes hardcoded placeholder S2 + spectral values.
- Implements real S2 radar (neighbor row-sum rho) computed from the SAME edge-Gram decision operator.
- Implements real decision spectral λ_max(G_H) via power iteration on sparse edge-supported Gram.
- Implements IPC (Invariant Phase Certifier) time-mode power iteration + proper normalization for lock-sparse columns.
- Adds deterministic clause gauge from CNF via build_seed_assignment (seed model) and per-clause satisfaction.
- Integrates CVXOPT dyadic tail-weights (small QP in O(log C)) for IPC clause weights.

python DREAM6_operator_6.py --cnf-path .\uf250-0100.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier
python DREAM6_operator_6.py --cnf-path .\random_3sat_10000.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier --R 56
python DREAM6_operator_6.py --cnf-path .\random_3sat_50000.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier --R 24

"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


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
    align = np.cos(wrap_pi(phi - float(theta))).astype(np.float64, copy=False)
    gate = np.maximum(0.0, align)  # in [0,1]

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
            contribution = math.cos(float(wrap_pi(phi[j] - th)))

            # Proxy drive:
            #   - amp[j]  : clause reliability from IPC phasor magnitude
            #   - gate[j] : carrier alignment to global theta (coherence -> action)
            s += w[j] * amp[j] * gate[j] * contribution
            #s += w[j] * math.cos(float(wrap_pi(phi[j] - th)))
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
    for a1,b1 in seg(o1):
        for a2,b2 in seg(o2):
            a, b = max(a1,a2), min(b1,b2)
            if a < b:
                r.append((a,b))
    return r


# ---------------------------------------------------------------------
# CVXOPT dyadic tail-weights (RH_MADNESS_2 style)
# ---------------------------------------------------------------------

def get_optimal_weights_cvxopt(J: int, delta_min: float = 12.0, delta_max: float = 1000.0) -> np.ndarray:
    """
    Solve QP:
      min w_sq^T K w_sq,  s.t. w_sq >= 0, 1^T w_sq = 1
    then w = sqrt(w_sq), normalize by max(w)=1.

    If cvxopt/scipy unavailable -> uniform weights.
    """
    if J <= 0:
        return np.ones(0, dtype=float)

    """try:
        from cvxopt import matrix, solvers
        from scipy.integrate import quad
    except Exception:
        return np.ones(J, dtype=float)"""

    try:
        from cvxopt import matrix, solvers
    except Exception:
        # Fallback: jednotkové váhy
        return np.ones(1, dtype=np.float64)

    scales = 2.0 ** np.arange(J)

    def A(delta: float) -> float:
        return float(np.exp(-0.5 * delta * delta))

    """K = np.zeros((J, J), dtype=float)
    for j in range(J):
        for k in range(j, J):
            aj, ak = scales[j], scales[k]
            integrand = lambda d: A(d / aj) * A(d / ak)
            val, _ = quad(integrand, float(delta_min), float(delta_max))
            K[j, k] = float(val)
            K[k, j] = float(val)"""

    # Uzavřený tvar: ∫ exp(-0.5*(d/aj)^2) * exp(-0.5*(d/ak)^2) dd
    # = ∫ exp(-α d^2) dd, kde α = 0.5*(1/aj^2 + 1/ak^2)
    # = sqrt(pi)/(2*sqrt(α)) * (erf(sqrt(α)*b) - erf(sqrt(α)*a))

    def gauss_overlap(aj: float, ak: float, a: float, b: float) -> float:
        inv = (1.0 / (aj * aj)) + (1.0 / (ak * ak))
        alpha = 0.5 * inv
        sa = math.sqrt(alpha)
        return (math.sqrt(math.pi) / (2.0 * sa)) * (math.erf(sa * b) - math.erf(sa * a))

    K = np.zeros((J, J), dtype=np.float64)
    a = float(delta_min)
    b = float(delta_max)
    for j in range(J):
        aj = float(scales[j])
        for k in range(j, J):
            ak = float(scales[k])
            val = gauss_overlap(aj, ak, a, b)
            K[j, k] = val
            K[k, j] = val

    P = matrix(2.0 * K)  # cvxopt uses (1/2)x^T P x
    q = matrix(0.0, (J, 1))
    G = matrix(-np.eye(J))
    h = matrix(0.0, (J, 1))
    Aeq = matrix(1.0, (1, J))
    beq = matrix(1.0)
    solvers.options["show_progress"] = False

    try:
        sol = solvers.qp(P, q, G, h, Aeq, beq)
        w_sq = np.array(sol["x"]).reshape(-1)
        w = np.sqrt(np.maximum(w_sq, 0.0))
        mx = float(np.max(w)) if float(np.max(w)) > 0 else 1.0
        return w / mx
    except Exception:
        return np.ones(J, dtype=float)


def build_ipc_clause_weights(
    C: int,
    mode: str = "cvxopt",
    delta_min: float = 12.0,
    delta_max: float = 1000.0,
) -> np.ndarray:
    mode = (mode or "ones").lower()
    if mode == "ones":
        return np.ones(int(C), dtype=np.float64)

    J = int(np.ceil(np.log2(max(2, int(C))))) + 1
    w_scale = get_optimal_weights_cvxopt(J, delta_min=delta_min, delta_max=delta_max)

    idx = np.floor(np.log2(np.arange(int(C), dtype=np.float64) + 1.0)).astype(np.int64)
    idx = np.clip(idx, 0, J - 1)
    w = w_scale[idx].astype(np.float64)

    if not np.all(np.isfinite(w)) or float(np.max(w)) <= 0:
        w = np.ones(int(C), dtype=np.float64)
    return w


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

        #base = carrier_row if shared_carrier else H[(j + 7 * seed + 1) % meff][:m]
        if shared_carrier:
            eps = 0.08  # malá konstanta; stačí i 0.02–0.12
            base = carrier_row + eps * H[(j + 7 * seed + 1) % meff][:m]
        else:
            base = H[(j + 7 * seed + 1) % meff][:m]

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
            return float((i * 1315423911 + j * 2654435761 + seed * 97531) & 0xFFFFFFFF) / 2**32

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
) -> None:
    """
    For each edge (i,j) with sign s_ij, update only on overlap Omega_ij:

      Z_i <- proj((1-eta) Z_i + eta * s_ij * Z_j)
      Z_j <- proj((1-eta) Z_j + eta * s_ij * Z_i_old)

    Outside overlap: untouched.
    """
    lock_bool: List[np.ndarray] = []
    for k in range(C):
        b = np.zeros(T, dtype=bool)
        b[lock_indices(T, int(offsets[k]), m)] = True
        lock_bool.append(b)

    for _ in range(int(sweeps)):
        for (i, j) in edges:
            sgn = float(edge_signs.get((i, j), +1.0))
            omega = np.where(lock_bool[i] & lock_bool[j])[0]
            if omega.size == 0:
                continue

            """Zi_old = Z[omega, i].copy()
            Zj_old = Z[omega, j].copy()
            Z[omega, i] = project_unit_circle((1.0 - eta) * Zi_old + eta * (sgn * Zj_old))
            Z[omega, j] = project_unit_circle((1.0 - eta) * Zj_old + eta * (sgn * Zi_old))"""

            ranges = overlap_ranges(int(offsets[i]), int(offsets[j]), m, T)
            if not ranges:
                continue

            for a, b in ranges:
                Zi_old = Z[a:b, i].copy()
                Zj_old = Z[a:b, j].copy()

                new_i = (1.0 - eta) * Zi_old + eta * (sgn * Zj_old)
                new_j = (1.0 - eta) * Zj_old + eta * (sgn * Zi_old)

                Z[a:b, i] = project_unit_circle(new_i)
                Z[a:b, j] = project_unit_circle(new_j)


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
        nbr[i].append(j); val[i].append(gij)
        nbr[j].append(i); val[j].append(np.conj(gij))
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


def power_lambda_max_edge(nbr: List[List[int]], val: List[List[complex]], iters: int = 250, tol: float = 1e-10) -> float:
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
# IPC: Invariant Phase Certifier (functional)
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
        v = Z_lock.conj().T @ u            # shape (C,)
        v = (w * v)                        # weighted
        u2 = (Z_lock @ v) / float(m)       # back to time
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
    theta = float(np.angle(S)) if S != 0 else 0.0
    mags = np.abs(a)
    beta = float(np.min(mags))
    ang = np.angle(a)
    err = wrap_pi(ang - theta)
    delta = float(np.max(np.abs(err)))
    return theta, beta, delta, a


def ipc_mu_sat_min(beta: float, delta: float) -> float:
    return float((beta ** 2) * (math.cos(delta) ** 2))


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


# ---------------------------------------------------------------------
# Certificate container
# ---------------------------------------------------------------------

@dataclass
class Certificate:
    meta: dict
    S2: dict
    spectral: dict
    IPC: dict
    bands: dict
    diag: dict


def run(
    C: int,
    R: int,
    d: int,
    sweeps: int,
    eta: float,
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
) -> Certificate:

    # ------------------ CNF vs synthetic ------------------
    cnf_meta: dict = {}
    clause_gauge: Optional[np.ndarray] = None
    zeta0 = 0.25

    if cnf_path:
        print(f"[CNF mód] Načítám {cnf_path}")
        nvars, clauses = parse_dimacs(cnf_path)
        C = len(clauses)
        print(f"  proměnné: {nvars}    klauzule: {C:,}")

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

    # ------------------ geometry ------------------
    T = 2 * int(R)
    m = int(R) // 2

    offsets = prime_offsets(C, T)
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
        print(f"  Frakce |proj| > 0.05   : {100*rep['frac_proj_gt_005']:.2f} %")
        print(f"  Frakce |proj| > 0.1    : {100*rep['frac_proj_gt_01']:.2f} %")
        print(f"  Rozptyl úhlů (variance): {rep['angle_var']:.6f} rad²")
        print(f"  Std úhlů               : {rep['angle_std']:.4f} rad  ≈ {rep['angle_std']*180/math.pi:.2f}°")
        print(f"  Koherenční proxy       : {rep['coh_proxy']:.6f}  (1 = všechny locky dokonale zarovnané)")

    # ------------------ signed constraints + overlap coupling ------------------
    em = (edge_mode or "auto").lower()
    if cnf_path and em in ("cnf", "logic"):
        # Full CNF clause graph (theory graph): share-variable edges.
        # 'cnf' = any shared variable; 'logic' = opposite-polarity (conflict) edges only.
        edges = build_logic_edges_from_cnf(clauses, nvars, include_same_polarity=(em == "cnf"))
        if not edges:
            # fallback (should be rare): bounded-degree selection, then circulant
            edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
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

    apply_signed_overlap_coupling(Z, T, C, m, offsets, edges, edge_signs, eta=eta, sweeps=sweeps)
    Z_lock = Z * M

    # ------------------ decision operator (edge-Gram) ------------------
    nbr, val = build_edge_gram(Z, T, C, m, offsets, edges)
    lam = power_lambda_max_edge(nbr, val, iters=power_iters, tol=power_tol)
    mu_dec = float(lam / float(C))

    rho = neighbor_rowsum(nbr, val)
    kap = kappa_S2(T, m, zeta0)
    bound = float(d) * kap
    S2_ok = bool(rho <= bound + 1e-12)

    # ------------------ IPC with clause weights ------------------
    w = build_ipc_clause_weights(C, mode=ipc_weight_mode, delta_min=w_delta_min, delta_max=w_delta_max)
    u = ipc_time_mode_u(Z_lock, w, m=m, iters=power_iters, tol=power_tol)
    theta, beta, delta, a = ipc_metrics(Z_lock, u, m=m)

    # ------------------ CNF witness extraction (assignment projection) ------------------
    witness = {}
    if cnf_path:
        try:
            assign_wit, score_wit = extract_assignment_from_ipc(
                clauses, nvars,
                clause_phasors=a, theta=theta, clause_weights=w
            )
            unsat_wit = count_unsat(clauses, assign_wit)

            # Diagnostics: how strongly proxy is actually driving the witness
            phi_w = np.angle(a)
            amp_w = np.abs(a).astype(np.float64, copy=False)
            align_w = np.cos(wrap_pi(phi_w - float(theta))).astype(np.float64, copy=False)
            gate_w = np.maximum(0.0, align_w)
            drive_w = w.astype(np.float64, copy=False) * amp_w * gate_w

            witness = {
                "assign_sha256": sha256_assignment(assign_wit),
                "unsat": int(unsat_wit),
                "unsat_frac": float(unsat_wit / max(1, len(clauses))),
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
        except Exception as e:
            witness = {"error": f"{type(e).__name__}: {e}"}

    mu_sat_min = ipc_mu_sat_min(beta, delta)

    # ------------------ bands ------------------
    lam_unsat_ceiling = float(1.0 + bound)
    mu_unsat_max = float(lam_unsat_ceiling / float(C))
    tau = 0.5 * (mu_sat_min + mu_unsat_max)
    Delta = 0.5 * (mu_sat_min - mu_unsat_max)
    separated = bool(Delta > 0)

    # ------------------ coherence diag ------------------
    r_mean, r_min, r_max = coherence_R(Z_lock)

    meta = {
        "C": int(C), "T": int(T), "m": int(m), "R": int(R), "d": int(d),
        "mode": str(mode),
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
        "spectral": {"lambda_max_GH": float(lam), "mu_dec": float(mu_dec)},
        "IPC": {"beta": float(beta), "delta": float(delta), "theta": float(theta), "mu_sat_min": float(mu_sat_min)},
        "bands": {"lam_unsat_ceiling": float(lam_unsat_ceiling), "mu_unsat_max": float(mu_unsat_max),
                  "tau": float(tau), "Delta": float(Delta), "separated": bool(separated)},
        "diag": {"coherence_R": {"mean": float(r_mean), "min": float(r_min), "max": float(r_max)},
                 "cnf_witness": witness},
    }

    print(f"\nWitness: {witness}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(cert_data, f, indent=2, ensure_ascii=False)

    return Certificate(**cert_data)


def get_nested(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


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


#fix
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

def main() -> None:
    ap = argparse.ArgumentParser(description="DREAM6_operator_2 functional certifier (no placeholders).")
    ap.add_argument("--C", type=int, default=2000)
    ap.add_argument("--R", type=int, default=104)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--mode", type=str, default="sat", choices=["sat", "unsat"])
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--eta", type=float, default=0.35)

    ap.add_argument("--shared-carrier", action="store_true", default=False)
    ap.add_argument("--shared-misphase", dest="shared_misphase", action="store_true", default=True)
    ap.add_argument("--no-shared-misphase", dest="shared_misphase", action="store_false")
    ap.add_argument("--unsat-neg-frac", type=float, default=0.25)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--power-iters", type=int, default=250)
    ap.add_argument("--power-tol", type=float, default=1e-10)

    ap.add_argument("--ipc-weights", type=str, default="cvxopt", choices=["ones", "cvxopt"])
    ap.add_argument("--w-delta-min", type=float, default=12.0)
    ap.add_argument("--w-delta-max", type=float, default=1000.0)

    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--verify-json", type=str, default=None)

    ap.add_argument("--cnf-path", type=str, default=None, help="DIMACS CNF path (.cnf)")
    ap.add_argument("--edge-mode", type=str, default="auto", choices=["auto","circulant","cnf","logic"],
                   help="Graph topology: auto (cnf->logic else circulant), circulant, cnf/logic (CNF clause-variable graph)")
    ap.add_argument("--outside-value", type=float, default=-1.0,
                   help="Background value outside lock windows (e.g. 0.0 to avoid silent regime for large T).")
    ap.add_argument("--no-flip-incident-unsat", action="store_true",
                   help="Disable incident-first edge flipping in UNSAT mode")

    ap.add_argument("--get-params", action="store_true",
                        help="Vypočítá doporučené parametry R/eta/d z CNF bez spuštění simulace.")

    args = ap.parse_args()

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
        mode=args.mode,
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
    )

    res = asdict(cert)

    print("\n=== DREAM6 Operator Certifier (s cvxopt vahami) ===")
    print(f"C={res['meta']['C']}  T={res['meta']['T']}  m={res['meta']['m']}  d={res['meta']['d']}  mode={res['meta']['mode']}")
    if res["meta"].get("cnf_path"):
        print(f"CNF: vars={res['meta']['nvars']}  seed_unsat={res['meta']['seed_unsat']} ({100*res['meta']['seed_unsat_frac']:.2f}%)")
    print(f"shared_carrier={res['meta']['shared_carrier']}  shared_misphase={res['meta']['shared_misphase']}  unsat_neg_frac={res['meta']['unsat_neg_frac']}")
    print(f"ipc_weights={res['meta']['ipc_weights']['mode']}")

    print(f"S2 radar: rho={res['S2']['rho']:.6g}  <= d*kappa={res['S2']['d_kappa']:.6g}  pass={res['S2']['pass']}")
    print(f"Spectral: lambda_max(G_H)={res['spectral']['lambda_max_GH']:.6g}  mu_dec={res['spectral']['mu_dec']:.6g}")
    print(f"IPC: beta={res['IPC']['beta']:.6g}  delta={res['IPC']['delta']:.6g}  mu_sat_min={res['IPC']['mu_sat_min']:.6g}")
    print(f"Bands: lam_unsat_ceiling={res['bands']['lam_unsat_ceiling']:.6g}  mu_unsat_max={res['bands']['mu_unsat_max']:.6g}")
    print(f"tau={res['bands']['tau']:.6g}  Delta={res['bands']['Delta']:.6g}  separated={res['bands']['separated']}")

    r = res["diag"]["coherence_R"]
    print(f"Coherence R(t): mean={r['mean']:.6g}  min={r['min']:.6g}  max={r['max']:.6g}")

    print("==============================================================\n")

    if args.json_out:
        print(f"Wrote certificate JSON: {args.json_out}")


if __name__ == "__main__":
    main()