#!/usr/bin/env python3
# DREAM_FINAL.py
# Spectral seed -> micro-polish -> full polish -> core-shake finisher -> verified SAT model output.
#
# Usage:
#   python DREAM_FINAL.py --cnf uf250-0100.cnf
#   python DREAM_FINAL.py --cnf random_3sat_10000.cnf --preset random10000
#
# Notes:
# - "Verified SAT" here means UNSAT=0 by direct clause check (no DRAT proof).
# - Designed to be stable on UF250 and scalable on random_3sat_10000.
#
import argparse, math, os, random, time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# ---------------- DIMACS ----------------

def parse_dimacs(path: str) -> Tuple[int, List[List[int]]]:
    clauses: List[List[int]] = []
    nvars = 0
    cur: List[int] = []
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
            for t in s.split():
                v = int(t)
                if v == 0:
                    if cur:
                        clauses.append(cur)
                        cur = []
                else:
                    cur.append(v)
                    nvars = max(nvars, abs(v))
    if cur:
        clauses.append(cur)
    return nvars, clauses


# ---------------- SAT eval ----------------

def clause_satisfied(clause: List[int], assign01: List[int]) -> bool:
    for lit in clause:
        v = abs(lit) - 1
        val = bool(assign01[v])
        if lit < 0:
            val = not val
        if val:
            return True
    return False

def count_unsat(clauses: List[List[int]], assign01: List[int]) -> int:
    u = 0
    for cl in clauses:
        if not clause_satisfied(cl, assign01):
            u += 1
    return u

def print_dimacs_model(assign01: List[int]):
    out = []
    for i, b in enumerate(assign01, start=1):
        out.append(i if b == 1 else -i)
    print("v " + " ".join(map(str, out)) + " 0")


# ======================================================================
# SCHEDULE + PRINCIPAL WEIGHT (spectral engine)
# ======================================================================

def wiring_neighbors_circulant(C: int, d: int = 6):
    nbrs = []
    for i in range(C):
        nbr_set = set()
        for step in range(1, d // 2 + 1):
            nbr_set.add((i - step) % C)
            nbr_set.add((i + step) % C)
        nbrs.append(nbr_set)
    return nbrs

def gcd_coprime_stride_near_half(T: int) -> int:
    s = max(1, T // 2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            return 1
    return s

def schedule_unsat_hadamard(C: int, R: int, rho: float, zeta0: float, L: int, sigma_up: float,
                           seed: int, couple: bool, neighbor_atten: float, d: int):
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(math.floor(rho * T))
    m = max(1, min(m, T))
    k = int(math.floor(zeta0 * m))
    k = max(0, min(k, m))

    s = gcd_coprime_stride_near_half(T)
    offsets = [(j * s) % T for j in range(C)]
    lock_idx = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]
    Phi = np.full((T, C), np.pi, dtype=float)

    Hlen = 1
    while Hlen < m:
        Hlen <<= 1

    def hadamard_sign(row: int, col: int) -> float:
        return 1.0 if (bin(row & col).count("1") % 2 == 0) else -1.0

    row_step = (Hlen // 2) + 1
    if math.gcd(row_step, Hlen) != 1:
        row_step |= 1
        while math.gcd(row_step, Hlen) != 1:
            row_step += 2

    g = (Hlen // 3) | 1
    while math.gcd(g, Hlen) != 1:
        g += 2

    cols = np.mod(g * np.arange(m, dtype=int), Hlen)

    for j in range(C):
        row = (j * row_step) % Hlen
        neg_idx = []
        for t in range(m):
            if hadamard_sign(row, int(cols[t])) < 0:
                neg_idx.append(t)
        neg_idx = np.array(neg_idx, dtype=int)

        if len(neg_idx) >= k:
            mask_pi = rng.choice(neg_idx, size=k, replace=False) if k > 0 else np.empty(0, dtype=int)
        else:
            pool = np.setdiff1d(np.arange(m, dtype=int), neg_idx, assume_unique=False)
            extra = rng.choice(pool, size=max(0, k - len(neg_idx)), replace=False) if k > len(neg_idx) else np.empty(0, dtype=int)
            mask_pi = np.concatenate([neg_idx, extra]) if len(extra) else neg_idx

        mask_0 = np.setdiff1d(np.arange(m, dtype=int), mask_pi, assume_unique=False)
        slots = lock_idx[j]
        if len(mask_pi) > 0:
            Phi[slots[mask_pi], j] = np.pi
        if len(mask_0) > 0:
            Phi[slots[mask_0], j] = rng.normal(loc=0.0, scale=sigma_up, size=len(mask_0))

    if couple and (abs(neighbor_atten - 1.0) > 1e-12):
        neighbors = wiring_neighbors_circulant(C, d=d)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        kappa = (1.0 - 2.0*zeta0)**2 + (2.0/max(1,m)) + (1.0/max(1,T))
        kappa = max(0.0, min(1.0, kappa))
        for j in range(C):
            Lj = lock_sets[j]
            for j_adj in neighbors[j]:
                if j_adj == j:
                    continue
                overlap = Lj.intersection(lock_sets[j_adj])
                if not overlap:
                    continue
                overlap_size = len(overlap)
                overlap_fraction = overlap_size / max(1, m)
                cross_term_weight = min(
                    1.0,
                    (len(neighbors[j]) * kappa) / max(1.0, C * (1 - 0.5*sigma_up)**2)
                    * (1.0 + 3.0 * overlap_fraction)
                )
                attenuation = max(
                    0.70,
                    neighbor_atten - 0.05 * overlap_size / (m * (1 + 0.25 * math.sqrt(max(1e-9, math.log(C))) * overlap_fraction))
                    * (1 - cross_term_weight)
                )
                idx = np.fromiter(overlap, dtype=int)
                Phi[idx, j_adj] *= attenuation
    return Phi

def principal_weight_power(Phi: np.ndarray, iters: int = 120) -> np.ndarray:
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.normal(size=C) + 1j * rng.normal(size=C)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        y = Z @ x
        x = (Z.conj().T @ y) / T
        x /= (np.linalg.norm(x) + 1e-12)
    w = np.abs(x)
    w = w / (w.mean() + 1e-12)
    return np.clip(w, 0.1, 10.0)

def seed_from_clause_weights(clauses: List[List[int]], nvars: int,
                             w_clause: np.ndarray,
                             score_norm_alpha: float,
                             bias_weight: float,
                             seed: int) -> List[int]:
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    pol = np.zeros(nvars + 1, dtype=int)
    deg = np.zeros(nvars + 1, dtype=int)

    for ci, cl in enumerate(clauses):
        for LIT in cl:
            v = abs(LIT)
            deg[v] += 1
            if LIT > 0:
                pos[v].append(ci); pol[v] += 1
            else:
                neg[v].append(ci); pol[v] -= 1

    score = np.zeros(nvars + 1, dtype=float)
    for v in range(1, nvars + 1):
        if pos[v]:
            score[v] += float(w_clause[pos[v]].sum())
        if neg[v]:
            score[v] -= float(w_clause[neg[v]].sum())
        if score_norm_alpha > 0.0:
            score[v] /= (float(deg[v]) ** score_norm_alpha + 1e-12)
        if bias_weight != 0.0:
            score[v] += bias_weight * float(pol[v]) / max(1, int(deg[v]))

    rng = np.random.default_rng(seed + 1337)
    dither = rng.uniform(-1e-7, 1e-7, size=nvars + 1)
    return [1 if (score[v] + dither[v]) >= 0.0 else 0 for v in range(1, nvars + 1)]


# ======================================================================
# Local search polish (the workhorse)
# ======================================================================

def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int,
    seed: int,
    alpha: float = 2.4,
    beta: float = 0.9,
    epsilon: float = 1e-3,
    probsat_quota: int = 4000,
) -> List[int]:
    rnd = random.Random(seed)
    nvars = len(assign01)
    C = len(clauses)

    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for L in cl:
            (pos if L > 0 else neg)[abs(L)].append(ci)

    assign = [False] + [bool(b) for b in assign01]
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list: List[int] = []

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

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

    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:
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

    def pick_unsat_clause() -> Optional[int]:
        if not unsat_list:
            return None
        i = rnd.randrange(len(unsat_list))
        for _ in range(3):
            ci = unsat_list[i]
            if in_unsat[ci]:
                return ci
            i = rnd.randrange(len(unsat_list))
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    def cur_unsat() -> int:
        return sum(1 for ci in unsat_list if in_unsat[ci])

    best_assign = assign[:]
    best_uns = cur_unsat()

    steps = 0
    while steps < flips:
        if best_uns == 0:
            return [1 if b else 0 for b in best_assign[1:]]
        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]

        freebies = []
        cand = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            mk = makecount(v)
            if bc == 0:
                freebies.append((mk, v))
            cand.append((bc, -mk, v))

        if freebies:
            freebies.sort(reverse=True)
            v_choice = freebies[0][1]
        else:
            v_choice = min(cand)[2]

        flip_var(v_choice)
        steps += 1

        u = cur_unsat()
        if u < best_uns:
            best_uns = u
            best_assign = assign[:]
        if u == 0:
            return [1 if b else 0 for b in assign[1:]]

        if steps % 2000 == 0 and u >= best_uns:
            for _ in range(min(probsat_quota, flips - steps)):
                ci2 = pick_unsat_clause()
                if ci2 is None:
                    return [1 if b else 0 for b in assign[1:]]
                clause2 = clauses[ci2]
                tot = 0.0
                scores = []
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

    return [1 if b else 0 for b in best_assign[1:]]


# ======================================================================
# Finisher: surgical core shake (low collateral) + re-polish
# ======================================================================

def surgical_core_shake(clauses: List[List[int]], assign01: List[int], k_flips: int, rng: random.Random) -> List[int]:
    # collect UNSAT clauses
    bad = []
    for ci, cl in enumerate(clauses):
        if not clause_satisfied(cl, assign01):
            bad.append(ci)
    if not bad or k_flips <= 0:
        return assign01[:]

    # core var frequency
    freq = {}
    for ci in bad:
        for lit in clauses[ci]:
            v = abs(lit) - 1
            freq[v] = freq.get(v, 0) + 1
    core_vars = list(freq.keys())
    if not core_vars:
        return assign01[:]

    # sat_count (for breakcount on critical clauses)
    sat_count = [0] * len(clauses)
    for ci, cl in enumerate(clauses):
        cnt = 0
        for lit in cl:
            v = abs(lit) - 1
            val = bool(assign01[v])
            if lit < 0:
                val = not val
            if val:
                cnt += 1
        sat_count[ci] = cnt

    # occurrences in critical clauses only
    occ_pos = {v: [] for v in core_vars}
    occ_neg = {v: [] for v in core_vars}
    for ci, cl in enumerate(clauses):
        if sat_count[ci] != 1:
            continue
        for lit in cl:
            v = abs(lit) - 1
            if v in freq:
                (occ_pos[v] if lit > 0 else occ_neg[v]).append(ci)

    def breakcount(v: int) -> int:
        # if v currently True, it satisfies positive lit; flip may break those critical clauses
        if assign01[v] == 1:
            return len(occ_pos[v])
        else:
            return len(occ_neg[v])

    # build candidate pool: prefer high freq, low breakcount
    pool_size = min(len(core_vars), max(300, 25 * k_flips))
    pool = rng.sample(core_vars, pool_size) if len(core_vars) > pool_size else core_vars

    scored = []
    for v in pool:
        bc = breakcount(v)
        f = freq[v]
        scored.append((bc, -f, v))
    scored.sort()
    top = scored[:max(60, 6 * k_flips)]
    if not top:
        return assign01[:]

    # weighted pick: exp(-lambda*bc)*(1+freq)
    lam = 0.7
    weights = []
    for bc, nf, v in top:
        f = -nf
        w = math.exp(-lam * bc) * (1.0 + f)
        weights.append(w)
    total = sum(weights) + 1e-12

    out = assign01[:]
    for _ in range(k_flips):
        r = rng.random() * total
        acc = 0.0
        pick = top[-1][2]
        for (bc, nf, v), w in zip(top, weights):
            acc += w
            if acc >= r:
                pick = v
                break
        out[pick] ^= 1
    return out


# ======================================================================
# Presets (safe defaults)
# ======================================================================

def get_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "uf250": {
            "base_sigma": 0.020,
            "rho_base": 0.734296875,
            "neighbor_atten": 0.9495,
            "d": 6,
            "cR": 12.0,
            "L": 4,
            "sigma_scale": 0.50,     # sigma=0.01000 worked well in your UF autotune
            "rho_nudge": -0.030,
            "zeta0": 0.44,
            "couple": 0,
            "score_norm_alpha": 0.40,
            "bias_weight": 0.0,
            "power_iters": 80,
            "micro_polish": 200_000,
            "full_polish": 2_000_000,      # UF usually needs far less than 12M
            "fin_rounds": 6,
            "fin_k_flips": 6,              # tiny core shakes
            "fin_polish": 300_000,
        },
        "random10000": {
            "base_sigma": 0.020,
            "rho_base": 0.734296875,  # DREAM6
            "neighbor_atten": 0.9495,
            "d": 6,

            "cR": 10.0,  # DREAM6
            "L": 3,  # DREAM6

            # !!! tady to nejdůležitější: sigma přímo, ne base_sigma*scale
            # v solve_one pak udělej: sigma = P["sigma_up_direct"] pokud existuje
            "sigma_up_direct": 0.045,  # DREAM6

            "rho_nudge": 0.0,  # držíme rho_base
            "zeta0": 0.40,  # DREAM6
            "couple": 1,  # DREAM6

            "score_norm_alpha": 0.50,  # DREAM6
            "bias_weight": 0.10,  # DREAM6

            "power_iters": 60,  # DREAM6 (speed!)

            "micro_polish": 120_000,
            "full_polish": 12_000_000,
            "fin_rounds": 18,
            "fin_k_flips": 0,  # už nebude použito (walksat finisher)
            "fin_polish": 2_500_000,
        }

    }


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def walksat_shake(clauses, assign01, steps, seed, noise=0.18):
    """
    Mini WalkSAT shake:
      - repeat 'steps' times:
        pick random UNSAT clause
        with prob 'noise': flip random var in it
        else: flip var with minimum breakcount (ties random)
    """
    rnd = random.Random(seed)
    nvars = len(assign01)

    # build occ lists for breakcount
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            (pos if lit > 0 else neg)[abs(lit)].append(ci)

    assign = [False] + [bool(b) for b in assign01]
    C = len(clauses)
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list = []

    def add_unsat(ci):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci):
        if in_unsat[ci]:
            in_unsat[ci] = False

    for ci, cl in enumerate(clauses):
        cnt = 0
        for lit in cl:
            v = abs(lit)
            val = assign[v]
            if lit < 0:
                val = not val
            if val:
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            add_unsat(ci)

    def breakcount(v):
        bc = 0
        if assign[v]:
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:
            for ci in neg[v]:
                if sat_count[ci] == 1:
                    bc += 1
        return bc

    def flip_var(v):
        old = assign[v]
        assign[v] = not old
        if old:
            for ci in pos[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0: add_unsat(ci)
            for ci in neg[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0: drop_unsat(ci)
        else:
            for ci in neg[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0: add_unsat(ci)
            for ci in pos[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0: drop_unsat(ci)

    def pick_unsat():
        if not unsat_list:
            return None
        for _ in range(6):
            ci = unsat_list[rnd.randrange(len(unsat_list))]
            if in_unsat[ci]:
                return ci
        # compact
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    for _ in range(steps):
        ci = pick_unsat()
        if ci is None:
            break
        cl = clauses[ci]
        if rnd.random() < noise:
            lit = rnd.choice(cl)
            flip_var(abs(lit))
        else:
            # greedy low-break
            best_bc = 10**9
            best_vars = []
            for lit in cl:
                v = abs(lit)
                bc = breakcount(v)
                if bc < best_bc:
                    best_bc = bc
                    best_vars = [v]
                elif bc == best_bc:
                    best_vars.append(v)
            flip_var(rnd.choice(best_vars))

    return [1 if b else 0 for b in assign[1:]]


def solve_one(cnf_path: str, preset: str, seed: int,
              micro_polish_override: int = 0,
              full_polish_override: int = 0,
              fin_rounds_override: int = 0) -> Tuple[bool, List[int]]:
    nvars, clauses = parse_dimacs(cnf_path)
    C = len(clauses)
    P = get_presets()[preset]

    base_sigma = float(P["base_sigma"])
    rho_base  = float(P["rho_base"])
    neighbor_atten = float(P["neighbor_atten"])
    d = int(P["d"])

    cR = float(P["cR"])
    L  = int(P["L"])
    R  = math.ceil(cR * math.log(max(2, C)))

    #sigma = base_sigma * float(P["sigma_scale"])

    if "sigma_up_direct" in P:
        sigma = float(P["sigma_up_direct"])
    else:
        sigma = base_sigma * float(P["sigma_scale"])

    rho = max(0.10, min(0.98, rho_base + float(P["rho_nudge"])))
    zeta0 = float(P["zeta0"])
    couple = bool(int(P["couple"]))
    score_norm_alpha = float(P["score_norm_alpha"])
    bias_weight = float(P["bias_weight"])
    power_iters = int(P["power_iters"])

    micro_polish = int(micro_polish_override) if micro_polish_override > 0 else int(P["micro_polish"])
    full_polish  = int(full_polish_override) if full_polish_override > 0 else int(P["full_polish"])
    fin_rounds   = int(fin_rounds_override) if fin_rounds_override > 0 else int(P["fin_rounds"])
    fin_k_flips  = int(P["fin_k_flips"])
    fin_polish   = int(P["fin_polish"])

    print(f"File: {os.path.basename(cnf_path)}  vars={nvars} clauses={C}")
    print(f"[preset] {preset}  R={R} T={R*L}  sigma={sigma:.5f} rho={rho:.3f} zeta={zeta0:.3f} couple={int(couple)} alpha={score_norm_alpha:.2f} bias={bias_weight:.2f}")
    print(f"[budget] micro={micro_polish:,} full={full_polish:,} fin_rounds={fin_rounds} fin_k={fin_k_flips} fin_polish={fin_polish:,}")

    t0 = time.time()

    # 1) spectral seed
    Phi = schedule_unsat_hadamard(
        C=C, R=R, rho=rho, zeta0=zeta0, L=L, sigma_up=sigma,
        seed=seed + 111,
        couple=couple,
        neighbor_atten=neighbor_atten,
        d=d
    )
    w = principal_weight_power(Phi, iters=power_iters)
    a0 = seed_from_clause_weights(
        clauses, nvars, w,
        score_norm_alpha=score_norm_alpha,
        bias_weight=bias_weight,
        seed=seed + 222
    )
    u0 = count_unsat(clauses, a0)
    print(f"[seed] unsat={u0}  time={time.time()-t0:.2f}s")

    # 2) micro-polish
    a1 = greedy_polish(clauses, a0, flips=micro_polish, seed=seed + 333)
    u1 = count_unsat(clauses, a1)
    print(f"[micro] flips={micro_polish:,} unsat={u1}  time={time.time()-t0:.2f}s")
    if u1 == 0:
        return True, a1

    # 2b) residual reweighting loop (DREAM19)
    # goal: reduce micro_unsat BEFORE heavy polish
    refine_rounds = 6 if preset == "random10000" else 2
    lam = 0.32 if preset == "random10000" else 0.25

    w_ref = w.copy()
    cur = a1
    cur_u = u1

    for rr in range(1, refine_rounds + 1):
        w_ref, ucount = refine_clause_weights_from_unsat(clauses, cur, w_ref, lam=lam, mu=0.0)

        # reseed from refined weights (small change, big effect)
        cur2 = seed_from_clause_weights(
            clauses, nvars, w_ref,
            score_norm_alpha=score_norm_alpha,
            bias_weight=bias_weight,
            seed=seed + 900 + rr
        )

        # short micro-polish to test basin
        cur2 = greedy_polish(clauses, cur2, flips=max(30_000, micro_polish//4), seed=seed + 1200 + rr)

        u2b = count_unsat(clauses, cur2)
        if u2b < cur_u:
            cur, cur_u = cur2, u2b

        print(f"[refine r{rr:02d}] unsat_core={ucount} -> micro_unsat={u2b} best={cur_u}  time={time.time()-t0:.2f}s")

        if cur_u == 0:
            return True, cur

    # continue with best refined state
    a1 = cur
    u1 = cur_u
    print(f"[refine] BEST micro_unsat={u1}  time={time.time()-t0:.2f}s")


    # 3) full polish
    a2 = greedy_polish(clauses, a1, flips=full_polish, seed=seed + 444)
    u2 = count_unsat(clauses, a2)
    print(f"[polish] flips={full_polish:,} unsat={u2}  time={time.time()-t0:.2f}s")
    best = (u2, a2)

    if u2 == 0:
        return True, a2

    # 4) finisher: core-shake + re-polish
    """rng = random.Random(seed ^ 0xC0DEC0DE)
    cur = a2
    for r in range(1, fin_rounds + 1):
        k = fin_k_flips
        # adaptive: if we're close, reduce k
        cur_u = count_unsat(clauses, cur)
        if cur_u <= 50:
            k = max(10, fin_k_flips // 2)
        if cur_u <= 15:
            k = max(6, fin_k_flips // 4)

        trial = surgical_core_shake(clauses, cur, k_flips=k, rng=rng)
        trial = greedy_polish(clauses, trial, flips=fin_polish, seed=seed + 5000 + 13*r)
        ut = count_unsat(clauses, trial)

        if ut < best[0]:
            best = (ut, trial)
            cur = trial

        print(f"[finisher r{r:02d}] k={k:3d} -> unsat={ut} best={best[0]}  time={time.time()-t0:.2f}s")
        if best[0] == 0:
            return True, best[1]"""

    # 4) finisher: multi-try WalkSAT shake + short polish
    rng = random.Random(seed ^ 0xC0DEC0DE)
    cur = best[1]

    for r in range(1, fin_rounds + 1):
        cur_u = count_unsat(clauses, cur)

        # adaptive shake length & noise near the end
        if cur_u > 120:
            shake_steps = 30_000
            noise = 0.22
            tries = 2
        elif cur_u > 40:
            shake_steps = 45_000
            noise = 0.20
            tries = 3
        else:
            shake_steps = 70_000
            noise = 0.18
            tries = 4

        local_best = (cur_u, cur)

        for t in range(tries):
            sseed = (seed + 9000*r + 101*t)
            trial = walksat_shake(clauses, cur, steps=shake_steps, seed=sseed, noise=noise)

            # shorter polish per try (more tries > one huge)
            trial = greedy_polish(clauses, trial, flips=fin_polish, seed=seed + 5000 + 97*r + t)

            ut = count_unsat(clauses, trial)
            if ut < local_best[0]:
                local_best = (ut, trial)

        if local_best[0] < best[0]:
            best = local_best
            cur = local_best[1]

        print(f"[finisher r{r:02d}] cur={cur_u} -> best_try={local_best[0]} global_best={best[0]} "
              f"(steps={shake_steps:,} noise={noise:.2f} tries={tries})  time={time.time()-t0:.2f}s")

        if best[0] == 0:
            return True, best[1]


    return (best[0] == 0), best[1]


def refine_clause_weights_from_unsat(clauses, assign01, w, lam=0.35, mu=0.00):
    # w: numpy array length = #clauses
    unsat_mask = np.zeros(len(clauses), dtype=bool)
    for i, cl in enumerate(clauses):
        if not clause_satisfied(cl, assign01):
            unsat_mask[i] = True
    ww = w.copy()
    ww[unsat_mask] *= math.exp(lam)
    if mu != 0.0:
        ww[~unsat_mask] *= math.exp(-mu)
    ww /= (ww.mean() + 1e-12)
    return np.clip(ww, 0.05, 20.0), int(unsat_mask.sum())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--preset", default="", choices=["", "uf250", "random10000"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro_polish", type=int, default=0)
    ap.add_argument("--full_polish", type=int, default=0)
    ap.add_argument("--fin_rounds", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    # auto-preset if not specified
    preset = args.preset
    if preset == "":
        name = os.path.basename(args.cnf).lower()
        preset = "random10000" if ("random" in name or "3sat" in name or "10000" in name) else "uf250"

    print("\n=== DREAM_FINAL ===")
    ok, model = solve_one(
        cnf_path=args.cnf,
        preset=preset,
        seed=args.seed,
        micro_polish_override=args.micro_polish,
        full_polish_override=args.full_polish,
        fin_rounds_override=args.fin_rounds
    )

    # final verify + output
    # (model can be best-so-far if not SAT)
    # always print final UNSAT and (if SAT) model
    # so you can diff/compare runs.
    nvars, clauses = parse_dimacs(args.cnf)
    u = count_unsat(clauses, model)
    print("\n=== RESULT ===")
    print(f"UNSAT: {u} / {len(clauses)}  ({(100.0*u/max(1,len(clauses))):.3f}%)")
    print(f"Verified SAT: {u == 0}")
    if u == 0:
        print_dimacs_model(model)

if __name__ == "__main__":
    main()
