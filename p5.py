# unity_prime_hunt.py
# ------------------------------------------------------------
# UNITY PRIME HUNT (bigint-ready) + Truth/Witness operator
# - Finds a probable prime just below 10^k (k up to 10000+)
# - Uses:
#    1) hard constraints (anti-echo / anti-trivial factors)
#    2) wheel stepping (2*3*5*7 = 210) for speed
#    3) BPSW primality test (MR base 2 + strong Lucas)
#    4) optional extra Miller–Rabin rounds (confidence boost)
# - Produces a "witness" log for auditability (UNITY)
# ------------------------------------------------------------

from __future__ import annotations
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(20000)   # stačí pro 10^10000
    # nebo sys.set_int_max_str_digits(0)  # úplně bez limitu (méně bezpečné)



# =========================
# 0) Small primes (filters)
# =========================

LOG10_2 = 0.3010299956639812

def decimal_digits(n: int) -> int:
    if n == 0:
        return 1
    return int(n.bit_length() * LOG10_2) + 1

def head_decimal(n: int, m: int = 80) -> str:
    d = decimal_digits(n)
    if d <= m:
        return str(n)  # malé číslo, safe
    return str(n // 10**(d - m))  # jen prvních m číslic, safe


def sieve_primes(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    r = int(limit ** 0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = b"\x00" * ((limit - start) // step + 1)
    return [i for i in range(limit + 1) if sieve[i]]


# =========================
# 1) Jacobi symbol
# =========================

def jacobi(a: int, n: int) -> int:
    # (a/n), n odd positive
    if n <= 0 or (n & 1) == 0:
        raise ValueError("jacobi: n must be positive odd")
    a %= n
    s = 1
    while a:
        while (a & 1) == 0:
            a >>= 1
            r = n & 7
            if r == 3 or r == 5:
                s = -s
        a, n = n, a
        if (a & 3) == 3 and (n & 3) == 3:
            s = -s
        a %= n
    return s if n == 1 else 0


# =========================
# 2) Miller–Rabin
# =========================

def _mr_decompose(n: int) -> Tuple[int, int]:
    # n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    return d, s

def is_strong_prp(n: int, a: int) -> bool:
    if n < 2:
        return False
    a %= n
    if a == 0:
        return True
    d, s = _mr_decompose(n)
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False

def is_prime_u64_det(n: int) -> bool:
    # Deterministic for n < 2^64
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = _mr_decompose(n)

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        if a % n == 0:
            return True
        if not check(a):
            return False
    return True


# =========================
# 3) Strong Lucas (Selfridge)
# =========================

def _selfridge_params(n: int) -> Optional[Tuple[int, int, int]]:
    # Find D such that Jacobi(D/n) = -1, with P=1, Q=(1-D)/4
    D = 5
    while True:
        j = jacobi(D, n)
        if j == -1:
            P = 1
            Q = (1 - D) // 4
            return D, P, Q
        if j == 0:
            return None  # gcd(D, n) > 1 => composite
        D = -D - 2 if D > 0 else -D + 2
        if abs(D) > 100000:
            return None

def _lucas_uv(n: int, P: int, Q: int, k: int) -> Tuple[int, int]:
    # Binary method for Lucas sequences mod n
    # Returns (U_k, V_k)
    if k == 0:
        return 0, 2 % n
    inv2 = (n + 1) // 2  # inverse of 2 mod n (n odd)
    D = (P * P - 4 * Q) % n

    U = 1
    V = P % n
    Qk = Q % n

    # iterate bits of k excluding top bit
    bits = bin(k)[3:]
    for b in bits:
        # Doubling
        U2 = (U * V) % n
        V2 = (V * V - 2 * Qk) % n
        Qk2 = (Qk * Qk) % n

        if b == "0":
            U, V, Qk = U2, V2, Qk2
        else:
            # Addition step: (2m)->(2m+1)
            U = (U2 * P + V2) % n
            U = (U * inv2) % n
            V = (V2 * P + U2 * D) % n
            V = (V * inv2) % n
            Qk = (Qk2 * Q) % n

    return U % n, V % n

def is_strong_lucas_prp(n: int) -> bool:
    if n < 2:
        return False
    if (n & 1) == 0:
        return n == 2
    params = _selfridge_params(n)
    if params is None:
        return False
    _, P, Q = params

    # n+1 = d * 2^s
    d = n + 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1

    Ud, Vd = _lucas_uv(n, P, Q, d)
    if Ud == 0 or Vd == 0:
        return True

    # Repeated squaring on V
    Qd = pow(Q, d, n)
    V = Vd
    for _ in range(1, s):
        V = (V * V - 2 * Qd) % n
        Qd = (Qd * Qd) % n
        if V == 0:
            return True
    return False


# =========================
# 4) BPSW + extra MR rounds
# =========================

def is_bpsw_prime(n: int, small_primes: List[int]) -> bool:
    if n < 2:
        return False
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    r = int(math.isqrt(n))
    if r * r == n:
        return False
    if not is_strong_prp(n, 2):
        return False
    return is_strong_lucas_prp(n)

def is_probable_prime_unity(n: int,
                           small_primes: List[int],
                           extra_mr_bases: List[int]) -> Tuple[bool, Dict[str, object]]:
    """
    UNITY operator:
      - hard constraints: small prime division + square check
      - truth filter: BPSW
      - optional: extra MR bases to crank confidence
    Returns (ok, witness)
    """
    w: Dict[str, object] = {}
    #w["n_digits"] = len(str(n))
    w["n_digits"] = decimal_digits(n)
    w["bit_length"] = n.bit_length()

    #w["n_digits"] = int(n.bit_length() * 0.3010299956639812) + 1

    w["mod_small"] = {}

    # small prime division witness
    for p in small_primes[:64]:
        w["mod_small"][p] = int(n % p)
        if n == p:
            return True, {**w, "result": "prime (small)"}
        if n % p == 0:
            return False, {**w, "result": f"composite (divisible by {p})"}

    # square check
    r = int(math.isqrt(n))
    if r * r == n:
        return False, {**w, "result": "composite (square)"}

    # BPSW
    ok_bpsw = is_bpsw_prime(n, small_primes)
    w["bpsw"] = bool(ok_bpsw)
    if not ok_bpsw:
        return False, {**w, "result": "composite (BPSW failed)"}

    # extra MR bases (optional)
    mr_passed = []
    for a in extra_mr_bases:
        mr_ok = is_strong_prp(n, a)
        mr_passed.append((a, bool(mr_ok)))
        if not mr_ok:
            return False, {**w, "mr_extra": mr_passed, "result": f"composite (MR failed base {a})"}

    w["mr_extra"] = mr_passed
    return True, {**w, "result": "probable prime (BPSW + extra MR)"}


# =========================
# 5) Wheel stepping (210)
# =========================

WHEEL = 210
# residues coprime to 2,3,5,7 in [0..209]
WHEEL_RES = [r for r in range(WHEEL) if math.gcd(r, WHEEL) == 1]

def prev_wheel_candidate(start: int) -> int:
    """
    Return the largest integer <= start that is odd and coprime with 2,3,5,7.
    """
    n = start
    if (n & 1) == 0:
        n -= 1
    while math.gcd(n, WHEEL) != 1:
        n -= 2
    return n

def wheel_step_down(n: int) -> int:
    """
    Step down to previous residue class coprime to 2,3,5,7.
    """
    r = n % WHEEL
    idx = WHEEL_RES.index(r)
    if idx > 0:
        r2 = WHEEL_RES[idx - 1]
        return n - (r - r2)
    else:
        # wrap to previous wheel block
        r2 = WHEEL_RES[-1]
        return n - (r + (WHEEL - r2))


# =========================
# 6) UNITY metrics + branch selection
# =========================

@dataclass
class BranchResult:
    name: str
    found: bool
    n: Optional[int]
    steps: int
    tests: int
    rejects: int
    runtime_s: float
    witness: Optional[Dict[str, object]]
    # "energy" functional: lower is better (UNITY choice)
    drift: float
    sat: float
    E: float
    confidence: float

def estimate_mr_error_bound(k: int) -> float:
    # For Miller–Rabin with k independent bases:
    # error <= 4^{-k}. (This is a standard worst-case bound for MR on odd composite.)
    return 4.0 ** (-k)

def unity_energy(drift: float, sat: float, steps: int) -> float:
    """
    Our current 'E' (stability cost):
      - drift: rejected fraction (noise)
      - sat: 1/(1+steps) (how quickly the channel resolves)
      - small pulse penalty: log(1+steps) scaled
    """
    pulse_pen = 0.02 * math.log1p(steps)
    return drift + sat + pulse_pen

def branch_alt_wheel_find_below_10k(k: int,
                                   *,
                                   max_steps: int,
                                   small_prime_limit: int,
                                   extra_mr_bases: List[int]) -> BranchResult:
    t0 = time.time()
    small_primes = sieve_primes(small_prime_limit)

    # Start at 10^k - 1, but align to wheel
    start = pow(10, k) - 1
    n = prev_wheel_candidate(start)

    rejects = 0
    tests = 0
    witness_last = None

    for step in range(1, max_steps + 1):
        # quick reject with tiny set (fast)
        bad = False
        for p in small_primes[:256]:
            if n % p == 0 and n != p:
                bad = True
                break
        if bad:
            rejects += 1
            n = wheel_step_down(n)
            continue

        tests += 1
        ok, w = is_probable_prime_unity(n, small_primes, extra_mr_bases)
        witness_last = w
        if ok:
            runtime = time.time() - t0
            drift = rejects / max(1, step)
            sat = 1.0 / (1.0 + step)
            # confidence: BPSW + extra MR (bound from extra MR only)
            # (BPSW has no known counterexample; MR bound gives a conservative numeric.)
            mr_k = len(extra_mr_bases) + 1  # +1 for base-2 MR inside BPSW
            conf = 1.0 - estimate_mr_error_bound(mr_k)
            E = unity_energy(drift, sat, step)
            return BranchResult(
                name="ALT_WHEEL_BPSW",
                found=True, n=n,
                steps=step, tests=tests, rejects=rejects,
                runtime_s=runtime,
                witness=witness_last,
                drift=drift, sat=sat, E=E,
                confidence=conf
            )

        n = wheel_step_down(n)

    runtime = time.time() - t0
    drift = rejects / max(1, max_steps)
    sat = 0.0
    E = unity_energy(drift, sat, max_steps)
    mr_k = len(extra_mr_bases) + 1
    conf = 1.0 - estimate_mr_error_bound(mr_k)
    return BranchResult(
        name="ALT_WHEEL_BPSW",
        found=False, n=None,
        steps=max_steps, tests=tests, rejects=rejects,
        runtime_s=runtime,
        witness=witness_last,
        drift=drift, sat=sat, E=E,
        confidence=conf
    )

def tail_decimal(n: int, m: int = 80) -> str:
    return str(n % 10**m).rjust(m, "0")


def find_largest_prime_below_10k_unity(k: int,
                                      *,
                                      max_steps: int = 400_000,
                                      small_prime_limit: int = 50_000,
                                      extra_mr_bases: Optional[List[int]] = None) -> Dict[str, object]:
    """
    Main UNITY orchestrator (Truth operator):
      - runs one aggressive branch (ALT_WHEEL_BPSW) built for bigints
      - returns winner + full witness + metrics
    """
    if extra_mr_bases is None:
        # Good default: deterministic-ish flavor without being slow
        # (Do NOT include too many if you're running on laptop.)
        extra_mr_bases = [3, 5, 7, 11, 13, 17, 19]

    res = branch_alt_wheel_find_below_10k(
        k,
        max_steps=max_steps,
        small_prime_limit=small_prime_limit,
        extra_mr_bases=extra_mr_bases
    )

    out = {
        "k": k,
        "winner": res.name if res.found else "NONE",
        "found": res.found,
        #"prime": str(res.n) if res.n is not None else None,
        "prime_head": head_decimal(res.n, 80) if res.n is not None else None,
        "prime_digits": decimal_digits(res.n) if res.n is not None else None,

        "metrics": {
            "steps": res.steps,
            "tests": res.tests,
            "rejects": res.rejects,
            "drift": res.drift,
            "sat": res.sat,
            "E": res.E,
            "confidence": res.confidence,
            "runtime_s": res.runtime_s,
        },
        "witness": res.witness,
        "notes": [
            "UNITY hard-constraints: small prime divisibility + square check.",
            "Truth filter: BPSW (MR base 2 + strong Lucas).",
            "Confidence number is conservative and comes from MR error bound only.",
            "If you want a formal certificate (ECPP/Pratt), say 'certifikat' and I’ll add prove-mode."
        ],
    }
    return out


# =========================
# 7) CLI demo
# =========================

if __name__ == "__main__":
    k = 5000
    report = find_largest_prime_below_10k_unity(
        k,
        max_steps=250_000,         # usually enough near 10^k
        small_prime_limit=30_000,  # speed/strength tradeoff
        extra_mr_bases=[3,5,7,11,13,17]  # tweak as you like
    )
    print(report["winner"], "| found:", report["found"])
    print("metrics:", report["metrics"])
    if report["found"]:
        print("prime digits:", report["prime_digits"])
        print("prime head:", report["prime_head"])
    else:
        print("No prime found within max_steps; increase max_steps.")
