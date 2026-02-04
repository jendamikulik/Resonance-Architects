import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def simulate(params, T=180, shocks=None, seed=None):
    """
    Discrete-time toy model:
    State: Trust Tt, Stability St, Liquidity Lt, Inflation pit, Transition rate rho
    Unity 100 gates:
      - continuity rail success depends on St and Lt
      - if continuity < threshold or disputes > threshold -> rollback (rho -> 0, slow recovery)
      - coercion incidents forced to 0 by design (not modeled as random here)
    """
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = rng

    # Unpack
    aW, bAI, cpi = params["trust_gain_witness"], params["trust_loss_info"], params["trust_loss_infl"]
    dW, eAC = params["stab_gain_witness"], params["stab_loss_clear"]
    liq_in, liq_outA, liq_outFX = params["liq_in"], params["liq_out_attack"], params["liq_out_fx"]
    pi_M, pi_y, pi_T = params["pi_M"], params["pi_y"], params["pi_T"]
    y_gain, y_loss = params["y_gain"], params["y_loss"]
    rho_max = params["rho_max"]
    gate_cont = params["gate_cont"]
    gate_disp = params["gate_disp"]
    # initial
    Tr = params["T0"];
    St = params["S0"];
    Lq = params["L0"];
    pi = params["pi0"];
    y = params["y0"]
    W = params["W0"]  # witness quality
    rho = 0.0
    D = params["D0"]  # legacy debt stock (normalized)

    # tracks
    T_hist = np.zeros(T);
    S_hist = np.zeros(T);
    L_hist = np.zeros(T);
    pi_hist = np.zeros(T)
    rho_hist = np.zeros(T);
    cont_hist = np.zeros(T);
    disp_hist = np.zeros(T);
    W_hist = np.zeros(T);
    D_hist = np.zeros(T)
    rollback_hist = np.zeros(T, dtype=int)

    for t in range(T):
        # Shocks
        if shocks is None:
            AL = local_rng.gamma(2.0, 0.15)  # liquidity attack
            AI = local_rng.gamma(2.0, 0.12)  # info attack
            AC = local_rng.gamma(2.0, 0.10)  # clearing attack
            AFX = local_rng.gamma(2.0, 0.10)  # FX pressure
        else:
            AL, AI, AC, AFX = shocks(t, local_rng)

        # Witness improvement schedule (Unity rollout)
        # grows, but slows if stability low
        W = clip01(W + params["W_growth"] * (0.3 + 0.7 * Tr) * (0.3 + 0.7 * St))

        # Transition rate gearbox (sigmoid-like) based on trust, stability, and attacks
        z = params["rho_eta_T"] * Tr + params["rho_eta_S"] * St - params["rho_eta_AL"] * AL - params["rho_eta_AC"] * AC
        rho = rho_max * (1 / (1 + np.exp(-z)))

        # Continuity success: depends on stability + liquidity, penalized by clearing attack
        cont = clip01(0.15 + 0.55 * St + 0.35 * clip01(Lq) - 0.20 * AC)
        # Disputes: decrease with witness quality, increase with attacks and inflation
        disp = clip01(0.20 * (1 - W) + 0.10 * AI + 0.10 * AC + 0.05 * pi)

        # Unity 100 gates: rollback if continuity too low OR disputes too high
        rollback = (cont < gate_cont) or (disp > gate_disp)
        if rollback:
            rho = 0.0
            # emergency stabilization bump: prioritize continuity rail & ops focus
            St = clip01(St + params["rollback_stab_boost"] * W - 0.05 * AC)
            Tr = clip01(Tr + params["rollback_trust_boost"] * W - 0.05 * AI)
            rollback_hist[t] = 1

        # Debt conversion (only when rho>0)
        D = D * (1 - rho * params["debt_convert_speed"])

        # Liquidity dynamics (simple)
        inflow = liq_in * (0.4 + 0.6 * Tr) * (0.4 + 0.6 * St)
        outflow = liq_outA * AL + liq_outFX * AFX + params["liq_out_panic"] * (1 - Tr)
        Lq = clip01(Lq + inflow - outflow)

        # Output dynamics
        y = clip01(y + y_gain * (Tr * St) - y_loss * (AI + AC) * 0.2 - params["y_transition_cost"] * rho)

        # Money growth proxy: assume emergency liquidity adds to M a bit when L low
        dM = params["dM_base"] + params["dM_emergency"] * (1 - Lq)
        # Inflation update
        pi = clip01(pi + pi_M * dM - pi_y * (y - 0.5) + pi_T * (1 - Tr))

        # Trust and Stability update
        Tr = clip01(Tr + aW * W - bAI * AI - cpi * pi)
        St = clip01(St + dW * W - eAC * AC)

        # store
        T_hist[t] = Tr;
        S_hist[t] = St;
        L_hist[t] = Lq;
        pi_hist[t] = pi
        rho_hist[t] = rho;
        cont_hist[t] = cont;
        disp_hist[t] = disp;
        W_hist[t] = W;
        D_hist[t] = D

    success = (cont_hist.mean() >= params["success_cont_mean"]) and (cont_hist.min() >= params["success_cont_min"]) \
              and (disp_hist.mean() <= params["success_disp_mean"]) and (
                          np.sum(rollback_hist) <= params["success_max_rollbacks"])
    return {
        "T": T_hist, "S": S_hist, "L": L_hist, "pi": pi_hist, "rho": rho_hist,
        "cont": cont_hist, "disp": disp_hist, "W": W_hist, "D": D_hist,
        "rollback": rollback_hist, "success": success
    }


# Parameters for "Unity 100" style (defensive, conservative)
params = dict(
    # initial conditions (0..1 normalized)
    T0=0.55, S0=0.60, L0=0.55, pi0=0.20, y0=0.55, W0=0.20, D0=1.0,
    # witness growth
    W_growth=0.010,
    # trust/stability response
    trust_gain_witness=0.030, trust_loss_info=0.060, trust_loss_infl=0.030,
    stab_gain_witness=0.028, stab_loss_clear=0.055,
    # liquidity
    liq_in=0.030, liq_out_attack=0.090, liq_out_fx=0.060, liq_out_panic=0.030,
    # inflation
    pi_M=0.025, pi_y=0.015, pi_T=0.020,
    # output
    y_gain=0.020, y_loss=0.020, y_transition_cost=0.020,
    # transition rate
    rho_max=0.060, rho_eta_T=2.0, rho_eta_S=2.2, rho_eta_AL=1.6, rho_eta_AC=1.4,
    debt_convert_speed=0.30,
    # gates
    gate_cont=0.86, gate_disp=0.18,
    rollback_stab_boost=0.050, rollback_trust_boost=0.040,
    # money growth proxy
    dM_base=0.010, dM_emergency=0.020,
    # success criteria
    success_cont_mean=0.90, success_cont_min=0.80,
    success_disp_mean=0.16,
    success_max_rollbacks=18
)

# Run one illustrative path
res = simulate(params, T=180, seed=123)

# Plot key series
plt.figure()
plt.plot(res["cont"])
plt.axhline(params["gate_cont"])
plt.axhline(params["success_cont_mean"], linestyle="--")
plt.ylim(0, 1)
plt.title("Continuity rail success")
plt.xlabel("day")
plt.ylabel("success")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res["disp"])
plt.axhline(params["gate_disp"])
plt.axhline(params["success_disp_mean"], linestyle="--")
plt.ylim(0, 1)
plt.title("Dispute rate proxy")
plt.xlabel("day")
plt.ylabel("disp")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res["T"], label="Trust")
plt.plot(res["S"], label="Stability")
plt.plot(res["L"], label="Liquidity")
plt.legend()
plt.ylim(0, 1)
plt.title("Core state variables")
plt.xlabel("day")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res["W"], label="Witness quality")
plt.plot(res["rho"], label="Transition rate rho")
plt.legend()
plt.ylim(0, 1)
plt.title("Witness and transition gearbox")
plt.xlabel("day")
plt.tight_layout()
plt.show()

res["success"], res["cont"].mean(), res["cont"].min(), res["disp"].mean(), res["rollback"].sum()

