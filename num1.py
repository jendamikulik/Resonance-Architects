import numpy as np
import matplotlib.pyplot as plt

def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def simulate(params, T=180, seed=0):
    rng = np.random.default_rng(seed)
    aW, bAI, cpi = params["trust_gain_witness"], params["trust_loss_info"], params["trust_loss_infl"]
    dW, eAC = params["stab_gain_witness"], params["stab_loss_clear"]
    liq_in, liq_outA, liq_outFX = params["liq_in"], params["liq_out_attack"], params["liq_out_fx"]
    pi_M, pi_y, pi_T = params["pi_M"], params["pi_y"], params["pi_T"]
    y_gain, y_loss = params["y_gain"], params["y_loss"]
    rho_max = params["rho_max"]
    gate_cont, gate_disp = params["gate_cont"], params["gate_disp"]

    Tr, St, Lq, pi, y, W, D = params["T0"], params["S0"], params["L0"], params["pi0"], params["y0"], params["W0"], params["D0"]
    rho = 0.0

    T_hist = np.zeros(T); S_hist = np.zeros(T); L_hist = np.zeros(T); pi_hist = np.zeros(T)
    rho_hist = np.zeros(T); cont_hist = np.zeros(T); disp_hist = np.zeros(T); W_hist = np.zeros(T); D_hist = np.zeros(T)
    rollback_hist = np.zeros(T, dtype=int)

    for t in range(T):
        # worst-case-ish attacks, slightly reduced when witness is high (resilience)
        base = 1.0 - 0.35*W
        AL  = base * rng.gamma(2.0, 0.14)
        AI  = base * rng.gamma(2.0, 0.11)
        AC  = base * rng.gamma(2.0, 0.09)
        AFX = base * rng.gamma(2.0, 0.09)

        # witness improves
        W = clip01(W + params["W_growth"] * (0.35 + 0.65*Tr) * (0.35 + 0.65*St))

        # transition gearbox (can be 0 often in worst-case)
        z = params["rho_eta_T"]*Tr + params["rho_eta_S"]*St - params["rho_eta_AL"]*AL - params["rho_eta_AC"]*AC
        rho = rho_max * (1/(1+np.exp(-z)))

        # continuity rail success (improved by dedicated ops floor)
        cont_floor = params["cont_floor"]
        cont = clip01(cont_floor + 0.50*St + 0.30*Lq - 0.18*AC)

        # disputes proxy (improved by better tooling)
        disp = clip01(0.18*(1-W) + 0.08*AI + 0.08*AC + 0.04*pi)

        # gates -> rollback / freeze expansion
        if (cont < gate_cont) or (disp > gate_disp):
            rho = 0.0
            # strong continuity mode: liquidity injections & ops priority
            St = clip01(St + params["rollback_stab_boost"]*W - 0.03*AC)
            Tr = clip01(Tr + params["rollback_trust_boost"]*W - 0.03*AI)
            rollback_hist[t] = 1

        # debt conversion (only when rho>0)
        D = D * (1 - rho*params["debt_convert_speed"])

        # liquidity dynamics: add emergency buffer specifically for continuity rail
        inflow = liq_in*(0.45 + 0.55*Tr)*(0.45 + 0.55*St) + params["liq_buffer"]*W
        outflow = liq_outA*AL + liq_outFX*AFX + params["liq_out_panic"]*(1-Tr)
        Lq = clip01(Lq + inflow - outflow)

        # output
        y = clip01(y + y_gain*(Tr*St) - y_loss*(AI+AC)*0.18 - params["y_transition_cost"]*rho)

        # money growth proxy (bounded)
        dM = params["dM_base"] + params["dM_emergency"]*(1-Lq)

        # inflation
        pi = clip01(pi + pi_M*dM - pi_y*(y-0.5) + pi_T*(1-Tr))

        # trust & stability update
        Tr = clip01(Tr + aW*W - bAI*AI - cpi*pi)
        St = clip01(St + dW*W - eAC*AC)

        # store
        T_hist[t]=Tr; S_hist[t]=St; L_hist[t]=Lq; pi_hist[t]=pi
        rho_hist[t]=rho; cont_hist[t]=cont; disp_hist[t]=disp; W_hist[t]=W; D_hist[t]=D

    success = (cont_hist.mean() >= params["success_cont_mean"]) and (cont_hist.min() >= params["success_cont_min"]) \
              and (disp_hist.mean() <= params["success_disp_mean"]) and (rollback_hist.sum() <= params["success_max_rollbacks"])
    return {"T":T_hist,"S":S_hist,"L":L_hist,"pi":pi_hist,"rho":rho_hist,"cont":cont_hist,"disp":disp_hist,"W":W_hist,"D":D_hist,"rollback":rollback_hist,"success":success}

# Strengthened "Unity 100" parameters: continuity floor + liquidity buffer + reduced attack effectiveness as W increases
params2 = dict(
    T0=0.60, S0=0.65, L0=0.75, pi0=0.18, y0=0.60, W0=0.25, D0=1.0,
    W_growth=0.014,
    trust_gain_witness=0.040, trust_loss_info=0.045, trust_loss_infl=0.025,
    stab_gain_witness=0.035, stab_loss_clear=0.040,
    liq_in=0.040, liq_out_attack=0.060, liq_out_fx=0.040, liq_out_panic=0.020,
    liq_buffer=0.030,
    pi_M=0.020, pi_y=0.018, pi_T=0.015,
    y_gain=0.024, y_loss=0.018, y_transition_cost=0.015,
    rho_max=0.050, rho_eta_T=2.0, rho_eta_S=2.2, rho_eta_AL=1.7, rho_eta_AC=1.6,
    debt_convert_speed=0.25,
    cont_floor=0.30,
    gate_cont=0.85, gate_disp=0.22,
    rollback_stab_boost=0.060, rollback_trust_boost=0.050,
    dM_base=0.008, dM_emergency=0.012,
    success_cont_mean=0.90, success_cont_min=0.80,
    success_disp_mean=0.18,
    success_max_rollbacks=60
)

res2 = simulate(params2, T=180, seed=7)

# Plot
plt.figure()
plt.plot(res2["cont"])
plt.axhline(params2["gate_cont"])
plt.axhline(params2["success_cont_mean"], linestyle="--")
plt.ylim(0,1)
plt.title("Continuity rail success (Unity 100 hardened)")
plt.xlabel("day")
plt.ylabel("success")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res2["disp"])
plt.axhline(params2["gate_disp"])
plt.axhline(params2["success_disp_mean"], linestyle="--")
plt.ylim(0,1)
plt.title("Dispute rate proxy (Unity 100 hardened)")
plt.xlabel("day")
plt.ylabel("disp")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res2["T"], label="Trust")
plt.plot(res2["S"], label="Stability")
plt.plot(res2["L"], label="Liquidity")
plt.legend()
plt.ylim(0,1)
plt.title("Core state variables (Unity 100 hardened)")
plt.xlabel("day")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res2["W"], label="Witness quality")
plt.plot(res2["rho"], label="Transition rate rho")
plt.legend()
plt.ylim(0,1)
plt.title("Witness & transition gearbox (Unity 100 hardened)")
plt.xlabel("day")
plt.tight_layout()
plt.show()

(res2["success"], float(res2["cont"].mean()), float(res2["cont"].min()), float(res2["disp"].mean()), int(res2["rollback"].sum()), float(res2["rho"].mean()))

