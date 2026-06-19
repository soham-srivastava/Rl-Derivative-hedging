"""
experiment2_tc_sweep.py
────────────────────────
TC Regime Analysis:
  Sweep transaction cost tc ∈ {0, 0.05%, 0.1%, 0.2%, 0.5%, 1%, 2%}
  Train fresh SAC at each TC level.
  Compare: SAC PnL, BLS PnL, hedge frequency (turnover), and spread of costs.

Key insight: BLS rebalances every step regardless of TC.
             RL should learn to hedge less frequently at high TC.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import ENV, SAC, EVAL
from env import OptionsHedgingEnv
from agents import SACAgent
from bls_baseline import run_baseline

os.makedirs(EVAL["plot_dir"], exist_ok=True)

TC_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


def train_and_eval(tc: float, steps=60_000, n_eval=300, seed=0):
    cfg = dict(ENV, tc=tc)

    # Train SAC
    env = OptionsHedgingEnv(cfg)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    agent = SACAgent(obs_dim, n_act, SAC)
    obs, _ = env.reset(seed=seed)
    for s in range(steps):
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.update()
        obs = next_obs
        if done:
            obs, _ = env.reset()

    # Eval SAC
    rng = np.random.default_rng(seed + 1)
    pnl_sac, turnover_sac = [], []
    for _ in range(n_eval):
        env2 = OptionsHedgingEnv(cfg)
        obs, _ = env2.reset(seed=int(rng.integers(0, 2**31)))
        prev_pos = 0
        turns = 0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, _, done, _, _ = env2.step(action)
            turns += abs(env2.position - prev_pos)
            prev_pos = env2.position
            if done:
                break
        pnl_sac.append(env2.running_pnl)
        turnover_sac.append(turns / env2.T)   # avg shares traded per step

    # Eval BLS at same TC
    bls_res = run_baseline("delta", n_episodes=n_eval, seed=seed+2)
    # Also measure BLS turnover (always full rebalance)
    rng2 = np.random.default_rng(seed + 2)
    turnover_bls = []
    from bls_baseline import BLSDeltaHedger
    for _ in range(n_eval):
        env3 = OptionsHedgingEnv(cfg)
        obs, _ = env3.reset(seed=int(rng2.integers(0, 2**31)))
        hedger = BLSDeltaHedger(env3, "delta")
        prev_pos = 0; turns = 0
        while True:
            action = hedger.select_action(obs)
            obs, _, done, _, _ = env3.step(action)
            turns += abs(env3.position - prev_pos)
            prev_pos = env3.position
            if done:
                break
        turnover_bls.append(turns / env3.T)

    return dict(
        tc=tc,
        pnl_sac=np.array(pnl_sac),
        pnl_bls=bls_res["pnl"],
        turn_sac=np.mean(turnover_sac),
        turn_bls=np.mean(turnover_bls),
        mean_sac=np.mean(pnl_sac),
        mean_bls=bls_res["mean"],
        std_sac=np.std(pnl_sac),
        std_bls=bls_res["std"],
        sharpe_sac=np.mean(pnl_sac)/(np.std(pnl_sac)+1e-8),
        sharpe_bls=bls_res["sharpe"],
    )


def run():
    print("\n" + "="*60)
    print(" EXPERIMENT 2 — Transaction Cost Sweep")
    print("="*60)

    results = []
    for tc in TC_GRID:
        print(f"\n  TC={tc*100:.2f}%  training …", end=" ", flush=True)
        t0 = time.time()
        r = train_and_eval(tc)
        print(f"done in {time.time()-t0:.0f}s | "
              f"SAC Sharpe={r['sharpe_sac']:+.3f}  BLS Sharpe={r['sharpe_bls']:+.3f}  "
              f"SAC turn={r['turn_sac']:.2f}  BLS turn={r['turn_bls']:.2f}")
        results.append(r)

    tcs         = [r["tc"]*100 for r in results]
    sharpe_sac  = [r["sharpe_sac"]  for r in results]
    sharpe_bls  = [r["sharpe_bls"]  for r in results]
    turn_sac    = [r["turn_sac"]    for r in results]
    turn_bls    = [r["turn_bls"]    for r in results]
    mean_sac    = [r["mean_sac"]    for r in results]
    mean_bls    = [r["mean_bls"]    for r in results]
    std_sac     = [r["std_sac"]     for r in results]
    std_bls     = [r["std_bls"]     for r in results]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#08090c")

    panels = [
        (axes[0,0], sharpe_sac, sharpe_bls, "Sharpe Ratio", True),
        (axes[0,1], mean_sac,   mean_bls,   "Mean PnL",     True),
        (axes[1,0], std_sac,    std_bls,     "PnL Std Dev (risk)", False),
        (axes[1,1], turn_sac,   turn_bls,    "Hedge Turnover (shares/step)", False),
    ]

    for ax, y_sac, y_bls, ylabel, higher_better in panels:
        ax.set_facecolor("#0f1117")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2130")
        ax.plot(tcs, y_sac, "o-", color="#e05c5c", linewidth=1.8,
                markersize=5, label="SAC")
        ax.plot(tcs, y_bls, "s--", color="#5c7ae0", linewidth=1.8,
                markersize=5, label="BLS-Δ")
        ax.set_xlabel("Transaction Cost (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, color="#eef0f7")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(colors="#4a4f63")

    plt.suptitle("Experiment 2 — TC Sensitivity: SAC vs BLS-Delta",
                 color="#eef0f7", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(EVAL["plot_dir"], "exp2_tc_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")

    # Summary
    print(f"\n  {'TC%':>6} | {'SAC Sharpe':>11} {'BLS Sharpe':>11} | "
          f"{'SAC Turn':>9} {'BLS Turn':>9} | {'Edge':>8}")
    print("  " + "-"*65)
    for r in results:
        edge = r['sharpe_sac'] - r['sharpe_bls']
        print(f"  {r['tc']*100:>5.2f}% | {r['sharpe_sac']:>+11.3f} {r['sharpe_bls']:>+11.3f} | "
              f"{r['turn_sac']:>9.2f} {r['turn_bls']:>9.2f} | {edge:>+8.3f}")

    return results


if __name__ == "__main__":
    run()
