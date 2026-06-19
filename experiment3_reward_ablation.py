"""
experiment3_reward_ablation.py
──────────────────────────────
Reward Function Ablation:
  Train SAC with three reward shapes:
    pnl_tc   : raw step PnL minus TC  (myopic)
    sharpe   : PnL − 0.5·PnL²        (variance penalty)
    cvar     : PnL − λ·max(−PnL,0)²  (tail-loss penalty)

  Evaluate all three on the SAME test environment.
  Metrics: mean PnL, std, Sharpe, CVaR-5%, max drawdown.

  Key question: does CVaR reward actually produce lower tail risk?
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

os.makedirs(EVAL["plot_dir"], exist_ok=True)

REWARD_FNS = ["pnl_tc", "sharpe", "cvar"]
COLORS_RW  = {"pnl_tc": "#f5c842", "sharpe": "#3de88a", "cvar": "#e05c5c"}
LABELS_RW  = {"pnl_tc": "PnL-TC", "sharpe": "Sharpe-shaped", "cvar": "CVaR-shaped"}


def train_agent(reward_fn: str, steps=80_000, seed=0):
    cfg = dict(ENV, reward_type=reward_fn)
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
    return agent


def evaluate_agent(agent, n=500, seed=42):
    """Always evaluate with pnl_tc so rewards are comparable."""
    cfg = dict(ENV, reward_type="pnl_tc")
    rng = np.random.default_rng(seed)
    pnls, max_dds = [], []
    for _ in range(n):
        env = OptionsHedgingEnv(cfg)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        pnls.append(env.running_pnl)
        # max drawdown of running PnL curve
        curve = np.array(env.history["pnl"])
        peak  = np.maximum.accumulate(curve)
        dd    = (curve - peak)
        max_dds.append(dd.min())
    pnls    = np.array(pnls)
    max_dds = np.array(max_dds)
    cvar5   = np.mean(pnls[pnls <= np.percentile(pnls, 5)])
    return dict(
        pnl=pnls, max_dd=max_dds,
        mean=np.mean(pnls), std=np.std(pnls),
        sharpe=np.mean(pnls)/(np.std(pnls)+1e-8),
        cvar5=cvar5,
        mean_dd=np.mean(max_dds),
    )


def run():
    print("\n" + "="*60)
    print(" EXPERIMENT 3 — Reward Function Ablation")
    print("="*60)

    agents, results = {}, {}
    for rw in REWARD_FNS:
        print(f"\n  Training SAC [{rw}] …", end=" ", flush=True)
        t0 = time.time()
        agent = train_agent(rw)
        agents[rw] = agent
        print(f"done in {time.time()-t0:.0f}s")
        print(f"  Evaluating …", end=" ", flush=True)
        res = evaluate_agent(agent)
        results[rw] = res
        print(f"μ={res['mean']:+.3f}  σ={res['std']:.3f}  "
              f"Sharpe={res['sharpe']:+.3f}  CVaR5={res['cvar5']:+.3f}  "
              f"MaxDD={res['mean_dd']:+.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#08090c")

    # 1. PnL distribution overlay
    ax = axes[0]; ax.set_facecolor("#0f1117")
    bins = np.linspace(-300, 200, 55)
    for rw in REWARD_FNS:
        ax.hist(results[rw]["pnl"], bins=bins, alpha=0.55,
                color=COLORS_RW[rw], label=LABELS_RW[rw], edgecolor="none")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2130")
    ax.set_xlabel("Episode PnL"); ax.set_ylabel("Count")
    ax.set_title("PnL Distribution by Reward", color="#eef0f7")
    ax.legend(); ax.grid(alpha=0.3); ax.tick_params(colors="#4a4f63")

    # 2. Max drawdown distribution
    ax = axes[1]; ax.set_facecolor("#0f1117")
    bins_dd = np.linspace(-200, 10, 50)
    for rw in REWARD_FNS:
        ax.hist(results[rw]["max_dd"], bins=bins_dd, alpha=0.55,
                color=COLORS_RW[rw], label=LABELS_RW[rw], edgecolor="none")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2130")
    ax.set_xlabel("Max Drawdown per Episode"); ax.set_ylabel("Count")
    ax.set_title("Max Drawdown Distribution", color="#eef0f7")
    ax.legend(); ax.grid(alpha=0.3); ax.tick_params(colors="#4a4f63")

    # 3. Bar chart of risk metrics
    ax = axes[2]; ax.set_facecolor("#0f1117")
    metrics  = ["mean", "std", "sharpe", "cvar5", "mean_dd"]
    m_labels = ["Mean PnL", "Std PnL", "Sharpe", "CVaR-5%", "Mean MaxDD"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, rw in enumerate(REWARD_FNS):
        vals = [results[rw][m] for m in metrics]
        bars = ax.bar(x + i*width - width, vals, width,
                      label=LABELS_RW[rw], color=COLORS_RW[rw], alpha=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2130")
    ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=7, rotation=15)
    ax.set_title("Risk Metrics Comparison", color="#eef0f7")
    ax.axhline(0, color="#4a4f63", linewidth=0.7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    ax.tick_params(colors="#4a4f63")

    plt.suptitle("Experiment 3 — Reward Function Ablation",
                 color="#eef0f7", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(EVAL["plot_dir"], "exp3_reward_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")

    print(f"\n  {'Reward':>16} | {'Mean':>8} {'Std':>8} {'Sharpe':>8} {'CVaR5':>8} {'MeanDD':>8}")
    print("  " + "-"*60)
    for rw in REWARD_FNS:
        r = results[rw]
        print(f"  {LABELS_RW[rw]:>16} | {r['mean']:>+8.3f} {r['std']:>8.3f} "
              f"{r['sharpe']:>+8.3f} {r['cvar5']:>+8.3f} {r['mean_dd']:>+8.3f}")

    return results


if __name__ == "__main__":
    run()
