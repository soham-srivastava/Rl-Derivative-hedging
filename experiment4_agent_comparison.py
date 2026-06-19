"""
experiment4_agent_comparison.py
─────────────────────────────────
Full head-to-head: DQN vs PPO vs SAC vs BLS-Delta vs BLS-DeltaGamma

  - Train all three RL agents from scratch (80k steps each)
  - Evaluate all five on the same GBM environment
  - Produce the histogram matching the slide image
  - Find which RL agent beats BLS and under what market conditions

Breakdown by moneyness bucket (OTM / ATM / ITM) to show where RL excels.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

sys.path.insert(0, os.path.dirname(__file__))
from config import ENV, DQN, PPO, SAC, EVAL
from env import OptionsHedgingEnv
from agents import DQNAgent, PPOAgent, SACAgent
from bls_baseline import run_baseline, BLSDeltaHedger

os.makedirs(EVAL["plot_dir"], exist_ok=True)

COLORS = {
    "DQN"      : "#e05c5c",
    "PPO"      : "#f5c842",
    "SAC"      : "#3de88a",
    "BLS-Δ"   : "#5c7ae0",
    "BLS-ΔΓ"  : "#a07ae0",
}


def train_dqn(steps=80_000, seed=0):
    env = OptionsHedgingEnv(ENV)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    agent = DQNAgent(obs_dim, n_act, DQN)
    obs, _ = env.reset(seed=seed)
    for s in range(steps):
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.update()
        obs = next_obs
        if done:
            agent.decay_epsilon()
            obs, _ = env.reset()
    return agent


def train_ppo(steps=80_000, seed=0):
    from config import PPO as PPO_CFG
    env = OptionsHedgingEnv(ENV)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    agent = PPOAgent(obs_dim, n_act, PPO_CFG)
    obs, _ = env.reset(seed=seed)
    rollout_buf_size = PPO_CFG["n_steps"]
    step = 0
    while step < steps:
        rb = 0
        while rb < rollout_buf_size and step < steps:
            action = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store(obs, action, reward, done)
            obs = next_obs; step += 1; rb += 1
            if done:
                obs, _ = env.reset()
        agent.update(last_obs=obs, last_done=False)
    return agent


def train_sac(steps=80_000, seed=0):
    env = OptionsHedgingEnv(ENV)
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


def rollout_rl(agent, n=500, seed=42):
    rng = np.random.default_rng(seed)
    pnls, moneyness = [], []
    for _ in range(n):
        env = OptionsHedgingEnv(ENV)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        S0 = env.S
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        pnls.append(env.running_pnl)
        moneyness.append(env.S / ENV["K"])   # terminal moneyness
    return np.array(pnls), np.array(moneyness)


def rollout_bls(mode="delta", n=500, seed=42):
    rng = np.random.default_rng(seed)
    pnls, moneyness = [], []
    for _ in range(n):
        env = OptionsHedgingEnv(ENV)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        hedger = BLSDeltaHedger(env, mode)
        while True:
            action = hedger.select_action(obs)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        pnls.append(env.running_pnl)
        moneyness.append(env.S / ENV["K"])
    return np.array(pnls), np.array(moneyness)


def stats(pnl):
    return dict(
        mean=np.mean(pnl), std=np.std(pnl),
        sharpe=np.mean(pnl)/(np.std(pnl)+1e-8),
        p5=np.percentile(pnl, 5),
        p95=np.percentile(pnl, 95),
    )


def run():
    print("\n" + "="*60)
    print(" EXPERIMENT 4 — Full Agent Comparison")
    print("="*60)

    # ── Train ─────────────────────────────────────────────────────────────────
    agents = {}
    for name, fn in [("DQN", train_dqn), ("PPO", train_ppo), ("SAC", train_sac)]:
        print(f"\n  Training {name} …", end=" ", flush=True)
        t0 = time.time()
        agents[name] = fn()
        print(f"done in {time.time()-t0:.0f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    N = 500
    print("\n  Evaluating all agents …")
    all_pnl, all_money = {}, {}
    for name, agent in agents.items():
        pnl, money = rollout_rl(agent, N)
        all_pnl[name] = pnl; all_money[name] = money
        s = stats(pnl)
        print(f"    {name:>4}: μ={s['mean']:+.3f}  σ={s['std']:.3f}  "
              f"Sharpe={s['sharpe']:+.3f}  P5={s['p5']:+.3f}")

    for mode, label in [("delta","BLS-Δ"), ("delta_gamma","BLS-ΔΓ")]:
        pnl, money = rollout_bls(mode, N)
        all_pnl[label] = pnl; all_money[label] = money
        s = stats(pnl)
        print(f"    {label:>6}: μ={s['mean']:+.3f}  σ={s['std']:.3f}  "
              f"Sharpe={s['sharpe']:+.3f}  P5={s['p5']:+.3f}")

    # ── Moneyness breakdown ────────────────────────────────────────────────────
    print("\n  Moneyness breakdown (Sharpe):")
    buckets = {"OTM (<0.95)": lambda m: m < 0.95,
               "ATM (0.95–1.05)": lambda m: (m >= 0.95) & (m <= 1.05),
               "ITM (>1.05)": lambda m: m > 1.05}
    for bname, bmask in buckets.items():
        print(f"\n    {bname}")
        for label in list(agents.keys()) + ["BLS-Δ"]:
            mask = bmask(all_money[label])
            if mask.sum() < 5:
                continue
            sub = all_pnl[label][mask]
            sh  = np.mean(sub) / (np.std(sub) + 1e-8)
            print(f"      {label:>6}: n={mask.sum():3d}  Sharpe={sh:+.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#08090c")
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    # 1. Overlapping cost histograms (mirrors slide)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f1117")
    bins = np.linspace(-300, 200, 65)
    order = ["BLS-Δ", "BLS-ΔΓ", "DQN", "PPO", "SAC"]
    for label in order:
        s = stats(all_pnl[label])
        ax1.hist(all_pnl[label], bins=bins, alpha=0.5,
                 color=COLORS[label],
                 label=f"{label}  Sharpe={s['sharpe']:+.3f}  μ={s['mean']:+.1f}",
                 edgecolor="none")
    ax1.set_xlabel("Episode PnL (Hedging Cost = −PnL)")
    ax1.set_ylabel("Number of Trials")
    ax1.set_title("RL Hedge Costs vs. BLS Hedge Costs", color="#eef0f7",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3, axis="y")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    for sp in ax1.spines.values(): sp.set_edgecolor("#1e2130")
    ax1.tick_params(colors="#4a4f63")

    # 2. Sharpe bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f1117")
    labels_ord = order
    sharpes = [stats(all_pnl[l])["sharpe"] for l in labels_ord]
    bar_colors = [COLORS[l] for l in labels_ord]
    bars = ax2.bar(labels_ord, sharpes, color=bar_colors, alpha=0.8, width=0.5)
    ax2.axhline(0, color="#4a4f63", linewidth=0.8)
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x()+bar.get_width()/2, v + 0.002*np.sign(v),
                 f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=8, color="#eef0f7")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio Comparison", color="#eef0f7")
    ax2.grid(alpha=0.3, axis="y")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1e2130")
    ax2.tick_params(colors="#4a4f63")

    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f1117")
    data_ord = [all_pnl[l] for l in labels_ord]
    bp = ax3.boxplot(data_ord, patch_artist=True, notch=True,
                     medianprops=dict(color="#eef0f7", linewidth=1.5),
                     whiskerprops=dict(color="#4a4f63"),
                     capprops=dict(color="#4a4f63"),
                     flierprops=dict(marker=".", markersize=2,
                                     markerfacecolor="#4a4f63", alpha=0.4))
    for patch, c in zip(bp["boxes"], bar_colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax3.set_xticklabels(labels_ord, fontsize=8)
    ax3.axhline(0, color="#4a4f63", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("Episode PnL")
    ax3.set_title("PnL Distribution (Box)", color="#eef0f7")
    ax3.grid(alpha=0.3, axis="y")
    for sp in ax3.spines.values(): sp.set_edgecolor("#1e2130")
    ax3.tick_params(colors="#4a4f63")

    plt.suptitle("Experiment 4 — Full Agent Comparison: RL vs BLS",
                 color="#eef0f7", fontsize=14, y=1.01)

    out = os.path.join(EVAL["plot_dir"], "exp4_agent_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")

    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n  {'Agent':>8} | {'Mean':>8} {'Std':>8} {'Sharpe':>8} {'P5':>8} {'P95':>8}")
    print("  " + "-"*55)
    for label in order:
        s = stats(all_pnl[label])
        print(f"  {label:>8} | {s['mean']:>+8.2f} {s['std']:>8.2f} "
              f"{s['sharpe']:>+8.3f} {s['p5']:>+8.2f} {s['p95']:>+8.2f}")

    return {l: stats(all_pnl[l]) for l in order}


if __name__ == "__main__":
    run()
