"""
evaluate.py — Evaluate trained agents vs BLS baseline and produce comparison plots.

Usage:
    python evaluate.py                        # uses saved models + BLS baseline
    python evaluate.py --quick               # 200-episode quick eval
    python evaluate.py --reward_fn sharpe    # override reward function
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import torch

from config import ENV, DQN, PPO, SAC, EVAL
from env import OptionsHedgingEnv
from agents import DQNAgent, PPOAgent, SACAgent, DEVICE
from bls_baseline import run_baseline


# ── Styling (seaborn, bright/light background) ────────────────────────────────
sns.set_theme(style="whitegrid", palette="bright")
_BRIGHT = sns.color_palette("bright", 5)
COLORS = {
    "DQN"       : _BRIGHT[3],
    "PPO"       : _BRIGHT[2],
    "SAC"       : _BRIGHT[1],
    "BLS-Delta" : _BRIGHT[0],
}

plt.rcParams.update({
    "figure.facecolor"  : "#ffffff",
    "axes.facecolor"    : "#eef3fb",
    "savefig.facecolor" : "#ffffff",
    "axes.edgecolor"    : "#c8d2e0",
    "axes.labelcolor"   : "#1b1f27",
    "xtick.color"       : "#4a4f63",
    "ytick.color"       : "#4a4f63",
    "text.color"        : "#1b1f27",
    "grid.color"        : "#d6dee8",
    "grid.linewidth"    : 0.6,
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 9,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "legend.fontsize"   : 8,
    "legend.facecolor"  : "#ffffff",
    "legend.edgecolor"  : "#c8d2e0",
})


def set_spine(ax, color="#c8d2e0"):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


# ── Agent rollout ──────────────────────────────────────────────────────────────

def rollout_agent(agent, agent_type: str, n_episodes: int, seed: int = 42):
    """Run a trained agent for n_episodes. Returns dict of episode stats."""
    env = OptionsHedgingEnv(ENV)
    rng = np.random.default_rng(seed)

    ep_pnl, ep_cost, ep_rewards = [], [], []
    ep_pos_std = []          # position volatility (over-hedging proxy)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_r = 0.0
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_r += reward
            if done:
                break
        ep_pnl.append(env.running_pnl)
        ep_cost.append(-env.running_pnl)
        ep_rewards.append(ep_r)
        ep_pos_std.append(np.std(env.history["position"]))

    return {
        "pnl"     : np.array(ep_pnl),
        "cost"    : np.array(ep_cost),
        "reward"  : np.array(ep_rewards),
        "pos_std" : np.array(ep_pos_std),
        "mean"    : np.mean(ep_pnl),
        "std"     : np.std(ep_pnl),
        "sharpe"  : np.mean(ep_pnl) / (np.std(ep_pnl) + 1e-8),
    }


def load_agent(name: str, obs_dim: int, n_actions: int):
    path = os.path.join(EVAL["model_dir"], f"{name}.pt")
    if not os.path.exists(path):
        return None
    if name == "dqn":
        a = DQNAgent(obs_dim, n_actions, DQN); a.load(path)
    elif name == "ppo":
        a = PPOAgent(obs_dim, n_actions, PPO); a.load(path)
    elif name == "sac":
        a = SACAgent(obs_dim, n_actions, SAC); a.load(path)
    return a


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_cost_distributions(results: dict, save_path: str):
    """Overlapping cost histograms — mirrors the slide image."""
    fig, ax = plt.subplots(figsize=(9, 5))

    all_costs = np.concatenate([res["cost"] for res in results.values()])
    lo, hi = np.percentile(all_costs, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    for label, res in results.items():
        costs = res["cost"]
        ax.hist(costs, bins=bins, alpha=0.55, color=COLORS.get(label, "#888"),
                label=f"{label}  μ={res['mean']:.3f}  σ={res['std']:.3f}",
                edgecolor="none")

    ax.set_xlabel("Hedging Cost  (−PnL per episode)")
    ax.set_ylabel("Number of Trials")
    ax.set_title("RL Hedge Costs vs. BLS Hedge Costs", pad=12,
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    set_spine(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_pnl_distributions(results: dict, save_path: str):
    """Box-and-whisker PnL comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels  = list(results.keys())
    data    = [results[k]["pnl"] for k in labels]
    colors  = [COLORS.get(k, "#888") for k in labels]

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="#1b1f27", linewidth=1.5),
                    whiskerprops=dict(color="#4a4f63"),
                    capprops=dict(color="#4a4f63"),
                    flierprops=dict(marker=".", markersize=2,
                                   markerfacecolor="#4a4f63", alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)

    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Episode PnL")
    ax.set_title("PnL Distribution by Agent", pad=10, fontsize=12)
    ax.axhline(0, color="#4a4f63", linestyle="--", linewidth=0.8)
    set_spine(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_training_curves(save_path: str):
    """Plot episode reward curves from saved logs."""
    log_dir = EVAL["log_dir"]
    found = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    names = ["dqn", "ppo", "sac"]

    for ax, name in zip(axes, names):
        path = os.path.join(log_dir, f"{name}_rewards.npy")
        color = COLORS.get(name.upper(), "#888")
        if os.path.exists(path):
            found = True
            rewards = np.load(path)
            ax.plot(rewards, color=color, alpha=0.25, linewidth=0.6)
            if len(rewards) >= 20:
                ma = np.convolve(rewards, np.ones(50)/50, mode="valid")
                ax.plot(np.arange(49, len(rewards)), ma, color=color, linewidth=1.4,
                        label=f"MA-50")
            ax.set_title(name.upper())
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, f"No log: {name}_rewards.npy",
                    ha="center", va="center", transform=ax.transAxes, color="#4a4f63")
            ax.set_title(name.upper(), color="#4a4f63")
        set_spine(ax)

    if not found:
        plt.close()
        return

    fig.suptitle("Training Curves", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_summary_table(results: dict, save_path: str):
    """Sharpe / mean / std / 5th percentile summary."""
    labels  = list(results.keys())
    metrics = {
        "Mean PnL"     : [results[k]["mean"]   for k in labels],
        "Std PnL"      : [results[k]["std"]    for k in labels],
        "Sharpe"       : [results[k]["sharpe"] for k in labels],
        "5th pct PnL"  : [np.percentile(results[k]["pnl"], 5) for k in labels],
        "95th pct PnL" : [np.percentile(results[k]["pnl"], 95) for k in labels],
    }

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    header = ["Agent"] + list(metrics.keys())
    rows   = []
    for i, label in enumerate(labels):
        row = [label] + [f"{metrics[m][i]:+.4f}" for m in metrics]
        rows.append(row)

    tbl = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    pos_color, neg_color = sns.color_palette("bright", 5)[1], sns.color_palette("bright", 5)[3]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#eef3fb" if r > 0 else "#dbe6f5")
        cell.set_edgecolor("#c8d2e0")
        cell.set_text_props(color="#1b1f27")
        if r > 0 and c > 0:
            val = float(cell.get_text().get_text())
            if c in (1, 3, 4, 5):  # PnL columns — green positive
                cell.set_text_props(color=pos_color if val >= 0 else neg_color)

    ax.set_title("Agent Comparison Summary", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_single_episode(agent, agent_name: str, save_path: str):
    """Plot one example episode: price path, position, cumulative PnL."""
    env = OptionsHedgingEnv(ENV)
    obs, _ = env.reset(seed=999)
    while True:
        action = agent.select_action(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    steps = np.arange(len(env.history["S"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    line_color = COLORS.get(agent_name, sns.color_palette("bright", 5)[0])

    axes[0].plot(steps, env.history["S"], color=line_color, linewidth=1.2)
    axes[0].axhline(ENV["K"], color="#4a4f63", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Spot Price"); axes[0].set_title(f"{agent_name} — Single Episode")

    axes[1].step(steps, env.history["position"], color=sns.color_palette("bright", 5)[2],
                 linewidth=1.2, where="post")
    axes[1].axhline(0, color="#4a4f63", linestyle="--", linewidth=0.6)
    axes[1].set_ylabel("Hedge Position (shares)")

    axes[2].plot(steps, env.history["pnl"], color=sns.color_palette("bright", 5)[1], linewidth=1.2)
    axes[2].axhline(0, color="#4a4f63", linestyle="--", linewidth=0.6)
    axes[2].set_ylabel("Cumulative PnL"); axes[2].set_xlabel("Step (trading day)")

    for ax in axes:
        set_spine(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=EVAL["n_episodes"])
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    n_ep = 200 if args.quick else args.n_episodes
    print(f"\n{'='*60}")
    print(f" RL Derivative Hedging — Evaluation ({n_ep} episodes)")
    print(f"{'='*60}\n")

    os.makedirs(EVAL["plot_dir"], exist_ok=True)

    # Env metadata
    env_tmp   = OptionsHedgingEnv(ENV)
    obs_dim   = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n
    del env_tmp

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("Running BLS baseline …")
    bls_d  = run_baseline("delta", n_ep, verbose=True)

    results = {
        "BLS-Delta" : bls_d,
    }

    # ── RL Agents ─────────────────────────────────────────────────────────────
    best_agent = None
    best_name  = None

    for name in ["dqn", "ppo", "sac"]:
        agent = load_agent(name, obs_dim, n_actions)
        if agent is None:
            print(f"  [{name.upper()}] No saved model found — skipping.")
            continue
        print(f"  [{name.upper()}] Evaluating …")
        res = rollout_agent(agent, name.upper(), n_ep)
        results[name.upper()] = res
        print(f"    Mean PnL={res['mean']:+.4f}  Std={res['std']:.4f}  "
              f"Sharpe={res['sharpe']:+.4f}")

        # track best for single-episode plot
        if best_agent is None or res["sharpe"] > results.get(best_name, {}).get("sharpe", -999):
            best_agent = agent
            best_name  = name.upper()

        # individual episode plot
        plot_single_episode(agent, name.upper(),
            os.path.join(EVAL["plot_dir"], f"episode_{name}.png"))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_cost_distributions(results,
        os.path.join(EVAL["plot_dir"], "cost_distributions.png"))
    plot_pnl_distributions(results,
        os.path.join(EVAL["plot_dir"], "pnl_boxplot.png"))
    plot_summary_table(results,
        os.path.join(EVAL["plot_dir"], "summary_table.png"))
    plot_training_curves(
        os.path.join(EVAL["plot_dir"], "training_curves.png"))

    print(f"\nAll plots saved to ./{EVAL['plot_dir']}/")

    # ── Console Summary ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Agent':<15} {'Mean PnL':>10} {'Std':>8} {'Sharpe':>8} {'P5':>8}")
    print(f"{'─'*60}")
    for label, res in results.items():
        p5 = np.percentile(res["pnl"], 5)
        print(f"{label:<15} {res['mean']:>+10.4f} {res['std']:>8.4f} "
              f"{res['sharpe']:>+8.4f} {p5:>+8.4f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
