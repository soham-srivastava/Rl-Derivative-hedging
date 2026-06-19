"""
experiment1_heston.py
─────────────────────
Misspecification experiment:
  - Train SAC under GBM (what BLS assumes)
  - Test both SAC and BLS-Delta under Heston (stochastic vol, mean-reverting)
  - Show RL adapts; BLS breaks down

Heston dynamics:
  dS = r·S·dt + √V·S·dW_S
  dV = κ(θ−V)dt + ξ√V·dW_V       corr(dW_S, dW_V) = ρ
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from config import ENV, SAC, EVAL
from env import OptionsHedgingEnv
from agents import SACAgent
from bls_baseline import BLSDeltaHedger

os.makedirs(EVAL["plot_dir"],  exist_ok=True)
os.makedirs(EVAL["model_dir"], exist_ok=True)

# ── Heston Environment ────────────────────────────────────────────────────────
class HestonHedgingEnv(OptionsHedgingEnv):
    """
    Drop-in replacement that uses Heston SV instead of GBM.
    Extra params: kappa, theta, xi, rho (passed via cfg).
    The agent's OBS still contains GBM-style delta (model misspecification).
    """
    def __init__(self, cfg=None):
        cfg = cfg or {}
        base = dict(ENV)
        base.update(cfg)
        super().__init__(base)
        self.kappa = base.get("kappa", 3.0)   # mean-reversion speed
        self.theta = base.get("theta", 0.04)  # long-run variance (≈ 20% vol)
        self.xi    = base.get("xi",    0.4)   # vol-of-vol
        self.rho   = base.get("rho",  -0.7)   # spot-vol correlation

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.V = self.sigma ** 2   # start at GBM implied variance
        return obs, info

    def _step_price(self):
        """Heston Euler-Maruyama step."""
        dt  = self.dt
        z1  = self.rng.standard_normal()
        z2  = self.rng.standard_normal()
        w_S = z1
        w_V = self.rho * z1 + np.sqrt(max(1 - self.rho**2, 0)) * z2

        # Full truncation to keep variance non-negative
        V_pos = max(self.V, 0.0)
        dV    = self.kappa * (self.theta - V_pos) * dt \
                + self.xi * np.sqrt(V_pos * dt) * w_V
        self.V = max(V_pos + dV, 1e-8)

        self.S *= np.exp((self.r - 0.5 * V_pos) * dt
                         + np.sqrt(V_pos * dt) * w_S)

        lr = np.log(self.S / self._S_prev) if self._S_prev > 0 else 0.0
        self._vol_window.append(lr)
        self._S_prev = self.S


# ── Helpers ───────────────────────────────────────────────────────────────────
def quick_train_sac(env_cls, env_cfg, steps=80_000, seed=0, label="SAC"):
    env = env_cls(env_cfg)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    agent = SACAgent(obs_dim, n_act, SAC)
    obs, _ = env.reset(seed=seed)
    t0, ep = time.time(), 0
    for s in range(steps):
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.update()
        obs = next_obs
        if done:
            ep += 1
            obs, _ = env.reset()
    elapsed = time.time() - t0
    print(f"  [{label}] {steps:,} steps | {ep} eps | {elapsed:.0f}s")
    return agent


def rollout(agent_or_hedger, env_cls, env_cfg, n=500, seed=42, is_bls=False):
    rng = np.random.default_rng(seed)
    pnls, vols = [], []
    for _ in range(n):
        env = env_cls(env_cfg)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        if is_bls:
            hedger = BLSDeltaHedger(env, "delta")
        while True:
            if is_bls:
                action = hedger.select_action(obs)
            else:
                action = agent_or_hedger.select_action(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        pnls.append(env.running_pnl)
        # realised vol of the episode
        rets = np.diff(np.log(env.history["S"]))
        vols.append(np.std(rets) / np.sqrt(env.dt) if len(rets) > 1 else env.sigma)
    return np.array(pnls), np.array(vols)


# ── Main experiment ───────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print(" EXPERIMENT 1 — Heston Misspecification")
    print("="*60)

    gbm_cfg    = dict(ENV)
    heston_cfg = dict(ENV, kappa=3.0, theta=0.04, xi=0.4, rho=-0.7)

    # Train SAC on GBM (same world BLS assumes)
    print("\n[1/2] Training SAC on GBM …")
    sac_gbm = quick_train_sac(OptionsHedgingEnv, gbm_cfg, steps=80_000,
                              label="SAC-GBM")

    # Test both on Heston (out-of-distribution)
    print("[2/2] Evaluating on Heston (OOD) …")
    pnl_sac,  vol_sac  = rollout(sac_gbm,    HestonHedgingEnv, heston_cfg,
                                  n=500, is_bls=False)
    pnl_bls,  vol_bls  = rollout(None,        HestonHedgingEnv, heston_cfg,
                                  n=500, is_bls=True)

    def stats(arr):
        return dict(mean=np.mean(arr), std=np.std(arr),
                    sharpe=np.mean(arr)/(np.std(arr)+1e-8),
                    p5=np.percentile(arr,5))

    s_sac = stats(pnl_sac)
    s_bls = stats(pnl_bls)

    print(f"\n  SAC (GBM-trained, Heston-tested) : "
          f"μ={s_sac['mean']:+.3f}  σ={s_sac['std']:.3f}  "
          f"Sharpe={s_sac['sharpe']:+.3f}  P5={s_sac['p5']:+.3f}")
    print(f"  BLS-Delta (Heston-tested)         : "
          f"μ={s_bls['mean']:+.3f}  σ={s_bls['std']:.3f}  "
          f"Sharpe={s_bls['sharpe']:+.3f}  P5={s_bls['p5']:+.3f}")

    improvement_sharpe = s_sac['sharpe'] - s_bls['sharpe']
    improvement_std    = s_bls['std']    - s_sac['std']
    print(f"\n  RL Sharpe advantage : {improvement_sharpe:+.3f}")
    print(f"  RL Std reduction    : {improvement_std:+.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#08090c")
    for ax in axes:
        ax.set_facecolor("#0f1117")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2130")

    bins = np.linspace(-300, 200, 60)
    axes[0].hist(pnl_bls, bins=bins, alpha=0.6, color="#5c7ae0",
                 label=f"BLS-Δ  Sharpe={s_bls['sharpe']:+.3f}")
    axes[0].hist(pnl_sac, bins=bins, alpha=0.6, color="#e05c5c",
                 label=f"SAC    Sharpe={s_sac['sharpe']:+.3f}")
    axes[0].set_xlabel("Episode PnL"); axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution under Heston\n(Both trained/derived under GBM)",
                       color="#eef0f7")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Scatter: realised vol vs PnL
    axes[1].scatter(vol_bls*100, pnl_bls, alpha=0.3, s=8, color="#5c7ae0", label="BLS-Δ")
    axes[1].scatter(vol_sac*100, pnl_sac, alpha=0.3, s=8, color="#e05c5c", label="SAC")
    axes[1].axhline(0, color="#4a4f63", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Realised Vol (% ann.)"); axes[1].set_ylabel("Episode PnL")
    axes[1].set_title("PnL vs Realised Volatility\n(RL robust in high-vol regimes?)",
                       color="#eef0f7")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    for ax in axes: ax.tick_params(colors="#4a4f63")

    plt.suptitle("Experiment 1 — Heston Misspecification",
                 color="#eef0f7", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(EVAL["plot_dir"], "exp1_heston_misspecification.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")
    return s_sac, s_bls


if __name__ == "__main__":
    run()
