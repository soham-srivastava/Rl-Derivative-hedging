"""
bls_baseline.py — Black-Scholes Delta & Delta-Gamma Hedge Baseline

Runs the textbook BLS hedge on the same environment to produce a cost
distribution for comparison with RL agents.
"""

import numpy as np
from env import OptionsHedgingEnv
from config import ENV, EVAL


class BLSDeltaHedger:
    """
    Textbook delta hedge: rebalance to BLS delta every step.
    Action: set position = round(delta * n_lots_max) lots.
    """

    def __init__(self, env: OptionsHedgingEnv, mode: str = "delta"):
        """
        mode: "delta"       — pure delta hedge
              "delta_gamma" — delta + gamma (requires 2nd instrument; approximated
                              here by adjusting delta by ½ Γ·ΔS² correction)
        """
        self.env  = env
        self.mode = mode
        self.n_max = env.n_max

    def select_action(self, obs) -> int:
        """
        obs = [S/K, tau/T, delta, gamma*S, position/n_max, pnl/S0, rv/sigma]
        Returns discrete action ∈ {0 … 2*n_max}.
        """
        delta = float(obs[2])
        gamma_S = float(obs[3])          # gamma × S (dimensionless)

        if self.mode == "delta_gamma":
            # Crude gamma correction: shift delta by Γ·S·(ΔS estimate)
            # ΔS estimate ≈ σ·S·√dt
            S   = float(obs[0]) * self.env.K
            dS  = self.env.sigma * S * np.sqrt(self.env.dt)
            gamma = gamma_S / S if S > 0 else 0
            delta = delta + 0.5 * gamma * dS        # 2nd-order correction

        # Target position in shares → clip → round → convert to action
        target_shares = delta * self.n_max * self.env.lot
        target_lots   = int(np.round(np.clip(target_shares / self.env.lot,
                                             -self.n_max, self.n_max)))
        return target_lots + self.n_max   # shift to {0 … 2*n_max}


def run_baseline(mode: str = "delta", n_episodes: int = None,
                 seed: int = None, verbose: bool = False):
    """
    Roll out the BLS hedger for n_episodes episodes.
    Returns dict with arrays of episode statistics.
    """
    n_episodes = n_episodes or EVAL["n_episodes"]
    seed       = seed or EVAL["seed"]
    rng        = np.random.default_rng(seed)

    env    = OptionsHedgingEnv(ENV)
    hedger = BLSDeltaHedger(env, mode=mode)

    ep_pnl, ep_costs, ep_tc = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        total_tc     = 0.0
        prev_pos     = 0

        while True:
            action = hedger.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            # estimate TC this step
            trade = abs(env.position - prev_pos)
            total_tc += trade * env.S * env.tc
            prev_pos = env.position

            if done:
                break

        ep_pnl.append(env.running_pnl)
        ep_costs.append(-env.running_pnl)   # cost = -pnl for a hedger
        ep_tc.append(total_tc)

        if verbose and (ep + 1) % 100 == 0:
            print(f"[BLS-{mode}] Episode {ep+1}/{n_episodes} | "
                  f"Mean PnL={np.mean(ep_pnl):.4f} | "
                  f"Std={np.std(ep_pnl):.4f}")

    return {
        "pnl"    : np.array(ep_pnl),
        "cost"   : np.array(ep_costs),
        "tc"     : np.array(ep_tc),
        "mode"   : mode,
        "mean"   : np.mean(ep_pnl),
        "std"    : np.std(ep_pnl),
        "sharpe" : np.mean(ep_pnl) / (np.std(ep_pnl) + 1e-8),
    }


if __name__ == "__main__":
    print("Running BLS delta baseline …")
    res = run_baseline("delta", n_episodes=500, verbose=True)
    print(f"\nBLS Delta  → Mean: {res['mean']:.4f}  Std: {res['std']:.4f}  "
          f"Sharpe: {res['sharpe']:.4f}")

    print("\nRunning BLS delta-gamma baseline …")
    res2 = run_baseline("delta_gamma", n_episodes=500, verbose=True)
    print(f"BLS DeltaΓ → Mean: {res2['mean']:.4f}  Std: {res2['std']:.4f}  "
          f"Sharpe: {res2['sharpe']:.4f}")
