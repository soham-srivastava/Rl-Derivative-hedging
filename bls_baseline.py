"""
bls_baseline.py — Black-Scholes Delta Hedge Baseline

Runs the textbook BLS hedge on the same environment to produce a cost
distribution for comparison with RL agents.
"""

import numpy as np
from env import OptionsHedgingEnv
from config import ENV, EVAL


class BLSDeltaHedger:
    """
    Textbook delta hedge: rebalance to BLS delta every step.
    Action: trade enough lots so final position matches rounded BLS delta target.
    """

    def __init__(self, env: OptionsHedgingEnv, mode: str = "delta"):
        """
        mode: "delta" — pure delta hedge (only mode currently implemented)
        """
        self.env  = env
        self.mode = mode
        self.n_max = env.n_max

    def select_action(self, obs) -> int:
        """
        obs = [S/K, tau/T, delta, gamma*S, position/n_max, pnl/S0, rv/sigma]
        Returns discrete action ∈ {0 … 2*n_max}, where env.step() interprets
        action as a TRADE of (action - n_max) lots, not a target position.
        """
        delta = float(obs[2])

        # Desired final position in lots → clip → round
        desired_lots = int(np.round(np.clip(delta * self.n_max,
                                            -self.n_max, self.n_max)))

        # Convert desired position into the trade required to get there
        current_lots = int(np.round(self.env.position / self.env.lot))
        trade_lots   = int(np.clip(desired_lots - current_lots,
                                   -self.n_max, self.n_max))

        return trade_lots + self.n_max   # shift to {0 … 2*n_max}


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

        while True:
            action = hedger.select_action(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

            # exact TC this step, as charged inside env.step()
            total_tc += info["tc_cost"]

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
