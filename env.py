"""
env.py — Options Hedging Environment (Gymnasium-compatible)

State  : [S/K, tau, delta_bls, gamma_bls, current_position, pnl_running, iv]
Action : discrete lots to trade  ∈ {-n_lots_max, ..., +n_lots_max}
         (2*n_lots_max + 1 actions total)

The agent holds a SHORT call position of 1 contract (100 shares notional).
It trades the underlying to hedge. Reward = Δportfolio_value − |TC|.

Price dynamics: Geometric Brownian Motion with optional stochastic vol (Heston).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
from config import ENV

# call defined
def bs_price(S, K, tau, r, sigma):
    """Black-Scholes call price. Returns (price, delta, gamma, vega)."""
    if tau <= 0:
        price = max(S - K, 0.0)
        return price, float(S >= K), 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    vega  = S * norm.pdf(d1) * np.sqrt(tau)
    return price, delta, gamma, vega


class OptionsHedgingEnv(gym.Env):
    """
    Hedge a short European call option by trading the underlying.

    Observation (7-dim, all normalised):
        0 : moneyness          S/K  (≈ 0.7–1.4)
        1 : time-to-expiry     τ/T  (1 → 0)
        2 : BS delta           (0–1)
        3 : BS gamma × S       (dimensionless sensitivity) # why multiply by S?
        4 : current position   / n_lots_max  (-1 to +1)
        5 : running PnL        / S0          (clipped ±1) # why is it clipped?
        6 : realised vol proxy (20-day)      / σ_ref what is sigma ref?
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict = None, render_mode=None):
        super().__init__()
        cfg = cfg or ENV
        self.S0       = cfg["S0"]
        self.K        = cfg["K"]
        self.T        = cfg["T"]
        self.r        = cfg["r"]
        self.sigma    = cfg["sigma"]
        self.dt       = cfg["dt"]
        self.tc       = cfg["tc"]
        self.lot      = cfg["lot_size"]
        self.n_max    = cfg["n_lots_max"]
        self.rew_type = cfg["reward_type"]
        self.render_mode = render_mode

        n_actions = 2 * self.n_max + 1          # −10..+10 discrete lots
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self._vol_window = []
        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tau(self):
        """Remaining time in years."""
        return max((self.T - self.step_idx) * self.dt, 1e-8)

    def _obs(self):
        S, K = self.S, self.K
        tau  = self._tau()
        _, delta, gamma, _ = bs_price(S, K, tau, self.r, self.sigma)

        # realised vol proxy from recent log-returns
        rv = self.sigma  # fallback
        if len(self._vol_window) >= 5:
            rv = np.std(self._vol_window[-20:]) / np.sqrt(self.dt)
            rv = np.clip(rv, 0.01, 2.0) # why clipping is required?

        obs = np.array([
            S / self.K,
            tau / (self.T * self.dt),
            delta,
            gamma * S,
            self.position / self.n_max,
            np.clip(self.running_pnl / self.S0, -1.0, 1.0),
            rv / self.sigma,
        ], dtype=np.float32)
        return obs

    def _step_price(self):
        """GBM step."""
        z = self.rng.standard_normal()
        self.S *= np.exp((self.r - 0.5 * self.sigma**2) * self.dt
                         + self.sigma * np.sqrt(self.dt) * z)
        lr = np.log(self.S / self._S_prev) if self._S_prev > 0 else 0.0
        self._vol_window.append(lr)
        self._S_prev = self.S

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.S           = self.S0
        self._S_prev     = self.S0
        self.step_idx    = 0
        self.position    = 0          # shares held (+ = long)
        self.running_pnl = 0.0
        self._vol_window = []

        # value of the short call at inception (our liability)
        self.option_value_prev, _, _, _ = bs_price(
            self.S, self.K, self._tau(), self.r, self.sigma
        )
        self._cash = 0.0              # cash from trading hedge

        self.history = dict(S=[], position=[], pnl=[], reward=[])
        return self._obs(), {}

    def step(self, action):
        # action ∈ {0, …, 2*n_max} → trade_lots ∈ {-n_max, …, n_max}
        trade_lots = int(action) - self.n_max
        trade_shares = trade_lots * self.lot

        # --- Execute trade ---
        S_before = self.S
        tc_cost  = abs(trade_shares) * S_before * self.tc
        self._cash      -= trade_shares * S_before + tc_cost
        self.position   += trade_shares

        # clip position
        if abs(self.position) > self.n_max * self.lot:
            excess = self.position - np.sign(self.position) * self.n_max * self.lot
            self._cash   += excess * S_before
            self.position -= excess

        # --- Evolve price ---
        self._step_price()
        self.step_idx += 1
        done = (self.step_idx >= self.T)

        # --- Portfolio P&L ---
        # hedge book: long position gains/losses
        hedge_pnl = self.position * (self.S - S_before)

        # option book: short call — its value change is our loss
        tau_new = self._tau()
        opt_val_new, _, _, _ = bs_price(self.S, self.K, tau_new, self.r, self.sigma)

        if done:
            # at expiry, option settled at intrinsic
            opt_val_new = max(self.S - self.K, 0.0)

        option_pnl = -(opt_val_new - self.option_value_prev)   # short = flip sign
        self.option_value_prev = opt_val_new

        step_pnl = hedge_pnl + option_pnl - tc_cost
        self.running_pnl += step_pnl

        # --- Reward ---
        if self.rew_type == "pnl_tc":
            reward = step_pnl / self.S0
        elif self.rew_type == "sharpe":
            # running Sharpe proxy (penalise variance)
            reward = step_pnl / self.S0 - 0.5 * (step_pnl / self.S0) ** 2
        else:  # cvar — penalise large losses
            reward = step_pnl / self.S0 - 0.1 * max(-step_pnl / self.S0, 0) ** 2

        # terminal liquidation penalty
        if done and self.position != 0:
            liq_cost = abs(self.position) * self.S * self.tc
            reward  -= liq_cost / self.S0

        # Record
        self.history["S"].append(self.S)
        self.history["position"].append(self.position)
        self.history["pnl"].append(self.running_pnl)
        self.history["reward"].append(reward)

        return self._obs(), float(reward), done, False, {}

    def render(self):
        print(f"Step {self.step_idx:3d} | S={self.S:.2f} | pos={self.position:+.0f} "
              f"| PnL={self.running_pnl:+.4f}")
