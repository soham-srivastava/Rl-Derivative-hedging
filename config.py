"""
config.py — Central hyperparameter store for RL Derivative Hedging system.
All environment, agent, and training parameters live here.
"""

#  Environment 
ENV = dict(
    S0          = 100.0,    # initial spot price
    K           = 100.0,    # strike price
    T           = 63,       # episode length in trading days (≈ 3 months)
    r           = 0.00,     # risk-free rate (annualised, set 0 for simplicity)
    sigma       = 0.20,     # GBM volatility (annualised)
    dt          = 1/252,    # one trading day
    tc          = 0.001,    # proportional transaction cost on notional traded
    lot_size    = 1,        # shares per lot
    n_lots_max  = 10,       # maximum absolute position in lots
    reward_type = "pnl_tc", # "pnl_tc" | "sharpe" | "cvar"
)

#  DQN 
DQN = dict(
    lr            = 3e-4,
    gamma         = 0.99,
    batch_size    = 256,
    buffer_size   = 100_000,
    tau           = 0.005,      # soft-update coefficient
    eps_start     = 1.0,
    eps_end       = 0.05,
    eps_decay     = 0.995,      # per episode
    hidden        = [256, 256],
    update_every  = 4,          # steps between gradient updates
    n_atoms       = 51,         # distributional (C51); set 1 for vanilla DQN
    v_min         = -10.0,
    v_max         = 10.0,
    train_steps   = 200_000,
)

#  PPO 
PPO = dict(
    lr            = 3e-4,
    gamma         = 0.99,
    lam           = 0.95,       # GAE lambda
    clip_eps      = 0.2,
    entropy_coef  = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    hidden        = [256, 256],
    n_steps       = 2048,       # rollout length before update
    n_epochs      = 10,         # gradient epochs per rollout
    batch_size    = 64,
    train_steps   = 200_000,
)

#  SAC 
SAC = dict(
    lr            = 3e-4,
    gamma         = 0.99,
    alpha         = 0.2,        # entropy temperature (auto-tuned if None)
    auto_alpha    = True,
    tau           = 0.005,
    batch_size    = 256,
    buffer_size   = 100_000,
    hidden        = [256, 256],
    update_every  = 1,
    train_steps   = 200_000,
)

#  Evaluation 
EVAL = dict(
    n_episodes  = 1000,
    seed        = 42,
    plot_dir    = "plots",
    model_dir   = "models",
    log_dir     = "logs",
)
