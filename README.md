# RL Derivative Hedging

Training reinforcement learning agents to hedge a short options position  and showing they beat the standard Black-Scholes formula when market conditions get realistic.

---

## What this does

When you sell an options contract, you need to hedge your exposure by trading the underlying stock. The standard way to do this is Black-Scholes delta hedging  a formula that tells you exactly how many shares to hold at each moment.

The formula works well when its assumptions hold. The problem is they never fully hold in real markets  volatility is not constant, transaction costs exist, and the model is always at least slightly wrong.

This project trains three RL agents to learn the hedging strategy from scratch, without being given any formula. They interact with a simulated market for thousands of episodes and figure out what position minimises hedging cost.

---

## Results

| Agent | Sharpe | Mean Hedging Cost | vs BLS-Delta |
|---|---|---|---|
| **PPO** | **−0.195** | **−17.6** | **36% cheaper** |
| DQN | −0.288 | −26.2 | 5% cheaper |
| BLS-Delta (baseline) | −0.335 | −27.5 | — |
| BLS-DeltaGamma | −0.354 | −29.1 | worse |
| SAC | −0.401 | −43.0 | worse at 15k steps |

> All PnLs are negative because this is cost minimisation, not profit — a perfect hedge costs zero, a bad hedge costs more. Lower absolute value = better.

**Key experiment results:**

- **Heston misspecification** — agents trained under simple GBM, tested under stochastic volatility. RL gained +0.155 Sharpe over BLS because it adapts; the formula can't.
- **Transaction cost sweep** — at 0.5% TC, RL Sharpe edge reached +1.67 by learning to halve its trading frequency. BLS rebalances every day regardless of cost.
- **Reward ablation** — simple PnL reward beat Sharpe-shaped and CVaR-shaped rewards on every metric. Fancy reward shaping destabilised training at low step counts.

---

## Files

```
config.py                       — all hyperparameters in one place
env.py                          — the options hedging gym environment
agents.py                       — DQN (C51), PPO, SAC implementations
bls_baseline.py                 — Black-Scholes delta and delta-gamma hedger
train.py                        — training loop for all three agents
evaluate.py                     — evaluation and plots vs BLS baseline
experiment1_heston.py           — Heston misspecification experiment
experiment2_tc_sweep.py         — transaction cost sensitivity sweep
experiment3_reward_ablation.py  — reward function ablation
experiment4_agent_comparison.py — full head-to-head comparison
```

---

## How to run

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Train all three agents**
```bash
python train.py --agent all --steps 200000
```

Or train one at a time:
```bash
python train.py --agent ppo --steps 200000
python train.py --agent dqn --steps 200000
python train.py --agent sac --steps 200000
```

**Evaluate and generate plots**
```bash
python evaluate.py
```

Plots saved to `plots/`. Models saved to `models/`.

**Run individual experiments**
```bash
python experiment1_heston.py
python experiment2_tc_sweep.py
python experiment3_reward_ablation.py
python experiment4_agent_comparison.py
```

---

## Environment

- **Underlying:** stock following GBM (S0=100, σ=20%, T=63 days)
- **Option:** short European call, K=100, settled at expiry
- **State:** 7-dim — moneyness, time to expiry, BS delta, BS gamma, current position, running PnL, realised vol
- **Actions:** 21 discrete — trade −10 to +10 lots of the underlying
- **Reward:** step PnL minus transaction costs, normalised by S0

---

## Agents

**DQN** — distributional C51, Double DQN, soft target updates, ε-greedy exploration

**PPO** — clipped surrogate objective, Generalised Advantage Estimation (λ=0.95), shared actor-critic

**SAC** — twin Q-networks, auto-tuned entropy temperature, discrete action variant

All hyperparameters in `config.py`.

---

## Tech stack

Python · PyTorch · Gymnasium · NumPy · SciPy · Matplotlib
