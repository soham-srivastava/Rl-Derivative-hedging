"""
train.py — Training loop for DQN, PPO, and SAC on the Options Hedging env.

Usage:
    python train.py --agent dqn --steps 200000
    python train.py --agent ppo --steps 200000
    python train.py --agent sac --steps 200000
    python train.py --agent all --steps 200000   # train all sequentially
"""

import os
import argparse
import numpy as np
import torch
import time
from collections import deque

from config import ENV, DQN, PPO, SAC, EVAL
from env import OptionsHedgingEnv
from agents import DQNAgent, PPOAgent, SACAgent, DEVICE


def moving_avg(arr, n=50):
    return np.convolve(arr, np.ones(n)/n, mode='valid')


# ─────────────────────────────────────────────────────────────────────────────
def train_dqn(train_steps: int = None, verbose: bool = True):
    train_steps = train_steps or DQN["train_steps"]
    env   = OptionsHedgingEnv(ENV)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, DQN)
    os.makedirs(EVAL["model_dir"], exist_ok=True)
    os.makedirs(EVAL["log_dir"],   exist_ok=True)

    ep_rewards, ep_pnls = [], []
    recent = deque(maxlen=100)
    obs, _ = env.reset(seed=0)
    ep_reward = 0
    step = 0
    ep = 0
    t0 = time.time()

    print(f"[DQN] Training on {DEVICE} for {train_steps:,} steps …")

    while step < train_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        metrics = agent.update()
        ep_reward += reward
        obs = next_obs
        step += 1

        if done:
            ep += 1
            agent.decay_epsilon()
            ep_rewards.append(ep_reward)
            ep_pnls.append(env.running_pnl)
            recent.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

            if verbose and ep % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Ep {ep:4d} | Step {step:7,d} | "
                      f"Avg Reward(100) {np.mean(recent):+.4f} | "
                      f"ε={agent.eps:.3f} | {elapsed:.0f}s")

    path = os.path.join(EVAL["model_dir"], "dqn.pt")
    agent.save(path)
    np.save(os.path.join(EVAL["log_dir"], "dqn_rewards.npy"), np.array(ep_rewards))
    np.save(os.path.join(EVAL["log_dir"], "dqn_pnls.npy"),    np.array(ep_pnls))
    print(f"[DQN] Done. Model → {path}")
    return agent, ep_rewards, ep_pnls


# ─────────────────────────────────────────────────────────────────────────────
def train_ppo(train_steps: int = None, verbose: bool = True):
    train_steps = train_steps or PPO["train_steps"]
    env   = OptionsHedgingEnv(ENV)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(obs_dim, n_actions, PPO)
    os.makedirs(EVAL["model_dir"], exist_ok=True)
    os.makedirs(EVAL["log_dir"],   exist_ok=True)

    n_steps = PPO["n_steps"]
    ep_rewards, ep_pnls = [], []
    recent = deque(maxlen=100)
    obs, _ = env.reset(seed=0)
    ep_reward = 0
    step = 0
    ep = 0
    t0 = time.time()

    print(f"[PPO] Training on {DEVICE} for {train_steps:,} steps …")

    while step < train_steps:
        rollout_steps = 0
        while rollout_steps < n_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store(obs, action, reward, done)
            ep_reward += reward
            obs = next_obs
            step += 1
            rollout_steps += 1

            if done:
                ep += 1
                ep_rewards.append(ep_reward)
                ep_pnls.append(env.running_pnl)
                recent.append(ep_reward)
                ep_reward = 0
                obs, _ = env.reset()

        metrics = agent.update(last_obs=obs, last_done=False)

        if verbose and ep % 50 == 0 and ep > 0:
            elapsed = time.time() - t0
            print(f"  Ep {ep:4d} | Step {step:7,d} | "
                  f"Avg Reward(100) {np.mean(recent):+.4f} | "
                  f"pg={metrics.get('pg_loss',0):.4f} | {elapsed:.0f}s")

    path = os.path.join(EVAL["model_dir"], "ppo.pt")
    agent.save(path)
    np.save(os.path.join(EVAL["log_dir"], "ppo_rewards.npy"), np.array(ep_rewards))
    np.save(os.path.join(EVAL["log_dir"], "ppo_pnls.npy"),    np.array(ep_pnls))
    print(f"[PPO] Done. Model → {path}")
    return agent, ep_rewards, ep_pnls


# ─────────────────────────────────────────────────────────────────────────────
def train_sac(train_steps: int = None, verbose: bool = True):
    train_steps = train_steps or SAC["train_steps"]
    env   = OptionsHedgingEnv(ENV)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = SACAgent(obs_dim, n_actions, SAC)
    os.makedirs(EVAL["model_dir"], exist_ok=True)
    os.makedirs(EVAL["log_dir"],   exist_ok=True)

    ep_rewards, ep_pnls = [], []
    recent = deque(maxlen=100)
    obs, _ = env.reset(seed=0)
    ep_reward = 0
    step = 0
    ep = 0
    t0 = time.time()

    print(f"[SAC] Training on {DEVICE} for {train_steps:,} steps …")

    while step < train_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.update()
        ep_reward += reward
        obs = next_obs
        step += 1

        if done:
            ep += 1
            ep_rewards.append(ep_reward)
            ep_pnls.append(env.running_pnl)
            recent.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

            if verbose and ep % 50 == 0:
                elapsed = time.time() - t0
                alpha = agent._alpha.item() if agent.auto_alpha else agent.alpha
                print(f"  Ep {ep:4d} | Step {step:7,d} | "
                      f"Avg Reward(100) {np.mean(recent):+.4f} | "
                      f"α={alpha:.4f} | {elapsed:.0f}s")

    path = os.path.join(EVAL["model_dir"], "sac.pt")
    agent.save(path)
    np.save(os.path.join(EVAL["log_dir"], "sac_rewards.npy"), np.array(ep_rewards))
    np.save(os.path.join(EVAL["log_dir"], "sac_pnls.npy"),    np.array(ep_pnls))
    print(f"[SAC] Done. Model → {path}")
    return agent, ep_rewards, ep_pnls


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RL hedging agents")
    parser.add_argument("--agent",  type=str, default="all",
                        choices=["dqn", "ppo", "sac", "all"])
    parser.add_argument("--steps",  type=int, default=None,
                        help="Override train_steps from config")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    v = not args.quiet
    if args.agent in ("dqn", "all"):
        train_dqn(args.steps, verbose=v)
    if args.agent in ("ppo", "all"):
        train_ppo(args.steps, verbose=v)
    if args.agent in ("sac", "all"):
        train_sac(args.steps, verbose=v)

    print("\nAll requested agents trained. Run `python evaluate.py` for comparison plots.")


if __name__ == "__main__":
    main()
