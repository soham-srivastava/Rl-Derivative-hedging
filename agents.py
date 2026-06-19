"""
agents.py — DQN (C51 distributional), PPO (GAE), SAC (auto-entropy)

All agents share a common interface:
    agent.select_action(obs, deterministic=False) → int
    agent.update(batch)                           → dict[str, float]
    agent.save(path) / agent.load(path)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from collections import deque, namedtuple
import random

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        obs      = torch.FloatTensor(np.stack([t.obs      for t in batch])).to(DEVICE)
        action   = torch.LongTensor( np.stack([t.action   for t in batch])).to(DEVICE)
        reward   = torch.FloatTensor(np.stack([t.reward   for t in batch])).to(DEVICE)
        next_obs = torch.FloatTensor(np.stack([t.next_obs for t in batch])).to(DEVICE)
        done     = torch.FloatTensor(np.stack([t.done     for t in batch])).to(DEVICE)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buf)


def mlp(dims: list, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(activation())
        elif output_activation:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# DQN — Distributional (C51)


class C51Network(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden, n_atoms, v_min, v_max):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        dims = [obs_dim] + hidden + [n_actions * n_atoms]
        self.net = mlp(dims)

    def forward(self, x):
        logits = self.net(x).view(-1, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)   # (B, A, N)

    def q_values(self, x):
        probs = self(x)                    # (B, A, N)
        return (probs * self.atoms).sum(-1) # (B, A)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.n_actions  = n_actions
        self.gamma      = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.tau        = cfg["tau"]
        self.update_every = cfg["update_every"]
        self.n_atoms    = cfg["n_atoms"]
        self.v_min      = cfg["v_min"]
        self.v_max      = cfg["v_max"]
        self.eps        = cfg["eps_start"]
        self.eps_end    = cfg["eps_end"]
        self.eps_decay  = cfg["eps_decay"]

        hidden = cfg["hidden"]
        mk = lambda: C51Network(obs_dim, n_actions, hidden,
                                self.n_atoms, self.v_min, self.v_max).to(DEVICE)
        self.online = mk()
        self.target = mk()
        self.target.load_state_dict(self.online.state_dict())

        self.opt    = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"])
        self._steps = 0
        atoms = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(DEVICE)
        self.register_atoms = atoms

    def select_action(self, obs, deterministic=False):
        if not deterministic and random.random() < self.eps:
            return random.randrange(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.online.q_values(obs_t)
        return q.argmax(1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, float(done))
        self._steps += 1

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}
        if self._steps % self.update_every != 0:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        B = obs.size(0)
        atoms = self.online.atoms  # (N,)
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        with torch.no_grad():
            # Double DQN: select action via online, evaluate via target
            next_q_online = self.online.q_values(next_obs)
            next_actions  = next_q_online.argmax(1)                # (B,)
            next_probs    = self.target(next_obs)                  # (B,A,N)
            next_probs    = next_probs[range(B), next_actions]     # (B,N)

            # Bellman projection
            Tz = (rewards.unsqueeze(1)
                  + (1 - dones.unsqueeze(1)) * self.gamma * atoms.unsqueeze(0))
            Tz = Tz.clamp(self.v_min, self.v_max)
            b  = (Tz - self.v_min) / delta_z
            l, u = b.floor().long(), b.ceil().long()
            l = l.clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)

            target_dist = torch.zeros_like(next_probs)
            offset = torch.arange(B, device=DEVICE).unsqueeze(1) * self.n_atoms
            target_dist.view(-1).scatter_add_(
                0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
            target_dist.view(-1).scatter_add_(
                0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))

        log_probs = torch.log(self.online(obs)[range(B), actions.squeeze()] + 1e-8)
        loss = -(target_dist * log_probs).sum(-1).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        soft_update(self.target, self.online, self.tau)
        return {"loss": loss.item()}

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])


# PPO — Clipped Surrogate + GAE

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.shared = mlp([obs_dim] + hidden, nn.Tanh)
        self.actor  = nn.Linear(hidden[-1], n_actions)
        self.critic = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.gamma      = cfg["gamma"]
        self.lam        = cfg["lam"]
        self.clip_eps   = cfg["clip_eps"]
        self.ent_coef   = cfg["entropy_coef"]
        self.vf_coef    = cfg["vf_coef"]
        self.max_grad   = cfg["max_grad_norm"]
        self.n_epochs   = cfg["n_epochs"]
        self.batch_size = cfg["batch_size"]

        self.ac  = ActorCritic(obs_dim, n_actions, cfg["hidden"]).to(DEVICE)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg["lr"])

        # rollout buffers
        self._reset_buffers()

    def _reset_buffers(self):
        self.buf = dict(obs=[], action=[], logp=[], reward=[], value=[], done=[])

    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, logp, _, value = self.ac.act(obs_t, deterministic)
        a = action.item()
        self._last = (logp.item(), value.item())
        return a

    def store(self, obs, action, reward, done):
        logp, value = self._last
        self.buf["obs"].append(obs)
        self.buf["action"].append(action)
        self.buf["logp"].append(logp)
        self.buf["reward"].append(reward)
        self.buf["value"].append(value)
        self.buf["done"].append(float(done))

    def update(self, last_obs=None, last_done=False):
        """Call after collecting n_steps. last_obs used for bootstrap."""
        if last_obs is not None and not last_done:
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, _, _, last_val = self.ac.act(obs_t)
            last_val = last_val.item()
        else:
            last_val = 0.0

        # GAE
        rewards  = self.buf["reward"]
        values   = self.buf["value"] + [last_val]
        dones    = self.buf["done"]
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1-dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.buf["value"], dtype=np.float32)

        # Tensors
        obs_t   = torch.FloatTensor(np.stack(self.buf["obs"])).to(DEVICE)
        act_t   = torch.LongTensor(self.buf["action"]).to(DEVICE)
        logp_t  = torch.FloatTensor(self.buf["logp"]).to(DEVICE)
        adv_t   = torch.FloatTensor(advantages).to(DEVICE)
        ret_t   = torch.FloatTensor(returns).to(DEVICE)
        adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        metrics = {"pg_loss": 0, "vf_loss": 0, "entropy": 0}
        for _ in range(self.n_epochs):
            idx = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                b = idx[start:start+self.batch_size]
                new_action, new_logp, ent, new_val = self.ac.act(obs_t[b])
                # re-evaluate with current policy
                from torch.distributions import Categorical
                logits, new_val2 = self.ac(obs_t[b])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[b])
                ent      = dist.entropy().mean()

                ratio = (new_logp - logp_t[b]).exp()
                pg1 = ratio * adv_t[b]
                pg2 = ratio.clamp(1-self.clip_eps, 1+self.clip_eps) * adv_t[b]
                pg_loss = -torch.min(pg1, pg2).mean()
                vf_loss = F.mse_loss(new_val2.squeeze(), ret_t[b])
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
                self.opt.step()

                metrics["pg_loss"] += pg_loss.item()
                metrics["vf_loss"] += vf_loss.item()
                metrics["entropy"] += ent.item()

        self._reset_buffers()
        return metrics

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.ac.state_dict(), path)

    def load(self, path):
        self.ac.load_state_dict(torch.load(path, map_location=DEVICE))



# SAC — Soft Actor-Critic (discrete action space variant)


class SACPolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = mlp([obs_dim] + hidden + [n_actions])

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)   # action probs

    def log_probs_and_entropy(self, x):
        probs = self(x)
        log_probs = torch.log(probs + 1e-8)
        entropy   = -(probs * log_probs).sum(-1, keepdim=True)
        return probs, log_probs, entropy


class SACQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = mlp([obs_dim] + hidden + [n_actions])

    def forward(self, x):
        return self.net(x)   # Q(s,·) for all actions


class SACAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.n_actions  = n_actions
        self.gamma      = cfg["gamma"]
        self.tau        = cfg["tau"]
        self.batch_size = cfg["batch_size"]
        self.update_every = cfg["update_every"]
        self.auto_alpha = cfg["auto_alpha"]

        hidden = cfg["hidden"]
        self.policy  = SACPolicyNet(obs_dim, n_actions, hidden).to(DEVICE)
        self.q1      = SACQNet(obs_dim, n_actions, hidden).to(DEVICE)
        self.q2      = SACQNet(obs_dim, n_actions, hidden).to(DEVICE)
        self.q1_tgt  = SACQNet(obs_dim, n_actions, hidden).to(DEVICE)
        self.q2_tgt  = SACQNet(obs_dim, n_actions, hidden).to(DEVICE)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

        self.opt_p  = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr"])
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=cfg["lr"])
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=cfg["lr"])

        if self.auto_alpha:
            self.target_entropy = -0.98 * math.log(1.0 / n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg["alpha"]

        self.buffer = ReplayBuffer(cfg["buffer_size"])
        self._steps = 0

    @property
    def _alpha(self):
        return self.log_alpha.exp() if self.auto_alpha else self.alpha

    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.policy(obs_t)
        if deterministic:
            return probs.argmax(1).item()
        return Categorical(probs=probs).sample().item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, float(done))
        self._steps += 1

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}
        if self._steps % self.update_every != 0:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_probs, next_log_probs, next_ent = \
                self.policy.log_probs_and_entropy(next_obs)
            q1_next = self.q1_tgt(next_obs)
            q2_next = self.q2_tgt(next_obs)
            min_q   = torch.min(q1_next, q2_next)
            # soft value:  V(s') = Σ π(a'|s') [Q(s',a') - α log π(a'|s')]
            v_next  = (next_probs * (min_q - self._alpha * next_log_probs)).sum(-1)
            target  = rewards + (1 - dones) * self.gamma * v_next

        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        self.opt_q1.zero_grad(); q1_loss.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); q2_loss.backward(); self.opt_q2.step()

        # Policy update
        probs, log_probs, entropy = self.policy.log_probs_and_entropy(obs)
        with torch.no_grad():
            q_pi = torch.min(self.q1(obs), self.q2(obs))
        # maximise expected Q - α H
        p_loss = (probs * (self._alpha * log_probs - q_pi)).sum(-1).mean()

        self.opt_p.zero_grad(); p_loss.backward(); self.opt_p.step()

        # Alpha update
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        soft_update(self.q1_tgt, self.q1, self.tau)
        soft_update(self.q2_tgt, self.q2, self.tau)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "p_loss" : p_loss.item(),
            "alpha"  : self._alpha.item() if self.auto_alpha else self.alpha,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha if self.auto_alpha else None,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(ck["policy"])
        self.q1.load_state_dict(ck["q1"]); self.q2.load_state_dict(ck["q2"])
        self.q1_tgt.load_state_dict(ck["q1"]); self.q2_tgt.load_state_dict(ck["q2"])
