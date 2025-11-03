# algos/deep_triple_q.py
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn.functional as F

from algos.components.actor import Actor
from algos.components.critic import TwinCritic
from algos.components.replay_buffer import ReplayBuffer
from algos.components.rnd import RNDTarget, RNDPredictor
from algos.components.safety_filter import project_to_safe
from algos.components.utils import set_seed, to_tensor, soft_update


@dataclass
class TripleQStats:
    ep: int
    ret: float
    len: int
    util: float
    Z: float
    rho: float
    violations: int
    success: int


class DeepTripleQ:
    """
    Deep Triple‑Q (neural adaptation của Triple‑Q cho CMDP):
      - Hai critic cho reward (Qr) và utility (Qc)
      - Actor tối đa hóa pseudo‑Q:  Qr + (Z/eta)*Qc
      - Virtual queue Z cập nhật theo "frame" các episode để ép ràng buộc an toàn
    """
    def __init__(self, env, cfg: Dict[str, Any], device="cpu"):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)

        set_seed(cfg["algo"]["seed"])

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_low = self.env.action_space.low
        act_high = self.env.action_space.high
        self.obs_dim, self.act_dim = obs_dim, act_dim

        # --- Networks ---
        self.actor = Actor(obs_dim, act_dim, act_low, act_high).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, act_low, act_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.qr = TwinCritic(obs_dim, act_dim).to(self.device)
        self.qr_target = TwinCritic(obs_dim, act_dim).to(self.device)
        self.qr_target.load_state_dict(self.qr.state_dict())

        self.qc = TwinCritic(obs_dim, act_dim).to(self.device)
        self.qc_target = TwinCritic(obs_dim, act_dim).to(self.device)
        self.qc_target.load_state_dict(self.qc.state_dict())

        actor_lr = float(cfg["algo"].get("actor_lr", 1e-3))
        critic_lr = float(cfg["algo"].get("critic_lr", 1e-3))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qr_opt = torch.optim.Adam(self.qr.parameters(), lr=critic_lr)
        self.qc_opt = torch.optim.Adam(self.qc.parameters(), lr=critic_lr)

        # --- Replay ---
        self.buf = ReplayBuffer(cfg["algo"]["buffer_capacity"], obs_dim, act_dim)

        # --- RND (intrinsic exploration) ---
        self.rnd_enabled = cfg.get("rnd", {}).get("enabled", False)
        if self.rnd_enabled:
            self.rnd_t = RNDTarget(obs_dim).to(self.device)
            self.rnd_p = RNDPredictor(obs_dim).to(self.device)
            self.rnd_opt = torch.optim.Adam(self.rnd_p.parameters(), lr=cfg["rnd"]["lr"])
            self.rnd_scale = float(cfg["rnd"]["scale"])

        # --- Safety filter (shield) ---
        self.filter_enabled = cfg.get("safety_filter", {}).get("enabled", True)
        self.filter_margin = float(cfg.get("safety_filter", {}).get("margin", 0.3))
        self.filter_hard_stop = bool(cfg.get("safety_filter", {}).get("hard_stop", True))

        # --- Discount & TD3 hyperparams ---
        self.gamma = float(cfg["algo"]["gamma"])
        self.tau = float(cfg["algo"]["tau"])
        self.batch_size = int(cfg["algo"]["batch_size"])
        self.warmup = int(cfg["algo"]["warmup_steps"])
        self.noise_std = float(cfg["algo"]["noise_std"])
        self.start_noise_std = float(cfg["algo"]["start_noise_std"])
        self.policy_delay = int(cfg["algo"]["policy_delay"])
        self.target_policy_noise = float(cfg["algo"]["target_policy_noise"])
        self.target_policy_noise_clip = float(cfg["algo"]["target_policy_noise_clip"])

        # --- Frames & constraint (virtual queue) ---
        H = int(self.env.max_steps)
        rho_frac = float(cfg["constraint"]["rho_frac"])
        self.rho = rho_frac * H               # target utility per-episode
        self.epsilon = float(cfg["constraint"]["epsilon"])
        self.episodes_per_frame = int(cfg["algo"]["episodes_per_frame"])
        self.eta = float(cfg["algo"]["eta"])
        self.Z = 0.0                          # virtual queue

        # --- Logging ---
        run_name = cfg["algo"]["run_name"]
        self.run_dir = os.path.join("out", run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("episode,return,length,utility,violations,success,Z,rho\n")

        self.global_step = 0

    # -------------------- Acting --------------------
    def select_action(self, obs, noise_std):
        self.actor.eval()
        with torch.no_grad():
            obs_t = to_tensor(obs, self.device).unsqueeze(0)
            act = self.actor.act(obs_t, deterministic=False)[0].cpu().numpy()
        self.actor.train()
        if noise_std > 0:
            act = act + noise_std * np.random.randn(*act.shape).astype(np.float32)
        low, high = self.env.action_space.low, self.env.action_space.high
        return np.clip(act, low, high)

    # -------------------- RND --------------------
    def compute_intrinsic(self, obs_batch):
        with torch.no_grad():
            t = self.rnd_t(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
        p = self.rnd_p(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
        loss = F.mse_loss(p, t, reduction="none").mean(dim=1, keepdim=True)  # [B,1]
        self.rnd_opt.zero_grad(set_to_none=True)
        F.mse_loss(p, t).backward()
        self.rnd_opt.step()
        return loss.detach()

    # -------------------- Learning --------------------
    def update(self):
        if self.buf.size() < self.batch_size:
            return {}

        obs, act, rew, util, next_obs, done = self.buf.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        util_t = torch.as_tensor(util, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        if self.rnd_enabled:
            with torch.no_grad():
                intr = self.compute_intrinsic(obs) * self.rnd_scale
            rew_t = rew_t + torch.as_tensor(intr, dtype=torch.float32, device=self.device)

        # ----- Critic targets (TD3 style) -----
        with torch.no_grad():
            mu, std = self.actor_target(next_obs_t)
            noise = (torch.randn_like(mu) * self.target_policy_noise
                    ).clamp(-self.target_policy_noise_clip, self.target_policy_noise_clip)
            next_a = torch.tanh(mu + noise)
            act_low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
            act_high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
            next_a = (next_a + 1) * 0.5 * (act_high - act_low) + act_low

            qr_next = self.qr_target.q_min(next_obs_t, next_a)
            qc_next = self.qc_target.q_min(next_obs_t, next_a)
            target_r = rew_t + (1.0 - done_t) * self.gamma * qr_next
            target_c = util_t + (1.0 - done_t) * self.gamma * qc_next
            target_r = target_r.detach()
            target_c = target_c.detach()

        # ----- Update reward critics -----
        q1, q2 = self.qr(obs_t, act_t)
        loss_qr = F.mse_loss(q1, target_r) + F.mse_loss(q2, target_r)
        self.qr_opt.zero_grad(set_to_none=True)
        loss_qr.backward()
        self.qr_opt.step()

        # ----- Update utility critics -----
        c1, c2 = self.qc(obs_t, act_t)
        loss_qc = F.mse_loss(c1, target_c) + F.mse_loss(c2, target_c)
        self.qc_opt.zero_grad(set_to_none=True)
        loss_qc.backward()
        self.qc_opt.step()

        info = {"loss_qr": float(loss_qr.item()), "loss_qc": float(loss_qc.item())}

        # ----- Delayed actor & target updates -----
        if self.global_step % self.policy_delay == 0:
            mu, _ = self.actor(obs_t)
            a = torch.tanh(mu)
            act_low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
            act_high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
            a = (a + 1) * 0.5 * (act_high - act_low) + act_low

            q_r = self.qr.q_min(obs_t, a)
            q_c = self.qc.q_min(obs_t, a)
            pseudo = q_r + (self.Z / max(1e-6, self.eta)) * q_c
            actor_loss = (-pseudo).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = float(actor_loss.item())

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.qr_target, self.qr, self.tau)
            soft_update(self.qc_target, self.qc, self.tau)

        return info

    # -------------------- Train loop --------------------
    def train(self):
        episodes = int(self.cfg["algo"]["episodes"])
        noise_std = self.start_noise_std
        frame_returns, frame_utils = [], []
        frame_success, frame_viol = 0, 0

        for ep in range(1, episodes + 1):
            obs, _ = self.env.reset()
            ep_ret, ep_len, ep_util = 0.0, 0, 0.0
            reached, viol = False, False

            while True:
                a = self.select_action(obs, noise_std)

                if self.filter_enabled:
                    state_xy = obs[0:2]
                    dist = obs[-1]  # min distance to obstacle
                    a[0:2] = project_to_safe(state_xy, a[0:2], dist,
                                             margin=self.filter_margin, hard_stop=self.filter_hard_stop)

                next_obs, r, term, trunc, info = self.env.step(a)
                u = info.get("utility", 1.0)
                done = term or trunc

                self.buf.add(obs, a, r, u, next_obs, done)
                obs = next_obs
                ep_ret += r
                ep_len += 1
                ep_util += u
                reached = reached or bool(info.get("reached", False))
                viol = viol or bool(info.get("collided", False))

                if self.global_step > self.warmup:
                    _ = self.update()
                    noise_std = self.noise_std
                self.global_step += 1

                if done:
                    break

            # frame collections
            frame_returns.append(ep_ret)
            frame_utils.append(ep_util)
            frame_success += int(reached)
            frame_viol += int(viol)

            # frame boundary -> update Z
            if ep % self.episodes_per_frame == 0:
                import numpy as _np
                mean_utility = float(_np.mean(frame_utils[-self.episodes_per_frame:]))
                self.Z = max(0.0, self.Z + self.rho + self.epsilon - mean_utility)

                # write a metrics row for last ep in frame
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{ep},{ep_ret:.3f},{ep_len},{ep_util:.3f},{frame_viol},{frame_success},{self.Z:.4f},{self.rho:.3f}\n")

            # save model
            if ep % self.cfg["algo"]["save_interval_episodes"] == 0 or ep == episodes:
                self.save(os.path.join(self.run_dir, "agent.pt"))

        print("Training finished. Checkpoints in:", self.run_dir)

    # -------------------- Save/Load --------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "qr": self.qr.state_dict(),
            "qc": self.qc.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "qr_target": self.qr_target.state_dict(),
            "qc_target": self.qc_target.state_dict(),
            "Z": self.Z, "rho": self.rho, "eta": self.eta,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.qr.load_state_dict(ckpt["qr"])
        self.qc.load_state_dict(ckpt["qc"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.qr_target.load_state_dict(ckpt["qr_target"])
        self.qc_target.load_state_dict(ckpt["qc_target"])
        self.Z = ckpt.get("Z", 0.0)
        self.rho = ckpt.get("rho", self.rho)
        self.eta = ckpt.get("eta", self.eta)


__all__ = ["DeepTripleQ"]
