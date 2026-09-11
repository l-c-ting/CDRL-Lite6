import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agents.networks import Actor, ImageFeatureExtractor, TwinQNetwork
from buffers import ReplayBuffer


class TD3:
    """TD3 agent with a shared image encoder."""

    def __init__(self, args, image_size, robot_feature_dim: int, action_dim: int, device):
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.update_count = 0

        image_shape = (3, *image_size)
        self.encoder = ImageFeatureExtractor(
            image_shape=image_shape,
            robot_feature_dim=robot_feature_dim,
            output_dim=args.cnn_output_dim,
        ).to(device)
        self.target_encoder = copy.deepcopy(self.encoder)

        feature_dim = self.encoder.output_dim
        self.actor = Actor(feature_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critics = TwinQNetwork(feature_dim, action_dim).to(device)
        self.target_critics = copy.deepcopy(self.critics)

        self.critic_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.critics.parameters()), lr=args.q_lr
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)

        self.replay_buffer = ReplayBuffer(
            capacity=args.buffer_size,
            feature_dim=robot_feature_dim,
            action_dim=action_dim,
            image_shape=image_shape,
            device=device,
        )

        self.critic_losses = []
        self.actor_losses = []

    def get_action(self, obs, test: bool = False):
        # Normalize raw camera frames only when needed.
        if isinstance(obs, dict) and "image" in obs:
            if obs["image"].dtype == torch.uint8:
                obs_normalized = dict(obs)
                obs_normalized["image"] = obs_normalized["image"].float() / 255.0
                obs = obs_normalized
        elif isinstance(obs, torch.Tensor):
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0

        features = self.encoder(obs)
        actions = self.actor(features)

        if not test and self.exploration_noise > 0:
            actions = actions + torch.randn_like(actions) * self.exploration_noise

        return actions.clamp(-1.0, 1.0)

    def store_transition(self, obs, next_obs, actions, rewards, terminations, truncations):
        self.replay_buffer.add_batch(obs, next_obs, actions, rewards, terminations, truncations)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.lerp_(online_param, self.tau)

    def update(self) -> None:
        self.update_count += 1
        batch = self.replay_buffer.sample(self.batch_size)

        # 1. Critic target: clipped double Q + target policy smoothing.
        with torch.no_grad():
            next_features = self.target_encoder(batch.next_observations)
            next_actions = self.target_actor(next_features)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            next_q1, next_q2 = self.target_critics(next_features, next_actions)
            next_q = torch.minimum(next_q1, next_q2)
            target_q = batch.rewards + (1.0 - batch.terminations) * self.gamma * next_q

        # 2. Update both critics and the shared image encoder.
        features = self.encoder(batch.observations)
        q1, q2 = self.critics(features, batch.actions)
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_losses.append(critic_loss.item())

        # 3. Delayed actor and target-network update.
        if self.update_count % self.policy_delay == 0:
            self._set_requires_grad(self.critics, False)

            # Keep actor updates out of the shared encoder.
            actor_features = self.encoder(batch.observations).detach()
            actor_actions = self.actor(actor_features)
            actor_q1, _ = self.critics(actor_features, actor_actions)
            actor_loss = -actor_q1.mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self._set_requires_grad(self.critics, True)

            self._soft_update(self.encoder, self.target_encoder)
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critics, self.target_critics)
            self.actor_losses.append(actor_loss.item())

    def log_losses(self, step: int, writer) -> None:
        if self.critic_losses:
            writer.add_scalar("losses/Q_loss", np.mean(self.critic_losses), step)
        if self.actor_losses:
            writer.add_scalar("losses/actor_loss", np.mean(self.actor_losses), step)

        self.critic_losses.clear()
        self.actor_losses.clear()

    def save(self, path: str, metadata=None) -> None:
        # Keep legacy keys for existing render scripts.
        checkpoint = {
            "critic_cnn_target": self.target_encoder.state_dict(),
            "critic_cnn": self.encoder.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.target_actor.state_dict(),
            "Qnetworks_state_dict": self.critics.state_dict(),
            "target_Qnetworks_state_dict": self.target_critics.state_dict(),
            "action_scale": np.array(1.0, dtype=np.float32),
            "action_bias": np.array(0.0, dtype=np.float32),
        }
        if metadata is not None:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)
        tqdm.write(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.target_encoder.load_state_dict(checkpoint["critic_cnn_target"])
        self.encoder.load_state_dict(checkpoint["critic_cnn"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.target_actor.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critics.load_state_dict(checkpoint["Qnetworks_state_dict"])
        self.target_critics.load_state_dict(checkpoint["target_Qnetworks_state_dict"])

        tqdm.write(f"Model loaded from: {path}")
