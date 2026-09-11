import torch
import numpy as np

from typing import Dict, NamedTuple
from common.runtime import DEVICE


class ReplayBufferSamples(NamedTuple):
    """A sampled transition batch."""

    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    next_observations: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor


class ReplayBuffer:
    """Store compact transitions in CPU memory."""

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        action_dim: int,
        image_shape=(3, 84, 84),
        device: torch.device = DEVICE,
    ):
        self.capacity = int(capacity)
        self.device = device

        self.features = np.empty((self.capacity, feature_dim), dtype=np.float32)
        self.next_features = np.empty((self.capacity, feature_dim), dtype=np.float32)
        # Keep images as uint8 until sampling.
        self.images = np.empty((self.capacity, *image_shape), dtype=np.uint8)
        self.next_images = np.empty((self.capacity, *image_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.terminations = np.empty((self.capacity, 1), dtype=np.float32)
        self.truncations = np.empty((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    @staticmethod
    def _to_numpy(x, dtype) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.ascontiguousarray(x, dtype=dtype)

    @classmethod
    def _to_uint8_image(cls, image) -> np.ndarray:
        image = cls._to_numpy(image, dtype=None)
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)

        image = image.astype(np.float32, copy=False)
        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0
        return np.ascontiguousarray(np.clip(np.rint(image), 0, 255), dtype=np.uint8)

    def add_batch(self, obs, next_obs, actions, rewards, terminations, truncations) -> None:
        features = self._to_numpy(obs["feature"], np.float32)
        next_features = self._to_numpy(next_obs["feature"], np.float32)
        images = self._to_uint8_image(obs["image"])
        next_images = self._to_uint8_image(next_obs["image"])
        actions = self._to_numpy(actions, np.float32)
        rewards = self._to_numpy(rewards, np.float32).reshape(-1, 1)
        terminations = self._to_numpy(terminations, np.float32).reshape(-1, 1)
        truncations = self._to_numpy(truncations, np.float32).reshape(-1, 1)

        if features.ndim == 1:
            # Accept a single unbatched transition.
            features = features[None]
            next_features = next_features[None]
            images = images[None]
            next_images = next_images[None]
            actions = actions[None]

        batch_size = features.shape[0]
        if batch_size > self.capacity:
            # Keep only the newest transitions.
            start = batch_size - self.capacity
            features = features[start:]
            next_features = next_features[start:]
            images = images[start:]
            next_images = next_images[start:]
            actions = actions[start:]
            rewards = rewards[start:]
            terminations = terminations[start:]
            truncations = truncations[start:]
            batch_size = self.capacity

        # Write through the circular buffer.
        indices = (np.arange(batch_size) + self.pos) % self.capacity
        self.features[indices] = features
        self.next_features[indices] = next_features
        self.images[indices] = images
        self.next_images[indices] = next_images
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.terminations[indices] = terminations
        self.truncations[indices] = truncations

        old_pos = self.pos
        self.pos = (self.pos + batch_size) % self.capacity
        self.full = self.full or old_pos + batch_size >= self.capacity

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        indices = np.random.randint(0, len(self), size=batch_size)

        # Move only the sampled batch to the device.
        images = torch.as_tensor(
            self.images[indices], dtype=torch.float32, device=self.device
        ).div_(255.0)
        next_images = torch.as_tensor(
            self.next_images[indices], dtype=torch.float32, device=self.device
        ).div_(255.0)

        return ReplayBufferSamples(
            observations={
                "image": images,
                "feature": torch.as_tensor(
                    self.features[indices], dtype=torch.float32, device=self.device
                ),
            },
            actions=torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            next_observations={
                "image": next_images,
                "feature": torch.as_tensor(
                    self.next_features[indices], dtype=torch.float32, device=self.device
                ),
            },
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            terminations=torch.as_tensor(
                self.terminations[indices], dtype=torch.float32, device=self.device
            ),
            truncations=torch.as_tensor(
                self.truncations[indices], dtype=torch.float32, device=self.device
            ),
        )
