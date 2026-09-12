import torch
from typing import Dict, Tuple
from common.runtime import DEVICE


class RewardMixin:
    """Compute approach, two-finger contact, and lifting rewards."""

    def _approach_reward(self, tcp_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """Return r1 in [0, 1], increasing as the gripper approaches the cube."""
        distance = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
        reward = 1.0 - torch.tanh(self.reward_config.approach_distance_scale * distance)
        return reward.clamp(0.0, 1.0)

    def _compute_reward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cube = self.cube.get_pos().to(DEVICE, dtype=torch.float32)
        cube = cube.unsqueeze(0) if cube.dim() == 1 else cube

        tool = self._tool_pos()
        tool = tool.unsqueeze(0) if tool.dim() == 1 else tool

        target = cube.clone()
        target[:, 2] += self.env_config.grasp_site_z_offset

        c = self.reward_config
        tool_distance = torch.linalg.norm(tool - target, dim=-1)
        both_finger_contact = self.contact_left & self.contact_right
        lifted = cube[..., 2] > self.env_config.lift_height_threshold

        r1 = self._approach_reward(tool, target)
        r2 = both_finger_contact.float() * c.both_finger_contact_reward
        r3 = lifted.float() * c.lifted_reward
        reward = r1 + r2 + r3

        info = {
            "r1_approach": r1.detach(),
            "r2_both_finger_contact": r2.detach(),
            "r3_lifted": r3.detach(),
            "tool_distance": tool_distance.detach(),
            "both_finger_contact": both_finger_contact.detach(),
            "success": lifted.detach(),
        }
        return reward, info
