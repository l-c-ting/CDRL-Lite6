import math
import torch
from typing import Dict, Tuple
from common.runtime import DEVICE


class RewardMixin:
    """Compute dense grasp-and-lift rewards."""

    def _approach_reward(self, tcp_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """Reward proximity to the grasp point."""
        dist = torch.linalg.norm(
            tcp_pos - target_pos, dim=-1
        )
        return 1.0 - torch.tanh(10.0 * dist)

    def _grasp_reward(self, tool: torch.Tensor, target: torch.Tensor):
        """Reward closing near or around the cube."""
        close_amount = self._gripper_close_amount()
        tool_dist = torch.linalg.norm(tool - target, dim=-1)
        near = tool_dist <= self.reward_config.grasp_near_distance
        contact = self.physical_grasp.float()
        near_ratio = self.reward_config.grasp_near_ratio
        reward = close_amount * (near_ratio * near.float() + (1.0 - near_ratio) * contact)
        premature_close = close_amount * (~near).float()
        return reward, premature_close, close_amount, tool_dist

    def _downward_orientation_penalty(self, ee_quat: torch.Tensor) -> torch.Tensor:
        """Penalize end-effector tilt."""
        w, x, y, z = ee_quat[..., 0], ee_quat[..., 1], ee_quat[..., 2], ee_quat[..., 3]

        z_dir_z = w**2 - x**2 - y**2 + z**2
        deviation = (z_dir_z - (-1.0)).abs()
        return deviation

    def _rotation_penalty(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Penalize finger rotation beyond 45 degrees."""
        vec_xy = left[..., :2] - right[..., :2]
        angle_from_y = torch.atan2(vec_xy[..., 0].abs(), vec_xy[..., 1].abs())

        excess_rotation = (angle_from_y - (math.pi / 4.0)).clamp(min=0.0)
        return excess_rotation

    def _straight_line_penalty(
        self, tcp_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> torch.Tensor:
        """Penalize horizontal tool offset."""
        xy_dist = torch.linalg.norm(tcp_pos[..., :2] - target_pos[..., :2], dim=-1)
        return xy_dist

    def _straight_lift_reward(self, cube_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return held lift progress and XY drift."""
        z_lift = (cube_pos[..., 2] - self.cube_init_zs).clamp(min=0.0)
        lift_progress = (z_lift / self.env_config.lift_height_threshold).clamp(0.0, 1.0)

        init_xy = self.cube_initial_positions[:, :2]
        xy_drift = torch.linalg.norm(cube_pos[..., :2] - init_xy, dim=-1)

        lift_score = lift_progress * self.physical_grasp.float()
        lift_drift_penalty = xy_drift * self.physical_grasp.float()

        return lift_score, lift_drift_penalty

    def _compute_reward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cube = self.cube.get_pos().to(DEVICE, dtype=torch.float32)
        cube = cube.unsqueeze(0) if cube.dim() == 1 else cube

        left = self.left_finger_link.get_pos()
        right = self.right_finger_link.get_pos()
        tool = self._tool_pos()
        ee_quat = self.ee_link.get_quat()
        left = left.unsqueeze(0) if left.dim() == 1 else left
        right = right.unsqueeze(0) if right.dim() == 1 else right
        tool = tool.unsqueeze(0) if tool.dim() == 1 else tool
        ee_quat = ee_quat.unsqueeze(0) if ee_quat.dim() == 1 else ee_quat

        target = cube.clone()
        target[:, 2] += self.env_config.grasp_site_z_offset

        # Positive terms.
        r_approach = self._approach_reward(tool, target)
        r_side_grasp, p_premature_close, close_amount, tool_dist = self._grasp_reward(tool, target)
        r_physical_grasp = self.physical_grasp.float()
        r_lift, p_lift_drift = self._straight_lift_reward(cube)

        # Pay success on every held step above the threshold.
        success = self.physical_grasp & (
            (cube[..., 2] - self.cube_init_zs) >= self.env_config.lift_height_threshold
        )
        r_success = success.float() * self.reward_config.success_bonus

        # Penalty terms.
        p_orientation = self._downward_orientation_penalty(ee_quat)
        p_rotation = self._rotation_penalty(left, right)
        p_straight_line = self._straight_line_penalty(tool, target)

        c = self.reward_config

        if c.reward_shaping:
            reward = (
                (c.approach_weight * r_approach)
                + (c.grasp_weight * r_side_grasp)
                + (c.physical_grasp_weight * r_physical_grasp)
                + (c.lift_weight * r_lift)
                + r_success
                - (c.orientation_penalty_weight * p_orientation)
                - (c.rotation_penalty_weight * p_rotation)
                - (c.straight_line_penalty_weight * p_straight_line)
                - (c.lift_drift_penalty_weight * p_lift_drift)
                - (c.premature_close_penalty_weight * p_premature_close)
            )
        else:
            reward = r_success

        # Expose raw terms for diagnostics.
        info = {
            "r_approach": r_approach.detach(),
            "r_side_grasp": r_side_grasp.detach(),
            "r_physical_grasp": r_physical_grasp.detach(),
            "r_lift": r_lift.detach(),
            "p_orientation": p_orientation.detach(),
            "p_rotation": p_rotation.detach(),
            "p_straight_line": p_straight_line.detach(),
            "p_lift_drift": p_lift_drift.detach(),
            "p_premature_close": p_premature_close.detach(),
            "tool_distance": tool_dist.detach(),
            "gripper_close_amount": close_amount.detach(),
            "success": success.detach(),
        }
        return reward, info
