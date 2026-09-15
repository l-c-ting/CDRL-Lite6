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

    def _midpoint_alignment(self, tool: torch.Tensor, target: torch.Tensor):
        """Reward XY alignment of the fingertip midpoint near the grasp height."""
        xy_error = torch.linalg.norm(tool[..., :2] - target[..., :2], dim=-1)
        xy_reward = 1.0 - torch.tanh(
            xy_error / self.reward_config.midpoint_align_scale
        )

        # ``tool`` is the configured center between the finger tips.  Gate the
        # XY reward by height so the policy cannot collect it while hovering far
        # above or below the cube.
        z_error = (tool[..., 2] - target[..., 2]).abs()
        height_gate = 1.0 - torch.tanh(
            z_error / self.reward_config.midpoint_align_scale
        )
        reward = xy_reward * height_gate
        aligned = xy_error <= self.reward_config.midpoint_align_threshold
        return reward, xy_error, z_error, aligned

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
        return (angle_from_y - (math.pi / 4.0)).clamp(min=0.0)

    def _lift_quality_reward(self, cube: torch.Tensor):
        """Return held lift progress, direction quality, and motion penalties."""
        grasped = self.physical_grasp.bool()
        grasped_float = grasped.float()

        z_lift = (cube[..., 2] - self.cube_init_zs).clamp(min=0.0)
        lift_progress = (
            z_lift / self.env_config.lift_height_threshold
        ).clamp(0.0, 1.0) * grasped_float

        delta_pos = cube - self.prev_cube_pos
        delta_z = delta_pos[..., 2]
        motion_norm = torch.linalg.norm(delta_pos, dim=-1)
        upward_progress = (
            delta_z / self.reward_config.lift_progress_scale
        ).clamp(-1.0, 1.0) * grasped_float

        vertical_direction = torch.zeros_like(delta_z)
        valid_motion = grasped & (
            motion_norm > self.reward_config.min_lift_motion
        )
        vertical_direction[valid_motion] = (
            delta_z[valid_motion] / (motion_norm[valid_motion] + 1e-8)
        )

        xy_drift_distance = torch.linalg.norm(
            cube[..., :2] - self.grasp_reference_xy, dim=-1
        )
        xy_drift = (
            xy_drift_distance / self.reward_config.lift_drift_scale
        ).clamp(0.0, 1.0) * grasped_float

        cube_vel = self.cube.get_vel().to(DEVICE, dtype=torch.float32)
        cube_vel = cube_vel.unsqueeze(0) if cube_vel.dim() == 1 else cube_vel
        lateral_speed = torch.linalg.norm(cube_vel[..., :2], dim=-1)
        lateral_velocity = (
            lateral_speed / self.reward_config.lateral_velocity_scale
        ).clamp(0.0, 1.0) * grasped_float

        return (
            lift_progress,
            upward_progress,
            vertical_direction,
            xy_drift,
            lateral_velocity,
            xy_drift_distance,
            lateral_speed,
        )

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

        current_grasp = self.physical_grasp.bool()
        new_grasp = current_grasp & (~self.prev_physical_grasp)
        if new_grasp.any():
            self.grasp_reference_xy[new_grasp] = cube[new_grasp, :2]

        # Positive terms.
        r_approach = self._approach_reward(tool, target)
        r_grasp, p_premature_close, close_amount, tool_dist = self._grasp_reward(
            tool, target
        )
        r_midpoint, midpoint_error, midpoint_z_error, midpoint_aligned = (
            self._midpoint_alignment(tool, target)
        )
        # Keep this term faithful to its name: alignment is rewarded separately.
        r_physical_grasp = current_grasp.float()
        (
            r_lift,
            r_upward,
            r_vertical,
            p_drift,
            p_lateral_vel,
            xy_drift_distance,
            lateral_speed,
        ) = self._lift_quality_reward(cube)

        # Pay success on every held step above the threshold.
        success = current_grasp & (
            (cube[..., 2] - self.cube_init_zs) >= self.env_config.lift_height_threshold
        )
        r_success = success.float() * self.reward_config.success_bonus

        # Penalty terms.
        p_orientation = self._downward_orientation_penalty(ee_quat)
        p_rotation = self._rotation_penalty(left, right)

        c = self.reward_config

        if c.reward_shaping:
            reward = (
                (c.approach_weight * r_approach)
                + (c.grasp_weight * r_grasp)
                + (c.midpoint_align_weight * r_midpoint)
                + (c.physical_grasp_weight * r_physical_grasp)
                + (c.lift_weight * r_lift)
                + (c.lift_progress_weight * r_upward)
                + (c.vertical_direction_weight * r_vertical)
                - (c.lift_drift_penalty_weight * p_drift)
                - (c.lateral_velocity_penalty_weight * p_lateral_vel)
                - (c.orientation_penalty_weight * p_orientation)
                - (c.rotation_penalty_weight * p_rotation)
                - (c.premature_close_penalty_weight * p_premature_close)
                + r_success
            )
        else:
            reward = r_success

        # Expose raw terms for diagnostics.
        info = {
            "r_approach": r_approach.detach(),
            "r_grasp": r_grasp.detach(),
            "r_midpoint": r_midpoint.detach(),
            "r_physical_grasp": r_physical_grasp.detach(),
            "r_lift": r_lift.detach(),
            "r_upward": r_upward.detach(),
            "r_vertical": r_vertical.detach(),
            "p_drift": p_drift.detach(),
            "p_lateral_velocity": p_lateral_vel.detach(),
            "p_orientation": p_orientation.detach(),
            "p_rotation": p_rotation.detach(),
            "p_premature_close": p_premature_close.detach(),
            "midpoint_error": midpoint_error.detach(),
            "midpoint_z_error": midpoint_z_error.detach(),
            "midpoint_aligned": midpoint_aligned.detach(),
            "xy_drift_distance": xy_drift_distance.detach(),
            "lateral_speed": lateral_speed.detach(),
            "tool_distance": tool_dist.detach(),
            "gripper_close_amount": close_amount.detach(),
            "success": success.detach(),
        }

        self.prev_cube_pos = cube.detach().clone()
        self.prev_physical_grasp = current_grasp.detach().clone()

        return reward, info
