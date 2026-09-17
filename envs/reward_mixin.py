import math
import torch
from typing import Dict, Tuple
from common.runtime import DEVICE


class RewardMixin:
    def _downward_orientation_penalty(self, ee_quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = ee_quat[..., 0], ee_quat[..., 1], ee_quat[..., 2], ee_quat[..., 3]
        z_dir_z = w**2 - x**2 - y**2 + z**2
        return (z_dir_z + 1.0).abs()

    def _rotation_penalty(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        vec_xy = left[..., :2] - right[..., :2]
        angle_from_y = torch.atan2(vec_xy[..., 0].abs(), vec_xy[..., 1].abs())
        return (angle_from_y - math.pi / 4.0).clamp(min=0.0)

    def _alignment(self, tool: torch.Tensor, cube: torch.Tensor):
        """Return XY error between the fingertip midpoint and cube center."""
        xy_error = torch.linalg.norm(tool[..., :2] - cube[..., :2], dim=-1)
        aligned = xy_error <= self.reward_config.midpoint_xy_threshold
        return xy_error, aligned

    def _compute_reward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c = self.reward_config

        cube = self.cube.get_pos().to(DEVICE, dtype=torch.float32)
        if cube.dim() == 1:
            cube = cube.unsqueeze(0)

        left = self.left_finger_link.get_pos()
        right = self.right_finger_link.get_pos()
        tool = self._tool_pos()
        ee_quat = self.ee_link.get_quat()

        if left.dim() == 1:
            left = left.unsqueeze(0)
        if right.dim() == 1:
            right = right.unsqueeze(0)
        if tool.dim() == 1:
            tool = tool.unsqueeze(0)
        if ee_quat.dim() == 1:
            ee_quat = ee_quat.unsqueeze(0)

        target = cube.clone()
        target[:, 2] += self.env_config.grasp_site_z_offset

        tool_dist = torch.linalg.norm(tool - target, dim=-1)
        midpoint_xy_error, midpoint_aligned = self._alignment(tool, cube)
        midpoint_z_error = (tool[..., 2] - target[..., 2]).abs()
        close_amount = self._gripper_close_amount()

        current_grasp = self.physical_grasp.bool()
        valid_grasp = current_grasp & midpoint_aligned
        new_grasp = valid_grasp & (~self.prev_valid_grasp)
        first_grasp = new_grasp & (~self.has_grasped)

        if new_grasp.any():
            self.grasp_reference_xy[new_grasp] = cube[new_grasp, :2]
        self.has_grasped |= first_grasp

        # Re-enable approach/alignment shaping after a dropped grasp.  The
        # grasp event bonus itself is still paid only once per episode.
        pre_grasp = (~valid_grasp).float()

        r_approach = (
            (self.prev_tool_dist - tool_dist) / c.approach_progress_scale
        ).clamp(-1.0, 1.0) * pre_grasp
        r_midpoint = (
            (self.prev_midpoint_xy_error - midpoint_xy_error)
            / c.midpoint_progress_scale
        ).clamp(-1.0, 1.0) * pre_grasp

        near = tool_dist <= c.grasp_near_distance
        p_premature_close = (
            close_amount * (~(near & midpoint_aligned)).float() * pre_grasp
        )
        r_grasp_event = first_grasp.float()

        delta_pos = cube - self.prev_cube_pos
        delta_z = delta_pos[..., 2]
        grasped_float = current_grasp.float()

        # Once the cube has been grasped, keep scoring its signed height change
        # even while contact is lost.  A released/falling cube therefore pays
        # back the progress reward instead of making the negative motion free.
        r_upward = (
            delta_z / c.lift_progress_scale
        ) * self.has_grasped.float()

        xy_drift_distance = torch.linalg.norm(
            cube[..., :2] - self.grasp_reference_xy, dim=-1
        )
        p_drift = (
            (xy_drift_distance / c.lift_drift_scale).clamp(0.0, 1.0)
            * grasped_float
        )

        cube_vel = self.cube.get_vel().to(DEVICE, dtype=torch.float32)
        if cube_vel.dim() == 1:
            cube_vel = cube_vel.unsqueeze(0)

        lateral_speed = torch.linalg.norm(cube_vel[..., :2], dim=-1)
        p_lateral_velocity = (
            (lateral_speed / c.lateral_velocity_scale).clamp(0.0, 1.0)
            * grasped_float
        )

        success = valid_grasp & (
            (cube[..., 2] - self.cube_init_zs) >= self.env_config.lift_height_threshold
        )
        new_success = success & (~self.has_succeeded)
        r_success_event = new_success.float()
        r_success_hold = success.float()
        self.has_succeeded |= new_success

        # Penalize the loss immediately.  Re-grasping is allowed, but does not
        # pay another event bonus; delaying the penalty until re-grasp would
        # discourage recovery instead of discouraging the drop/open action.
        dropped = self.has_grasped & self.prev_physical_grasp & (~current_grasp)
        p_drop = dropped.float()

        p_orientation = self._downward_orientation_penalty(ee_quat)
        p_rotation = self._rotation_penalty(left, right)

        if c.reward_shaping:
            reward = (
                c.approach_weight * r_approach
                + c.midpoint_align_weight * r_midpoint
                + c.grasp_event_bonus * r_grasp_event
                + c.upward_progress_weight * r_upward
                + c.success_bonus * r_success_event
                + c.success_hold_weight * r_success_hold
                - c.drop_penalty_weight * p_drop
                - c.lift_drift_penalty_weight * p_drift
                - c.lateral_velocity_penalty_weight * p_lateral_velocity
                - c.orientation_penalty_weight * p_orientation
                - c.rotation_penalty_weight * p_rotation
                - c.premature_close_penalty_weight * p_premature_close
            )
        else:
            reward = c.success_bonus * r_success_event

        info = {
            "r_approach": r_approach.detach(),
            "r_midpoint": r_midpoint.detach(),
            "r_grasp_event": r_grasp_event.detach(),
            "r_upward": r_upward.detach(),
            "r_success_event": r_success_event.detach(),
            "r_success_hold": r_success_hold.detach(),
            "p_drop": p_drop.detach(),
            "p_drift": p_drift.detach(),
            "p_lateral_velocity": p_lateral_velocity.detach(),
            "p_orientation": p_orientation.detach(),
            "p_rotation": p_rotation.detach(),
            "p_premature_close": p_premature_close.detach(),
            "tool_distance": tool_dist.detach(),
            "midpoint_error": midpoint_xy_error.detach(),
            "midpoint_xy_error": midpoint_xy_error.detach(),
            "midpoint_z_error": midpoint_z_error.detach(),
            "midpoint_aligned": midpoint_aligned.detach(),
            "xy_drift_distance": xy_drift_distance.detach(),
            "lateral_speed": lateral_speed.detach(),
            "gripper_close_amount": close_amount.detach(),
            "physical_grasp": current_grasp.detach(),
            "success": success.detach(),
        }

        self.prev_cube_pos = cube.detach().clone()
        self.prev_tool_dist = tool_dist.detach().clone()
        self.prev_midpoint_xy_error = midpoint_xy_error.detach().clone()
        self.prev_physical_grasp = current_grasp.detach().clone()
        self.prev_valid_grasp = valid_grasp.detach().clone()

        return reward, info
