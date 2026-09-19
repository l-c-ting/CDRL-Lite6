import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from common.runtime import DEVICE


class RewardMixin:
    """Compute phase-gated grasp and target-dwell rewards."""

    def _side_reach_reward(self, cube: torch.Tensor) -> torch.Tensor:
        grasp_target = cube.clone()
        grasp_target[:, 2] += self.env_config.grasp_site_z_offset

        tool = self._tool_pos()
        if tool.dim() == 1:
            tool = tool.unsqueeze(0)
        distance = torch.linalg.norm(tool - grasp_target, dim=-1)
        position_reward = 1.0 - torch.tanh(
            self.reward_config.reach_distance_scale * distance
        )

        left = self.left_finger_link.get_pos()
        right = self.right_finger_link.get_pos()
        if left.dim() == 1:
            left = left.unsqueeze(0)
        if right.dim() == 1:
            right = right.unsqueeze(0)
        finger_axis = F.normalize(right - left, dim=-1, eps=1e-6)
        y_alignment = finger_axis[:, 1].abs()

        return 0.85 * position_reward + 0.15 * y_alignment

    def _grasp_reward(self, side_reach: torch.Tensor) -> torch.Tensor:
        close_amount = self._gripper_close_amount()
        close_near_target = side_reach * close_amount
        contact = self.contact_opposite.float()
        c = self.reward_config
        return (
            c.grasp_close_weight * close_near_target
            + c.grasp_contact_weight * contact
        )

    def _lift_target_reward(
        self, cube: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_distance = torch.linalg.norm(cube - self.target_positions, dim=-1)
        lift_reward = 1.0 - torch.tanh(
            target_distance / self.reward_config.lift_target_std
        )
        return lift_reward, target_distance

    def _compute_reward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cube = self.cube.get_pos().to(DEVICE, dtype=torch.float32)
        cube = cube.unsqueeze(0) if cube.dim() == 1 else cube

        c = self.reward_config
        r_side_reach = self._side_reach_reward(cube)
        r_grasp = self._grasp_reward(r_side_reach)
        r_lift, target_distance = self._lift_target_reward(cube)

        current_grasp = self.physical_grasp.bool()
        pre_grasp = (~current_grasp).float()
        grasped = current_grasp.float()
        new_grasp = current_grasp & (~self.prev_physical_grasp)
        first_grasp = new_grasp & (~self.has_grasped)
        dropped = self.has_grasped & self.prev_physical_grasp & (~current_grasp)
        self.has_grasped |= first_grasp

        in_target = current_grasp & (target_distance <= c.target_tolerance)

        if c.reward_shaping:
            reward = (
                pre_grasp
                * (c.side_reach_weight * r_side_reach + c.grasp_weight * r_grasp)
                + c.grasp_weight * c.grasp_event_weight * first_grasp.float()
                + grasped * c.lift_weight * r_lift
                + c.target_dwell_reward * in_target.float()
                - c.drop_penalty_weight * dropped.float()
            )
        else:
            reward = c.target_dwell_reward * in_target.float()

        lifted_height = (cube[:, 2] - self.cube_init_zs).clamp(min=0.0)
        info = {
            "r_side_reach": r_side_reach.detach(),
            "r_grasp": r_grasp.detach(),
            "r_grasp_event": first_grasp.float().detach(),
            "r_lift": r_lift.detach(),
            "r_target_dwell": in_target.float().detach(),
            "p_drop": dropped.float().detach(),
            "target_distance": target_distance.detach(),
            "lifted_height": lifted_height.detach(),
            "in_target": in_target.detach(),
            "physical_grasp": current_grasp.detach(),
        }

        self.prev_physical_grasp = current_grasp.detach().clone()
        return reward, info
