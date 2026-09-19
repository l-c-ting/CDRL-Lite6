import torch
import torch.nn.functional as F
from common.runtime import DEVICE

# Map segmentation IDs to policy colors.
SEG_COLORS = torch.zeros((256, 3), dtype=torch.uint8, device=DEVICE)
SEG_COLORS[:] = torch.tensor([255, 165, 0], dtype=torch.uint8, device=DEVICE)
SEG_COLORS[8] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=DEVICE)
SEG_COLORS[9] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=DEVICE)
SEG_COLORS[10] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=DEVICE)
SEG_COLORS[11] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=DEVICE)
SEG_COLORS[12] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=DEVICE)

class ObsMixin:
    """Build image and robot-state observations."""

    def _get_obs(self):
        # The policy sees wrist-camera segmentation.
        self.wrist_camera.move_to_attach()
        _, _, seg, _ = self.wrist_camera.render(
            rgb=True, depth=False, segmentation=True, normal=False
        )
        img = self._resize(
            self._crop(SEG_COLORS[torch.as_tensor(seg, device=DEVICE).long().clamp(0, 255)]),
            self.image_size,
        )
        if img.shape[0] != self.num_envs:
            raise RuntimeError(
                f"Wrist camera returned {img.shape[0]} images, expected {self.num_envs}."
            )
        return {"image": img.contiguous(), "feature": self._robot_feature()}

    def _gripper_state(self):
        if not self.has_gripper:
            return torch.ones((self.num_envs, 1), dtype=torch.float32, device=DEVICE)
        q = self.robot.get_dofs_position(self.gripper_dofs)
        den = torch.where(
            (self.gripper_close - self.gripper_open).abs() < 1e-6,
            torch.full_like(self.gripper_close, 1e-6),
            self.gripper_close - self.gripper_open,
        )
        return 1.0 - 2.0 * ((q - self.gripper_open) / den).clamp(0.0, 1.0).mean(dim=-1, keepdim=True)

    def _robot_feature(self):
        # Arm state, tool pose, tool velocity, gripper state, and target vector.
        q = self.robot.get_dofs_position(self.arm_dofs)
        dq = self.robot.get_dofs_velocity(self.arm_dofs)
        p = self.ee_link.get_pos()
        quat = self.ee_link.get_quat()
        v = self.ee_link.get_vel()
        if q.dim() == 1:
            q, dq, p, quat, v = (
                q.unsqueeze(0),
                dq.unsqueeze(0),
                p.unsqueeze(0),
                quat.unsqueeze(0),
                v.unsqueeze(0),
            )
        target_from_tool = self.target_positions - self._tool_pos()
        return torch.cat(
            (q, dq, p, quat, v, self._gripper_state(), target_from_tool),
            dim=-1,
        ).float()

    @staticmethod
    def _crop(x):
        if x.dim() == 3:
            h, w, _ = x.shape
            s = min(h, w)
            return x[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
        _, h, w, _ = x.shape
        s = min(h, w)
        return x[:, (h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]

    @staticmethod
    def _resize(x, size):
        if x.dim() == 3:
            return (
                F.interpolate(x.permute(2, 0, 1).unsqueeze(0).float(), size=size, mode="nearest")
                .squeeze(0)
                .to(torch.uint8)
            )
        return F.interpolate(x.permute(0, 3, 1, 2).float(), size=size, mode="nearest").to(torch.uint8)
