import numpy as np
import torch
from common.runtime import DEVICE
from config import EnvConfig, RewardConfig
from .scene_mixin import SceneMixin
from .obs_mixin import ObsMixin
from .reward_mixin import RewardMixin

class Lite6LiftEnv(SceneMixin, ObsMixin, RewardMixin):
    """Vectorized Lite6 cube-lifting environment."""

    def __init__(
        self,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        image_size=(84, 84),
        num_envs=1,
        env_spacing=(1.0, 1.0),
        render_mode=None,
    ):
        self.env_config, self.reward_config = env_config, reward_config
        self.image_size, self.num_envs, self.env_spacing, self.render_mode = (
            image_size,
            int(num_envs),
            env_spacing,
            render_mode,
        )
        # Six arm velocities plus one gripper command.  The policy feature also
        # includes a 3D vector from the tool site to the per-episode target.
        self.action_dim, self.feature_dim = 7, 26
        self.arm_init = torch.as_tensor(env_config.arm_init, dtype=torch.float32, device=DEVICE)
        self.cube_sizes_np = self._sample_cube_sizes()
        self.cube_sizes = torch.as_tensor(
            self.cube_sizes_np, dtype=torch.float32, device=DEVICE
        )
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)
        self.target_hold_steps = torch.zeros_like(self.current_step)
        self.success_count = torch.zeros_like(self.current_step)
        self.gripper_command = torch.ones(self.num_envs, dtype=torch.float32, device=DEVICE)
        initial_position = torch.as_tensor(
            env_config.cube_init_position, dtype=torch.float32, device=DEVICE
        ).repeat(self.num_envs, 1)
        initial_position[:, 2] = 0.5 * self.cube_sizes[:, 2]
        self.cube_initial_positions = initial_position
        self.cube_init_zs = initial_position[:, 2].clone()
        self.cube_init_z = float(self.cube_init_zs.mean().item())
        self.target_positions = initial_position.clone()
        self.target_positions[:, 2] += self.env_config.lift_target_height
        self._build_scene()
        self._find_robot_components()
        self._attach_wrist_camera()
        self._setup_gripper_limits()
        self._setup_physics()
        self._setup_gripper()
        self._setup_grasp_state()

    def _sample_cube_sizes(self):
        """Return one cube size per environment."""
        cfg = self.env_config
        if not cfg.randomize_cube_size:
            return np.repeat(np.asarray(cfg.cube_size, dtype=np.float32)[None, :], self.num_envs, 0)
        low = np.asarray(cfg.cube_size_min, dtype=np.float32)
        high = np.asarray(cfg.cube_size_max, dtype=np.float32)
        if low.shape != (3,) or high.shape != (3,) or np.any(low <= 0.0) or np.any(low > high):
            raise ValueError("cube_size_min/max must be positive 3-vectors with min <= max")
        return np.random.uniform(low, high, size=(self.num_envs, 3)).astype(np.float32)

    def _sample_cube_positions(self):
        """Sample reset positions on the table."""
        cfg = self.env_config
        positions = torch.empty((self.num_envs, 3), dtype=torch.float32, device=DEVICE)
        if cfg.randomize_cube_position:
            low = torch.as_tensor(cfg.cube_position_xy_min, dtype=torch.float32, device=DEVICE)
            high = torch.as_tensor(cfg.cube_position_xy_max, dtype=torch.float32, device=DEVICE)
            if low.shape != (2,) or high.shape != (2,) or bool(torch.any(low > high).item()):
                raise ValueError("cube_position_xy_min/max must be 2-vectors with min <= max")
            positions[:, :2] = low + torch.rand(
                (self.num_envs, 2), dtype=torch.float32, device=DEVICE
            ) * (high - low)
        else:
            positions[:, :2] = torch.as_tensor(
                cfg.cube_init_position[:2], dtype=torch.float32, device=DEVICE
            )
        positions[:, 2] = 0.5 * self.cube_sizes[:, 2]
        return positions

    def _setup_gripper(self):
        """Configure gripper gains and force limits."""
        if not self.has_gripper:
            return
        left = self.robot.get_joint("gripper_left_finger")
        right = self.robot.get_joint("gripper_right_finger")
        self.left_gripper_dof = int(left.dofs_idx_local[0])
        self.right_gripper_dof = int(right.dofs_idx_local[0])
        lim = np.asarray(left.dofs_limit, dtype=np.float32).reshape(-1, 2)[0]
        self.left_open, self.left_close = float(lim[0]), float(lim[1])
        c = self.env_config
        self.robot.set_dofs_kp(
            np.array([c.gripper_kp], dtype=np.float32), dofs_idx_local=[self.left_gripper_dof]
        )
        self.robot.set_dofs_kv(
            np.array([c.gripper_kv], dtype=np.float32), dofs_idx_local=[self.left_gripper_dof]
        )
        self.robot.set_dofs_force_range(
            lower=np.array([-c.gripper_force_limit], dtype=np.float32),
            upper=np.array([c.gripper_force_limit], dtype=np.float32),
            dofs_idx_local=[self.left_gripper_dof],
        )
        self.robot.set_dofs_force_range(
            lower=np.array([-c.gripper_force_limit], dtype=np.float32),
            upper=np.array([c.gripper_force_limit], dtype=np.float32),
            dofs_idx_local=[self.right_gripper_dof],
        )

    def _setup_grasp_state(self):
        """Allocate batched contact state."""
        self.left_finger_link_idx = int(self.left_finger_link.idx)
        self.right_finger_link_idx = int(self.right_finger_link.idx)
        self.physical_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        self.grasp_contact_ready = torch.zeros_like(self.physical_grasp)
        self.contact_left = torch.zeros_like(self.physical_grasp)
        self.contact_right = torch.zeros_like(self.physical_grasp)
        self.contact_opposite = torch.zeros_like(self.physical_grasp)
        self.contact_left_y = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=DEVICE
        )
        self.contact_right_y = self.contact_left_y.clone()
        self.contact_left_axis = self.contact_left_y.clone()
        self.contact_right_axis = self.contact_left_y.clone()
        self.prev_physical_grasp = torch.zeros_like(self.physical_grasp)
        self.has_grasped = torch.zeros_like(self.physical_grasp)

    @staticmethod
    def _quat_rotate(q, v):
        uv = torch.cross(q[..., 1:], v, dim=-1)
        return v + 2.0 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))

    def _tool_pos(self):
        p, q = self.ee_link.get_pos(), self.ee_link.get_quat()
        p, q = (p.unsqueeze(0), q.unsqueeze(0)) if p.dim() == 1 else (p, q)
        local = torch.as_tensor(
            self.env_config.tool_site_local, dtype=torch.float32, device=DEVICE
        ).expand(p.shape[0], -1)
        return p + self._quat_rotate(q, local)

    def _gripper_close_amount(self):
        if not self.has_gripper:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=DEVICE)
        q = self.robot.get_dofs_position(self.gripper_dofs)
        q = q.unsqueeze(0) if q.dim() == 1 else q
        span = self.gripper_close - self.gripper_open
        span = torch.where(span.abs() < 1e-6, torch.full_like(span, 1e-6), span)
        return ((q - self.gripper_open) / span).clamp(0.0, 1.0).min(dim=-1).values

    def _contact_state(self):
        """Measure finger contacts around each cube."""
        contacts = self.robot.get_contacts(with_entity=self.cube)
        a, b, pos = contacts["link_a"], contacts["link_b"], contacts["position"]
        if a.dim() == 1:
            a, b = a.unsqueeze(0), b.unsqueeze(0)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        false = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        nan = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=DEVICE)
        if a.shape[1] == 0:
            return (
                false,
                false.clone(),
                false.clone(),
                nan,
                nan.clone(),
                nan.clone(),
                nan.clone(),
            )
        valid = contacts.get("valid_mask", torch.ones_like(a, dtype=torch.bool))
        valid = valid.unsqueeze(0) if valid.dim() == 1 else valid
        left = ((a == self.left_finger_link_idx) | (b == self.left_finger_link_idx)) & valid
        right = ((a == self.right_finger_link_idx) | (b == self.right_finger_link_idx)) & valid
        cube = self.cube.get_pos()
        cube = cube.unsqueeze(0) if cube.dim() == 1 else cube
        y = pos[..., 1] - cube[:, None, 1]
        jaw_axis = self.right_finger_link.get_pos() - self.left_finger_link.get_pos()
        jaw_axis = jaw_axis.unsqueeze(0) if jaw_axis.dim() == 1 else jaw_axis
        jaw_axis = torch.nn.functional.normalize(jaw_axis, dim=-1, eps=1e-6)
        axis_offset = ((pos - cube[:, None, :]) * jaw_axis[:, None, :]).sum(-1)
        z = (pos[..., 2] - cube[:, None, 2]).abs() <= (
            0.5 * self.cube_sizes[:, None, 2] + self.env_config.contact_z_margin
        )
        left, right = left & z, right & z
        lc, rc = left.sum(1), right.sum(1)
        ly = torch.where(lc > 0, torch.where(left, y, torch.zeros_like(y)).sum(1) / lc.clamp(min=1).float(), nan)
        ry = torch.where(rc > 0, torch.where(right, y, torch.zeros_like(y)).sum(1) / rc.clamp(min=1).float(), nan)
        la = torch.where(
            lc > 0,
            torch.where(left, axis_offset, torch.zeros_like(axis_offset)).sum(1)
            / lc.clamp(min=1).float(),
            nan,
        )
        ra = torch.where(
            rc > 0,
            torch.where(right, axis_offset, torch.zeros_like(axis_offset)).sum(1)
            / rc.clamp(min=1).float(),
            nan,
        )
        l, r = lc > 0, rc > 0
        # Valid grasps need contacts on opposite sides.
        opposite = l & r & torch.isfinite(la) & torch.isfinite(ra) & ((la * ra) < 0.0)
        return l, r, opposite, ly, ry, la, ra

    def _update_grasp(self, grip):
        l, r, opp, ly, ry, la, ra = self._contact_state()
        self.contact_left, self.contact_right, self.contact_opposite = l, r, opp
        self.contact_left_y, self.contact_right_y = ly, ry
        self.contact_left_axis, self.contact_right_axis = la, ra
        # Require opposing contacts and a close command.
        closing = grip <= self.env_config.physical_grasp_close_threshold
        self.grasp_contact_ready = opp & closing
        self.physical_grasp = self.grasp_contact_ready.clone()

    def reset(self, seed=None, options=None):
        """Reset all batched environments."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.current_step.zero_()
        self.target_hold_steps.zero_()
        self.success_count.zero_()
        self.gripper_command.fill_(1.0)
        for x in (
            self.physical_grasp,
            self.grasp_contact_ready,
            self.contact_left,
            self.contact_right,
            self.contact_opposite,
            self.prev_physical_grasp,
            self.has_grasped,
        ):
            x.zero_()
        self.contact_left_y.fill_(float("nan"))
        self.contact_right_y.fill_(float("nan"))
        self.contact_left_axis.fill_(float("nan"))
        self.contact_right_axis.fill_(float("nan"))
        zero_arm = torch.zeros(len(self.arm_dofs), dtype=torch.float32, device=DEVICE)
        self.robot.set_dofs_position(self.arm_init, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_velocity(zero_arm, dofs_idx_local=self.arm_dofs)
        self.robot.control_dofs_velocity(zero_arm, dofs_idx_local=self.arm_dofs)
        if self.has_gripper:
            zg = torch.zeros(len(self.gripper_dofs), dtype=torch.float32, device=DEVICE)
            self.robot.set_dofs_position(self.gripper_open, dofs_idx_local=self.gripper_dofs)
            self.robot.set_dofs_velocity(zg, dofs_idx_local=self.gripper_dofs)
            self.robot.control_dofs_force(
                torch.zeros(
                    (self.num_envs, len(self.gripper_dofs)), dtype=torch.float32, device=DEVICE
                ),
                dofs_idx_local=self.gripper_dofs,
            )

        cube_positions = self._sample_cube_positions()
        self.cube.set_pos(cube_positions, zero_velocity=True)
        self.cube.set_quat(
            torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE
            ).repeat(self.num_envs, 1),
            zero_velocity=True,
        )
        self.cube.zero_all_dofs_velocity()

        for _ in range(self.env_config.reset_settle_steps):
            self.scene.step()
        settled = self.cube.get_pos()
        settled = settled.unsqueeze(0) if settled.dim() == 1 else settled
        self.cube_initial_positions = settled.detach().clone()
        self.cube_init_zs = self.cube_initial_positions[:, 2]
        self.cube_init_z = float(self.cube_init_zs.mean().item())
        self.target_positions = self.cube_initial_positions.clone()
        self.target_positions[:, 2] += self.env_config.lift_target_height
        return self._get_obs(), {}

    def step(self, actions):
        """Apply one velocity-control step."""
        a = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
        a = a.unsqueeze(0) if a.dim() == 1 else a
        if a.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(a.shape)}."
            )
        self.current_step += 1
        a = a.clamp(-1.0, 1.0)
        vel = a[:, : len(self.arm_dofs)] * self.env_config.max_joint_velocity
        cur = self.robot.get_dofs_velocity(self.arm_dofs)
        cur = cur.unsqueeze(0) if cur.dim() == 1 else cur
        # Limit abrupt velocity changes.
        self.robot.control_dofs_velocity(
            cur
            + (vel - cur).clamp(
                -self.env_config.max_delta_velocity, self.env_config.max_delta_velocity
            ),
            dofs_idx_local=self.arm_dofs,
        )
        grip = (
            a[:, 6]
            if self.has_gripper
            else torch.ones(self.num_envs, dtype=torch.float32, device=DEVICE)
        )
        if self.has_gripper:
            self.gripper_command = grip.detach()
            force = grip * self.env_config.gripper_force_scale
            self.robot.control_dofs_force(
                torch.stack((-force, force), dim=1),
                dofs_idx_local=[self.left_gripper_dof, self.right_gripper_dof],
            )
        self.scene.step()
        if self.has_gripper:
            self._update_grasp(grip)
        obs = self._get_obs()
        reward, info = self._compute_reward()
        in_target = info["in_target"].bool()

        # Count consecutive target occupancy. Leaving the target resets the
        # current streak. Each full hold interval records one success and
        # starts a fresh interval without interrupting the episode.
        self.target_hold_steps = torch.where(
            in_target,
            self.target_hold_steps + 1,
            torch.zeros_like(self.target_hold_steps),
        )
        success_event = (
            self.target_hold_steps >= self.env_config.success_hold_steps
        )
        self.success_count += success_event.long()
        self.target_hold_steps = torch.where(
            success_event,
            torch.zeros_like(self.target_hold_steps),
            self.target_hold_steps,
        )
        success = self.success_count > 0

        # Reaching the success threshold never ends an episode early. Every
        # environment runs to the configured time limit so target reward can
        # continue accumulating.
        terminated = torch.zeros_like(success)
        truncated = self.current_step >= self.env_config.max_episode_steps
        info.update(
            is_success=success.detach(),
            success_event=success_event.detach(),
            success_count=self.success_count.detach().clone(),
            target_hold_steps=self.target_hold_steps.detach().clone(),
            contact_left=self.contact_left.detach().clone(),
            contact_right=self.contact_right.detach().clone(),
            contact_opposite=self.contact_opposite.detach().clone(),
            contact_left_y=self.contact_left_y.detach().clone(),
            contact_right_y=self.contact_right_y.detach().clone(),
            contact_left_axis=self.contact_left_axis.detach().clone(),
            contact_right_axis=self.contact_right_axis.detach().clone(),
            physical_grasp=self.physical_grasp.detach().clone(),
            cube_size=self.cube_sizes.detach().clone(),
            cube_initial_position=self.cube_initial_positions.detach().clone(),
            target_position=self.target_positions.detach().clone(),
        )
        return (
            obs,
            reward.detach(),
            terminated.detach(),
            truncated.detach(),
            info,
        )

    def close(self):
        self.physical_grasp.zero_()
