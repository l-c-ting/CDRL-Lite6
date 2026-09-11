import inspect
import genesis as gs
import numpy as np


def camera_pose_transform(position, lookat, up):
    """Build a camera transform from look-at vectors."""
    position = np.asarray(position, dtype=np.float32)
    lookat = np.asarray(lookat, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = lookat - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        raise ValueError("Wrist camera position and look-at point must differ.")
    forward /= forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("Wrist camera up vector must not be parallel to its view axis.")
    right /= right_norm
    camera_up = np.cross(right, forward)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = position
    transform[:3, :3] = np.column_stack((right, camera_up, -forward))
    return transform


class SceneMixin:
    """Create the Genesis scene and robot hardware."""

    def _rigid_options(self):
        args = dict(
            dt=self.env_config.dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=True,
            iterations=self.env_config.rigid_iterations,
            ls_iterations=self.env_config.rigid_ls_iterations,
            noslip_iterations=self.env_config.noslip_iterations,
            noslip_tolerance=self.env_config.noslip_tolerance,
            constraint_timeconst=self.env_config.rigid_constraint_timeconst,
            box_box_detection=self.env_config.rigid_box_box_detection,
            max_dynamic_constraints=self.env_config.max_dynamic_constraints,
        )
        try:
            # Support Genesis versions with fewer options.
            params = inspect.signature(gs.options.RigidOptions).parameters
            args = {k: v for k, v in args.items() if k in params}
        except Exception:
            pass
        return gs.options.RigidOptions(**args)

    def _build_scene(self):
        """Build the robot, cube, cameras, and table."""
        cfg = self.env_config
        angle = np.deg2rad({"front": 0.0, "right": -45.0, "left": 45.0}[cfg.side_camera_view])
        lookat = 0.5 * (cfg.cube_init_position + cfg.arm_base_position)
        lookat = np.asarray(lookat, dtype=np.float32).copy()
        lookat[2] += cfg.camera_lookat_z_offset
        pos = np.array(
            [
                lookat[0] + cfg.side_camera_distance * np.cos(angle),
                lookat[1] + cfg.side_camera_distance * np.sin(angle),
                cfg.side_camera_height,
            ],
            dtype=np.float32,
        )
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=tuple(pos), camera_lookat=tuple(lookat), camera_fov=cfg.side_camera_fov
            ),
            vis_options=gs.options.VisOptions(
                segmentation_level="link",
                shadow=False,
                background_color=(0.67, 1.0, 0.114514),
                ambient_light=(0.8, 0.8, 0.8),
            ),
            renderer=gs.renderers.BatchRenderer(use_rasterizer=True),
            rigid_options=self._rigid_options(),
            sim_options=gs.options.SimOptions(
                dt=cfg.dt, substeps=cfg.substeps, gravity=(0.0, 0.0, -9.81)
            ),
            show_viewer=self.render_mode == "human",
        )

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=str(cfg.robot_xml)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        
        cube_morphs = [
            gs.morphs.Box(
                size=size,
                pos=(cfg.cube_init_position[0], cfg.cube_init_position[1], size[2] * 0.5),
                collision=True,
            )
            for size in self.cube_sizes_np
        ]
        self.cube = self.scene.add_entity(
            morph=cube_morphs,
            material=gs.materials.Rigid(rho=cfg.cube_density, friction=cfg.cube_friction),
        )

        self.side_camera = self.scene.add_camera(
            res=cfg.side_camera_resolution,
            pos=tuple(pos),
            lookat=tuple(lookat),
            up=cfg.side_camera_up,
            fov=cfg.side_camera_fov,
            near=cfg.side_camera_near,
            far=cfg.side_camera_far,
            GUI=False,
        )
        self.wrist_camera = self.scene.add_camera(
            res=cfg.wrist_camera_resolution,
            fov=cfg.wrist_camera_fov,
            near=cfg.wrist_camera_near,
            far=cfg.wrist_camera_far,
            GUI=False,
        )
        self.scene.add_light(
            pos=(0.0, 0.0, 3.0),
            dir=(-0.5, -0.5, -1.0),
            color=(1.0, 1.0, 1.0),
            intensity=1.0,
            directional=True,
            castshadow=True,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.build(n_envs=self.num_envs, env_spacing=self.env_spacing)

    def _find_robot_components(self):
        """Cache robot joints, DOFs, and links."""
        joints = sorted(
            (j for j in self.robot.joints if j.n_dofs > 0), key=lambda j: j.dofs_idx_local[0]
        )
        self.arm_joints = [j for j in joints if j.name.startswith("joint")]
        self.gripper_joints = [j for j in joints if "gripper" in j.name.lower()]
        self.arm_dofs = [int(j.dofs_idx_local[0]) for j in self.arm_joints]
        self.gripper_dofs = [int(j.dofs_idx_local[0]) for j in self.gripper_joints]
        self.has_gripper = bool(self.gripper_dofs)
        self.ee_link = self.robot.get_link("link6")
        self.left_finger_link = self.robot.get_link("gripper_left_finger")
        self.right_finger_link = self.robot.get_link("gripper_right_finger")

    def _attach_wrist_camera(self):
        """Attach the policy camera to link6."""
        cfg = self.env_config
        camera_offset = camera_pose_transform(
            cfg.wrist_camera_position,
            cfg.wrist_camera_lookat,
            cfg.wrist_camera_up,
        )
        self.wrist_camera.attach(self.ee_link, camera_offset)
        self.wrist_camera.move_to_attach()

    def _setup_gripper_limits(self):
        """Cache gripper open and close limits."""
        if not self.has_gripper:
            self.gripper_open = self.gripper_close = None
            return
        left = np.asarray(
            self.robot.get_joint("gripper_left_finger").dofs_limit, dtype=np.float32
        ).reshape(-1, 2)[0]
        right = np.asarray(
            self.robot.get_joint("gripper_right_finger").dofs_limit, dtype=np.float32
        ).reshape(-1, 2)[0]

        import torch
        from common.runtime import DEVICE

        self.gripper_open = torch.tensor([left[0], right[1]], dtype=torch.float32, device=DEVICE)
        self.gripper_close = torch.tensor([left[1], right[0]], dtype=torch.float32, device=DEVICE)

    def _setup_physics(self):
        """Apply arm dynamics and controller gains."""
        c = self.env_config
        self.robot.set_dofs_force_range(-c.force_limit, c.force_limit, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_damping(c.damping, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_armature(c.armature, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_frictionloss(c.friction_loss, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_kv(c.kv, dofs_idx_local=self.arm_dofs)
        self.robot.set_dofs_kp(c.kp, dofs_idx_local=self.arm_dofs)
