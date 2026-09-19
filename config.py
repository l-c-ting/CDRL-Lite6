from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np

ROOT = Path(__file__).resolve().parent


def arr(x):
    return np.array(x, dtype=np.float32)


@dataclass
class EnvConfig:
    # Simulation timing and episode limits.
    robot_xml: Path = ROOT / "ufactory_lite6" / "lite6_gripper_wide.xml"

    dt: float = 0.003
    substeps: int = 12
    max_episode_steps: int = 5000

    # Rigid-body solver settings.
    reset_settle_steps: int = 80
    rigid_iterations: int = 100
    rigid_ls_iterations: int = 100
    rigid_constraint_timeconst: float = 0.005
    rigid_box_box_detection: bool = True
    max_dynamic_constraints: int = 32

    # Arm control and dynamics.
    arm_init: np.ndarray = field(default_factory=lambda: arr([0.0, 0.17, 0.55, 0.0, 0.38, 0.0]))
    arm_base_position: np.ndarray = field(default_factory=lambda: arr([0.0, 0.0, 0.0]))
    max_joint_velocity: float = np.pi / 3.0
    max_delta_velocity: float = 0.5
    kp: np.ndarray = field(default_factory=lambda: np.full(6, 2000.0, dtype=np.float32))
    kv: np.ndarray = field(default_factory=lambda: arr([2500, 3500, 2500, 2500, 2500, 2500]))
    armature: np.ndarray = field(default_factory=lambda: arr([0.05, 0.01, 0.05, 0.05, 0.05, 0.05]))
    damping: np.ndarray = field(default_factory=lambda: arr([0.3, 0.01, 0.3, 0.3, 0.3, 0.3]))
    friction_loss: np.ndarray = field(default_factory=lambda: np.full(6, 3.0, dtype=np.float32))
    force_limit: np.ndarray = field(default_factory=lambda: np.full(6, 100000.0, dtype=np.float32))

    # Gripper and contact detection.
    gripper_kp: float = 1500.0
    gripper_kv: float = 35.0
    gripper_force_limit: float = 10.0
    gripper_force_scale: float = 5.0
    physical_grasp_close_threshold: float = -0.8
    contact_z_margin: float = 0.003
    noslip_iterations: int = 5
    noslip_tolerance: float = 1e-6
    tool_site_local: np.ndarray = field(default_factory=lambda: arr([0.0, 0.0, 0.0811]))
    grasp_site_z_offset: float = -0.005

    # Cube task and randomization.
    cube_init_position: np.ndarray = field(default_factory=lambda: arr([0.205, 0.010, 0.010]))
    cube_size: np.ndarray = field(default_factory=lambda: arr([0.025, 0.025, 0.020]))
    randomize_cube_size: bool = False
    cube_size_min: np.ndarray = field(default_factory=lambda: arr([0.020, 0.020, 0.015]))
    cube_size_max: np.ndarray = field(default_factory=lambda: arr([0.030, 0.030, 0.025]))
    randomize_cube_position: bool = True
    cube_position_xy_min: np.ndarray = field(default_factory=lambda: arr([0.180, -0.015]))
    cube_position_xy_max: np.ndarray = field(default_factory=lambda: arr([0.230, 0.035]))
    cube_density: float = 500.0
    cube_friction: float = 1.0
    lift_target_height: float = 0.10
    success_hold_steps: int = 500

    # Side camera used for external viewing.
    side_camera_view: Literal["front", "right", "left"] = "front"
    side_camera_resolution: tuple[int, int] = (1280, 720)
    side_camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    side_camera_fov: float = 40.0
    side_camera_near: float = 0.0001
    side_camera_far: float = 10.0
    side_camera_distance: float = 0.3767
    side_camera_height: float = 0.1467
    camera_lookat_z_offset: float = 0.067

    # Wrist camera used by the policy.
    wrist_camera_resolution: tuple[int, int] = (1280, 720)
    wrist_camera_fov: float = 70.0
    wrist_camera_near: float = 0.0001
    wrist_camera_far: float = 10.0
    wrist_camera_position: tuple[float, float, float] = (0.06, 0.0, 0.03)
    wrist_camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.11)
    wrist_camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass
class WarmupConfig:
    # Uniform-random replay warmup.
    exploration_total_steps: int = 80_000
    log_interval_steps: int = 10_000
    reset_interval_steps: int = 1_000


@dataclass
class TD3Config:
    # Run output and reproducibility.
    seed: int = 0
    save_model: bool = True
    output_dir: Path = Path("runs/td3_lite6")

    # TD3 optimization settings.
    train_total_steps: int = 1_600_000
    buffer_size: int = 1_200_000
    q_lr: float = 1e-4
    policy_lr: float = 1e-4
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    exploration_noise: float = 0.3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    cnn_output_dim: int = 32
    num_envs: int = 16
    gradient_steps: int = 16
    # Vectorized training and evaluation.
    train_max_episode_steps: int = 5000
    eval_max_episode_steps: int = 5000
    eval_num_envs: int = 3
    eval_episodes: int = 3
    evaluate_freq: int = 16_000


@dataclass
class RewardConfig:
    reward_shaping: bool = True

    # Phase 1: approach and establish a physical grasp.
    side_reach_weight: float = 0.15
    grasp_weight: float = 0.50
    grasp_close_weight: float = 0.20
    grasp_contact_weight: float = 0.30
    grasp_event_weight: float = 0.50
    reach_distance_scale: float = 25.0

    # Phase 2: move the grasped cube to its per-episode target.
    lift_weight: float = 3.00
    lift_target_std: float = 0.10

    # Reward every step spent within the target sphere while still grasped.
    target_tolerance: float = 0.02
    target_dwell_reward: float = 5.00

    # Discourage losing an established grasp.
    drop_penalty_weight: float = 4.00
