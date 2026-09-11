import argparse
import os
import subprocess
import sys


def _available_gpus():
    """Read physical GPU status from nvidia-smi."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Unable to query GPUs with nvidia-smi: {exc}") from exc

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, name, total, used, free, utilization = [
            value.strip() for value in line.split(",", maxsplit=6)
        ]
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "total": total,
                "used": used,
                "free": free,
                "utilization": utilization,
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi did not report any GPUs.")
    return gpus


def select_render_gpu():
    """Select a GPU before CUDA is initialized."""
    parser = argparse.ArgumentParser(description="Render a TD3 video on a selected physical GPU.")
    parser.add_argument(
        "--gpu",
        metavar="INDEX",
        help="physical GPU index shown by nvidia-smi; skips the interactive prompt",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="show the available GPU choices and exit",
    )
    args = parser.parse_args()
    gpus = _available_gpus()

    print("Available GPUs:")
    for gpu in gpus:
        print(
            f"  [{gpu['index']}] {gpu['name']} | "
            f"total {gpu['total']} MiB | used {gpu['used']} MiB | "
            f"free {gpu['free']} MiB | util {gpu['utilization']}%"
        )
    if args.list_gpus:
        return False

    valid_indices = {gpu["index"]: gpu for gpu in gpus}
    selected_index = args.gpu
    if selected_index is None:
        if not sys.stdin.isatty():
            parser.error("interactive GPU selection requires a terminal; use --gpu INDEX")
        while True:
            selected_index = input("Select GPU index: ").strip()
            if selected_index in valid_indices:
                break
            print(f"Invalid GPU index. Choose one of: {', '.join(valid_indices)}")
    elif selected_index not in valid_indices:
        parser.error(
            f"invalid GPU index {selected_index!r}; "
            f"choose one of: {', '.join(valid_indices)}"
        )

    selected = valid_indices[selected_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    print(
        f"Selected physical GPU {selected['index']}: {selected['name']} "
        f"({selected['free']} MiB free). It will appear inside rendering as cuda:0.",
        flush=True,
    )
    return True

# Select CUDA before importing Torch or Genesis.
if __name__ == "__main__" and not select_render_gpu():
    raise SystemExit(0)


import imageio
import numpy as np
import torch
import torch.nn.functional as F
from agents.td3_agents import TD3
from common.runtime import DEVICE
from config import EnvConfig, RewardConfig, TD3Config
from envs import Lite6LiftEnv


def frame_uint8(x):
    """Convert a rendered frame to uint8."""
    x = x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else np.squeeze(x)
    return (
        (x * 255).astype(np.uint8) if x.dtype != np.uint8 and x.max() <= 1.0 else x.astype(np.uint8)
    )


def render_video(model_path):
    """Render wrist RGB and policy-view videos."""
    cfg = EnvConfig()
    env = Lite6LiftEnv(cfg, RewardConfig(), num_envs=1)
    agent = TD3(TD3Config(), (84, 84), env.feature_dim, env.action_dim, DEVICE)
    agent.load(model_path)
    obs, _ = env.reset(seed=42)
    human, ai = [], []
    for _ in range(cfg.max_episode_steps):
        with torch.no_grad():
            act = agent.get_action(obs, test=True)
        obs, _, term, trunc, _ = env.step(act)
        # Human view uses wrist-camera RGB.
        human.append(
            frame_uint8(
                env.wrist_camera.render(
                    rgb=True, depth=False, segmentation=False, normal=False
                )[0]
            )
        )
        # Policy view uses colored segmentation.
        img = (
            F.interpolate(
                obs["image"].squeeze(0).float().unsqueeze(0), scale_factor=8, mode="nearest"
            )
            .squeeze(0)
            .byte()
        )
        ai.append(img.permute(1, 2, 0).cpu().numpy())
        if term.any() or trunc.any():
            break

    # Keep one out of every five frames.
    imageio.mimsave("video_human.mp4", human[::5], fps=30)
    imageio.mimsave("video_ai.mp4", ai[::5], fps=30)


if __name__ == "__main__":
    path = input("Enter model path: ")
    render_video(path)
