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


def select_training_gpu():
    """Select a GPU before CUDA is initialized."""
    parser = argparse.ArgumentParser(description="Train TD3 on a selected physical GPU.")
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
        f"({selected['free']} MiB free). It will appear inside training as cuda:0.",
        flush=True,
    )
    return True

# Select CUDA before importing Torch or Genesis.
if __name__ == "__main__" and not select_training_gpu():
    raise SystemExit(0)


import torch
import wandb
from dataclasses import asdict, replace
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from agents import TD3
from common.runtime import DEVICE
from config import EnvConfig, RewardConfig, TD3Config, WarmupConfig
from envs import Lite6LiftEnv
from evaluation import EvaluationManager
from utils import seed_everything
from warmup import run_warmup, reset_if_done


def train(
    td3_config=TD3Config(),
    env_config=EnvConfig(),
    reward_config=RewardConfig(),
    warmup_config=WarmupConfig(),
):
    """Run random warmup followed by TD3 training."""
    seed_everything(td3_config.seed)
    image_size = (84, 84)
    # Keep train and evaluation episode limits separate.
    train_cfg = replace(env_config, max_episode_steps=td3_config.train_max_episode_steps)
    eval_cfg = replace(env_config, max_episode_steps=td3_config.eval_max_episode_steps)
    run_dir = os.path.join(
        td3_config.output_dir, "test_TD3", f"seed_{td3_config.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    model_dir, wandb_dir, tb_dir = [
        os.path.join(run_dir, x) for x in ("model", "wandb", "tensorboard")
    ]

    for d in (model_dir, wandb_dir, tb_dir):
        os.makedirs(d, exist_ok=True)

    wandb.init(
        project="Lite6_Lifting",
        sync_tensorboard=True,
        config={
            "td3": asdict(td3_config),
            "warmup": {
                "mode": "uniform_random",
                "exploration_total_steps": warmup_config.exploration_total_steps,
                "episode_steps": train_cfg.max_episode_steps,
                "reset_interval_steps": warmup_config.reset_interval_steps,
                "action_distribution": "Uniform(-1, 1)",
                "store_all_transitions": True,
            },
            "reward": asdict(reward_config),
        },
        name=f"TD3/{td3_config.seed}",
        save_code=False,
        dir=wandb_dir,
    )
    writer = SummaryWriter(tb_dir)
    env = Lite6LiftEnv(
        train_cfg, reward_config, image_size=image_size, num_envs=td3_config.num_envs
    )
    eval_env = Lite6LiftEnv(
        eval_cfg, reward_config, image_size=image_size, num_envs=td3_config.eval_num_envs
    )
    agent = TD3(td3_config, image_size, env.feature_dim, env.action_dim, DEVICE)
    evaluator = EvaluationManager(eval_env, td3_config.seed, td3_config.eval_episodes)
    obs, _ = env.reset(seed=td3_config.seed)
    best = -float("inf")
    exploration_progress = tqdm(
        total=warmup_config.exploration_total_steps,
        desc="Exploration",
        dynamic_ncols=True,
    )
    # Fill replay before the first gradient update.
    obs, exploration_steps = run_warmup(env, agent, obs, warmup_config, exploration_progress)
    exploration_progress.close()
    tqdm.write(
        f"[step budgets] exploration={exploration_steps}, "
        f"training={td3_config.train_total_steps}, "
        f"combined={exploration_steps + td3_config.train_total_steps}"
    )

    train_step = 0
    progress = tqdm(total=td3_config.train_total_steps, desc="Training", dynamic_ncols=True)
    next_eval = td3_config.evaluate_freq

    while train_step < td3_config.train_total_steps:
        with torch.no_grad():
            act = agent.get_action(obs)
        nxt, rew, term, trunc, _ = env.step(act)
        agent.store_transition(obs, nxt, act, rew, term, trunc)
        obs = reset_if_done(env, nxt, term, trunc)
        # One scene step adds one transition per environment.
        train_step += env.num_envs
        progress.update(env.num_envs)
        for _ in range(td3_config.gradient_steps):
            agent.update()
        if train_step >= next_eval:
            score, _, _ = evaluator.evaluate(agent, train_step, writer)
            agent.log_losses(train_step, writer)
            if score > best:
                # Save the highest-reward checkpoint.
                best = score
                agent.save(os.path.join(model_dir, "model_best.pt"))
            next_eval += td3_config.evaluate_freq
    if td3_config.save_model:
        agent.save(os.path.join(model_dir, "model_final.pt"))
    progress.close()
    env.close()
    eval_env.close()
    writer.close()
    wandb.finish()


if __name__ == "__main__":
    train()
