import torch
from tqdm import tqdm

from common.runtime import DEVICE


def reset_if_done(env, obs, term, trunc):
    """Reset the batch when any environment ends."""
    return env.reset()[0] if torch.any(term | trunc) else obs


@torch.no_grad()
def run_warmup(env, agent, obs, cfg, progress):
    """Fill replay with uniform-random actions."""
    total = cfg.exploration_total_steps
    forced_resets = 0
    for step in range(0, total, env.num_envs):
        act = torch.rand(
            (env.num_envs, env.action_dim),
            dtype=torch.float32,
            device=DEVICE,
        ) * 2.0 - 1.0
        nxt, rew, term, trunc, _ = env.step(act)
        agent.store_transition(obs, nxt, act, rew, term, trunc)
        scene_steps = (step + env.num_envs) // env.num_envs
        # Prevent long random rollouts from dominating replay.
        force_reset = scene_steps % cfg.reset_interval_steps == 0
        if torch.any(term | trunc) or force_reset:
            obs, _ = env.reset()
            forced_resets += int(force_reset)
        else:
            obs = nxt
        progress.update(env.num_envs)
        if (step + env.num_envs) % cfg.log_interval_steps == 0:
            tqdm.write(
                f"[random warmup] {step + env.num_envs}/{total}, "
                f"replay_size={len(agent.replay_buffer)}, "
                f"forced_resets={forced_resets}"
            )
    tqdm.write(
        f"[random warmup final] replay_size={len(agent.replay_buffer)}, "
        f"forced_resets={forced_resets}"
    )
    return obs, total
