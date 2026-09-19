import torch
from tqdm import tqdm
from common.runtime import DEVICE


class EvaluationManager:
    """Run deterministic evaluation episodes and log their averages."""

    def __init__(self, env, seed, num_episodes):
        self.env = env
        self.num_episodes = int(num_episodes)
        self.obs, _ = env.reset(seed=seed)

    @torch.no_grad()
    def evaluate(self, agent, step: int, writer=None):
        num_envs = self.env.num_envs
        episode_count = 0
        reward_sum = 0.0
        length_sum = 0.0
        success_sum = 0.0
        dwell_steps_sum = 0.0

        episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)
        episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)

        while episode_count < self.num_episodes:
            actions = agent.get_action(self.obs, test=True)
            self.obs, rewards, terminations, truncations, info = self.env.step(actions)

            episode_rewards += rewards.float().view(-1)
            episode_lengths += 1
            dones = (terminations | truncations).view(-1)
            done_ids = torch.nonzero(dones, as_tuple=False).flatten()

            for env_id in done_ids.tolist():
                if episode_count >= self.num_episodes:
                    break
                reward_sum += episode_rewards[env_id].item()
                length_sum += episode_lengths[env_id].item()
                success_sum += info["is_success"][env_id].float().item()
                dwell_steps_sum += info["target_dwell_steps"][env_id].item()
                episode_count += 1

            if done_ids.numel() > 0:
                # Reset the full batched scene.
                self.obs, _ = self.env.reset()
                episode_rewards.zero_()
                episode_lengths.zero_()

        mean_reward = reward_sum / self.num_episodes
        mean_length = length_sum / self.num_episodes
        success_rate = success_sum / self.num_episodes
        mean_dwell_steps = dwell_steps_sum / self.num_episodes
        mean_dwell_ratio = mean_dwell_steps / mean_length if mean_length > 0 else 0.0

        if writer is not None:
            writer.add_scalar("eval/eval_rews", mean_reward, step)
            writer.add_scalar("eval/eval_lens", mean_length, step)
            writer.add_scalar("eval/success_rate", success_rate, step)
            writer.add_scalar("eval/target_dwell_steps", mean_dwell_steps, step)
            writer.add_scalar("eval/target_dwell_ratio", mean_dwell_ratio, step)

        tqdm.write(
            f"[Eval] step={step} | reward={mean_reward:.2f} | "
            f"length={mean_length:.1f} | success_rate={success_rate:.3f} | "
            f"target_dwell={mean_dwell_steps:.1f}"
        )
        return mean_reward, mean_length, success_rate
