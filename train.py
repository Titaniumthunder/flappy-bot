import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

from flappy_env import FlappyBirdEnv

# ── CNN (same architecture as CarRacing) ──────────────────────────────────────
class BiggerCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# ── Reward plot callback (same as CarRacing) ──────────────────────────────────
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, save_path):
        super().__init__()
        self.episode_rewards = []
        self.save_path = save_path
        self.fig, self.ax = plt.subplots()

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                print(f"Episode {len(self.episode_rewards):4d} | "
                      f"Reward: {ep_reward:8.2f} | "
                      f"Timestep: {self.num_timesteps:7d}")
                self._update_plot()
        return True

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("PPO Flappy Bird — Episode Reward")
        eps = list(range(1, len(self.episode_rewards) + 1))
        self.ax.plot(eps, self.episode_rewards, alpha=0.4, color="steelblue", label="Reward")
        if len(self.episode_rewards) >= 20:
            rolling = [
                sum(self.episode_rewards[max(0, i - 19):i + 1]) / min(20, i + 1)
                for i in range(len(self.episode_rewards))
            ]
            self.ax.plot(eps, rolling, color="orange", linewidth=2, label="20-ep mean")
        self.ax.legend()
        self.fig.savefig(self.save_path, dpi=100)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    N_ENVS      = 256
    TOTAL_STEPS = 4_000_000
    MODEL_PATH  = "flappy_ppo"
    PLOT_PATH   = "reward_plot.png"

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Creating {N_ENVS} parallel environments …")
    vec_env = make_vec_env(FlappyBirdEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    policy_kwargs = dict(
        features_extractor_class=BiggerCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model — continuing training …")
        model = PPO.load(MODEL_PATH, env=vec_env, device="mps")
        model.learning_rate = 1e-4
    else:
        print("No saved model found — starting fresh …")
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=2048,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device="mps",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path="./checkpoints/",
        name_prefix="flappy_ppo",
        verbose=1,
    )
    reward_cb = EpisodeRewardCallback(save_path=PLOT_PATH)

    print(f"Training for {TOTAL_STEPS:,} timesteps …\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_cb, reward_cb])

    model.save(MODEL_PATH)
    vec_env.close()
    print(f"\nDone! Model saved → {MODEL_PATH}.zip")
    print(f"Reward plot saved → {PLOT_PATH}")

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()