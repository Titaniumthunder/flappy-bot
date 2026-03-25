"""
Load the trained PPO model and watch it play Flappy Bird in real time.

Usage:
  .venv/bin/python watch.py [--model flappy_ppo] [--speed 1.0]

Keys:
  Q  – quit
  R  – restart current episode immediately

Close the window or press Ctrl-C to stop.
"""

import argparse
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from flappy_env import FlappyBirdEnv

DISPLAY_SCALE = 1.5


def main(model_path: str = "flappy_ppo", speed: float = 1.0) -> None:
    env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="human",
                                             display_scale=DISPLAY_SCALE)])
    env = VecFrameStack(env, n_stack=4)

    print(f"Loading model from {model_path}.zip …")
    model = PPO.load(model_path, env=env, device="cpu")

    inner_env = env.venv.envs[0]
    display   = inner_env.display

    pygame.font.init()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    obs       = env.reset()
    ep_reward = 0.0
    ep_num    = 0

    print("Running – Q to quit, R to restart episode, Ctrl-C to stop.\n")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    if event.key == pygame.K_r:
                        obs = env.reset()
                        ep_reward = 0.0
                        print(f"  [restarted episode {ep_num + 1}]")

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])

            score = infos[0].get("score", 0)
            hud_lines = [
                f"Episode  {ep_num + 1}",
                f"Score    {score}",
                f"Reward   {ep_reward:.1f}",
                f"[R] restart   [Q] quit",
            ]

            # Scale game frame to display
            if inner_env.display_scale != 1.0:
                scaled = pygame.transform.scale(inner_env.screen, display.get_size())
                display.blit(scaled, (0, 0))
            else:
                display.blit(inner_env.screen, (0, 0))

            # Draw HUD on top
            y = 6
            for line in hud_lines:
                shadow = font.render(line, True, (0, 0, 0))
                text   = font.render(line, True, (255, 255, 255))
                display.blit(shadow, (7, y + 1))
                display.blit(text,   (6, y))
                y += text.get_height() + 2

            pygame.display.flip()
            inner_env.clock.tick(inner_env.metadata["render_fps"])

            if dones[0]:
                ep_num += 1
                print(f"Episode {ep_num:4d} | pipes: {score:4d} | "
                      f"reward: {ep_reward:8.2f}")
                ep_reward = 0.0
                obs = env.reset()

            if speed < 1.0:
                import time
                time.sleep((1.0 / 30.0) * (1.0 / speed - 1.0))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the Flappy Bird PPO agent.")
    parser.add_argument("--model", default="flappy_ppo",
                        help="Path to the saved model (without .zip)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 0.5 = half speed)")
    args = parser.parse_args()
    main(model_path=args.model, speed=args.speed)