"""
Flappy Bird – Gymnasium environment backed by Pygame.

Observation : (84, 84, 1) uint8 grayscale pixel frame
Actions      : 0 = do nothing, 1 = flap
Rewards      : +0.01 / frame survived · +1.0 / pipe passed · −1.0 on death

Episode parameters are re-randomised each reset() so the agent cannot
memorise a fixed course:
  • scroll_speed   – pixels/frame the pipes move leftward   [2.0, 4.5]
  • gap_size       – vertical opening between pipes (px)    [110, 175]
  • pipe_spacing   – horizontal distance between pipes (px) [160, 240]
"""

import os

import numpy as np
import pygame
from PIL import Image
import gymnasium as gym
from gymnasium import spaces


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Screen
    W, H = 288, 512

    # Bird
    BIRD_X = 72
    BIRD_R = 12          # visual radius; hitbox uses R-2 for slight forgiveness

    # Physics
    GRAVITY = 0.5
    FLAP_VEL = -9.0
    MAX_VEL = 12.0       # terminal velocity cap

    # Pipe
    PIPE_W = 52

    # Ground strip height
    GROUND_H = 30

    def __init__(self, render_mode: str | None = None, display_scale: float = 1.0):
        super().__init__()
        self.render_mode = render_mode
        self.display_scale = display_scale

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        # Headless SDL for training workers
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        pygame.init()

        # self.screen is always the internal W×H surface used for game logic
        # and pixel observations — its size never changes.
        self.screen = pygame.Surface((self.W, self.H))

        # self.display is the visible window (only exists in human mode).
        if render_mode == "human":
            dw = int(self.W * display_scale)
            dh = int(self.H * display_scale)
            self.display = pygame.display.set_mode((dw, dh))
            pygame.display.set_caption("Flappy Bird – PPO")
        else:
            self.display = None

        self.clock = pygame.time.Clock()

        # Game-state variables (populated by reset)
        self.bird_y: float = 0.0
        self.bird_vel: float = 0.0
        self.pipes: list[dict] = []
        self.score: int = 0
        self.scroll_speed: float = 0.0
        self.gap_size: int = 0
        self.pipe_spacing: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise episode parameters
        self.scroll_speed = float(self.np_random.uniform(2.0, 4.5))
        self.gap_size     = int(self.np_random.integers(100, 160))
        self.pipe_spacing = int(self.np_random.integers(160, 240))

        # Bird state
        self.bird_y   = float(self.H // 2)
        self.bird_vel = 0.0

        # Pipes – pre-spawn one off-screen to the right
        self.pipes = []
        self._spawn_pipe(self.W + 80)

        self.score = 0
        return self._get_obs(), {}

    def step(self, action):
        # ---- Action ----
        if action == 1:
            self.bird_vel = self.FLAP_VEL

        # ---- Physics ----
        self.bird_vel = min(self.bird_vel + self.GRAVITY, self.MAX_VEL)
        self.bird_y  += self.bird_vel

        # ---- Move pipes & detect passes ----
        reward = 0.1  # per-frame survival bonus
        for pipe in self.pipes:
            pipe["x"] -= self.scroll_speed
            if not pipe["passed"] and pipe["x"] + self.PIPE_W < self.BIRD_X:
                pipe["passed"] = True
                reward += 5.0
                self.score += 1

        # Remove pipes that have scrolled off screen
        self.pipes = [p for p in self.pipes if p["x"] > -(self.PIPE_W + 10)]

        # Spawn a new pipe when the last one has moved far enough left
        if not self.pipes or self.pipes[-1]["x"] < self.W - self.pipe_spacing:
            self._spawn_pipe()

        # ---- Termination checks ----
        terminated = False
        br = self.BIRD_R - 2  # slightly forgiving hitbox
        bx, by = self.BIRD_X, int(self.bird_y)

        # Floor or ceiling
        if (self.bird_y + br >= self.H - self.GROUND_H) or (self.bird_y - br <= 0):
            terminated = True

        # Pipe collision (AABB)
        if not terminated:
            bird_rect = pygame.Rect(bx - br, by - br, br * 2, br * 2)
            half_gap  = self.gap_size // 2
            for pipe in self.pipes:
                px = int(pipe["x"])
                top_rect = pygame.Rect(px, 0, self.PIPE_W, pipe["gap_y"] - half_gap)
                bot_rect = pygame.Rect(px, pipe["gap_y"] + half_gap,
                                       self.PIPE_W, self.H - (pipe["gap_y"] + half_gap))
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                    terminated = True
                    break

        if terminated:
            reward = -2.0

        return self._get_obs(), reward, terminated, False, {"score": self.score}

    def render(self):
        if self.render_mode == "human":
            self._draw_frame()
            self.present()

    def present(self) -> None:
        """Scale the internal surface to the display window and flip.

        watch.py calls this *after* blitting any HUD overlay so that both
        the game frame and the HUD appear in a single flip, eliminating
        flicker.  render() calls it automatically for standalone use.
        """
        if self.render_mode != "human" or self.display is None:
            return
        if self.display_scale != 1.0:
            scaled = pygame.transform.scale(self.screen, self.display.get_size())
            self.display.blit(scaled, (0, 0))
        else:
            self.display.blit(self.screen, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame.get_init():
            pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_pipe(self, x: float | None = None):
        if x is None:
            x = float(self.W + 50)
        half = self.gap_size // 2
        # Gap centre: keep enough margin from ceiling and ground
        gap_y = int(self.np_random.integers(half + 60, self.H - self.GROUND_H - half - 20))
        self.pipes.append({"x": x, "gap_y": gap_y, "passed": False})

    def _draw_frame(self):
        """Render one frame onto self.screen."""
        # Sky gradient approximation (solid for speed)
        self.screen.fill((113, 197, 207))

        # Pipes
        half_gap = self.gap_size // 2
        for pipe in self.pipes:
            px  = int(pipe["x"])
            gy  = pipe["gap_y"]
            top = gy - half_gap
            bot = gy + half_gap

            # Pipe body
            pygame.draw.rect(self.screen, (78, 176, 56),
                             (px, 0, self.PIPE_W, top))
            pygame.draw.rect(self.screen, (78, 176, 56),
                             (px, bot, self.PIPE_W, self.H - bot))
            # Pipe cap (wider, darker)
            cap_h = 18
            pygame.draw.rect(self.screen, (55, 148, 36),
                             (px - 3, top - cap_h, self.PIPE_W + 6, cap_h))
            pygame.draw.rect(self.screen, (55, 148, 36),
                             (px - 3, bot, self.PIPE_W + 6, cap_h))

        # Ground
        pygame.draw.rect(self.screen, (222, 184, 135),
                         (0, self.H - self.GROUND_H, self.W, self.GROUND_H))
        pygame.draw.rect(self.screen, (110, 90, 50),
                         (0, self.H - self.GROUND_H, self.W, 3))

        # Bird
        bx = self.BIRD_X
        by = int(self.bird_y)
        pygame.draw.circle(self.screen, (255, 215, 0),  (bx, by), self.BIRD_R)
        pygame.draw.circle(self.screen, (255, 255, 255), (bx + 5, by - 3), 4)
        pygame.draw.circle(self.screen, (20,  20,  20),  (bx + 6, by - 3), 2)
        # Wing hint
        pygame.draw.ellipse(self.screen, (255, 180, 0),
                            (bx - self.BIRD_R, by - 2, 14, 8))

        pygame.event.pump()

    def _get_obs(self) -> np.ndarray:
        """Return (84, 84, 1) uint8 grayscale observation."""
        self._draw_frame()

        # surfarray gives (W, H, 3); transpose to (H, W, 3)
        raw = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

        # Grayscale + resize via Pillow (no extra SDL2 conflicts)
        img = Image.fromarray(raw, mode="RGB").convert("L").resize(
            (84, 84), Image.BILINEAR
        )
        return np.array(img, dtype=np.uint8)[:, :, np.newaxis]
