from typing import Optional
import numpy as np
import gymnasium as gym
import time

import pyautogui
import win32gui

from ZumaInterface.StateReader import StateReader


class ZumaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self.render_mode = "rgb_array"

        self.state_reader = StateReader(0)

        self.max_delay_ms = 1000
        self.max_delay_rollback = 1500

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Box(0, 255, shape=(40, 40, 3), dtype=np.uint8)

        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64)

    def _get_obs(self):
        # self.state_reader.stack_frames(False)
        img = self.state_reader.screenshot_process()
        img_arr = np.array(img)
        return img_arr

    def _get_info(self):
        self.state_reader.read_game_values()
        score = self.state_reader.score

        return {
            "score": score
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state_reader.score = 0
        self.state_reader.progress = 0
        self.state_reader.lives = 3
        self.state_reader.write_game_values()
        time.sleep(3)
        self.state_reader.shoot_ball(180)
        time.sleep(1)
        self.state_reader.shoot_ball(180)
        time.sleep(1.5)

        # self.state_reader.stack_frames(True)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        angle = np.interp(action, (-1.0, 1.0), (0.0, 360.0))

        # angle = action
        old_score = self.state_reader.score
        old_lives = self.state_reader.lives

        self.state_reader.shoot_ball(angle)
        time.sleep(0.11)
        self.state_reader.reset_rotation()
        time.sleep(0.11)

        terminated = False
        self.state_reader.read_game_values()
        new_score = self.state_reader.score
        new_lives = self.state_reader.lives
        if new_lives != old_lives:
            terminated = True

        score_change = new_score - old_score
        reward = 0
        if score_change > 100:
            reward = 1
        elif score_change > 0:
            reward = 0.5
        # elif terminated:
        #     reward = -1
        # else:
        #     reward = -0.01

        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.state_reader.screenshot_process()
            return img

    # NOTE: old
    def step_old(self, action):
        angle = np.interp(action, (-1.0, 1.0), (0, 360))
        # angle = action
        terminated = False

        self.state_reader.read_game_values()
        old_score = self.state_reader.score
        new_score = old_score
        old_lives = self.state_reader.lives
        new_lives = old_lives

        start_s = time.time()
        elapsed_s = 0
        self.state_reader.shoot_ball(angle)

        score_changed = False
        lives_changed = False
        while elapsed_s * 1000 < self.max_delay_ms:
            elapsed_s = time.time() - start_s
            self.state_reader.read_game_values()
            new_score = self.state_reader.score
            new_lives = self.state_reader.lives
            if new_lives != old_lives:
                lives_changed = True
                break
            if new_score != old_score:
                score_changed = True
                break

        if lives_changed:
            terminated = True
        elif score_changed:
            start_s = time.time()
            elapsed_s = 0

            current_score = new_score
            while elapsed_s * 1000 < self.max_delay_rollback:
                elapsed_s = time.time() - start_s
                self.state_reader.read_game_values()
                new_score = self.state_reader.score
                if new_score != current_score:
                    current_score = new_score
                    start_s = time.time()

        truncated = False
        reward = (new_score - old_score)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info