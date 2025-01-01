import threading
from typing import Optional

import cv2
import numpy as np
import gymnasium as gym
import time

import pyautogui
import win32gui

from ZumaInterface.ImageProcessing import ImageProcessing
from ZumaInterface.StateReader import StateReader


class ZumaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_index):
        self.index = env_index
        self.render_mode = "rgb_array"

        self.state_reader = StateReader(env_index=env_index)

        self.step_delay_s = 0.3
        self.max_delay_ms = 1000
        self.max_delay_rollback = 1500
        self.playable = True
        self.reset_delay = 3
        self.reset_delay_start = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Box(0, 255, shape=(80, 112, 3), dtype=np.uint8)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32)

    def _get_obs(self):
        # self.state_reader.stack_frames(False)
        img = self.state_reader.screenshot_process()
        img = ImageProcessing.prepare_image(img)
        # img_arr = np.array(img)
        return img

    def _get_info(self):
        self.state_reader.read_game_values()
        score = self.state_reader.score

        return {
            "score": score
        }

    def _reset_delay(self):
        time.sleep(self.reset_delay)
        self.playable = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self.state_reader.score = 0
        # self.state_reader.progress = 0
        # self.state_reader.lives = 3
        # self.state_reader.write_game_values()
        # time.sleep(3)
        # self.state_reader.shoot_ball(180)
        # time.sleep(1)
        # self.state_reader.shoot_ball(180)
        # time.sleep(1.5)
        # self.state_reader.focus_lock_enable = False
        # self.state_reader.restart_level()
        # self.state_reader.focus_lock_enable = True

        # self.state_reader.restart_level()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        angle = np.interp(action, (-1.0, 1.0), (0.0, 360.0))
        # angle = action

        old_score = self.state_reader.score

        reward = -0.01
        terminated = False
        if time.time() - self.reset_delay_start > self.reset_delay:
            self.playable = True

        if self.playable:
            self.state_reader.shoot_ball(angle)
            time.sleep(self.step_delay_s/2)
            self.state_reader.reset_rotation()
            time.sleep(self.step_delay_s/2)

            self.state_reader.read_game_values()
            # new_lives = self.state_reader.lives
            if self.state_reader.lives < 3:
                self.state_reader.score = 0
                self.state_reader.progress = 0
                self.state_reader.lives = 3
                self.state_reader.write_game_values()
                self.playable = False
                self.reset_delay_start = time.time()
                reward = -1
            else:
                new_score = self.state_reader.score
                score_change = new_score - old_score
                # if score_change > 0:
                #     reward = 1
                if score_change > 100:
                    reward = 1
                elif score_change > 0:
                    reward = 0.5

        else:
            self.state_reader.shoot_ball(180)
            time.sleep(self.step_delay_s)

        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.state_reader.screenshot_process()
            img = ImageProcessing.prepare_image(img)

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