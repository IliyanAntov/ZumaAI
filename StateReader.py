import math
import time
from collections import deque
from time import sleep

import numpy as np
import win32gui
import win32ui
import win32con
import pygetwindow as gw
import pyautogui
from skimage import transform  # Help us to preprocess the frames
from PIL import Image, ImageDraw

from pyMeow import *


from pynput.keyboard import Key, Listener


class StateReader:
    def __init__(self, level):
        self.level = level
        self.frog_positions_raw = [(242, 248)]
        self.level_score_limits = [2500]

        self.window_name = "Zuma Deluxe 1.0"
        self.process = open_process("game.exe")
        self.window = next((win for win in gw.getWindowsWithTitle(self.window_name) if win.visible), None)
        self.hwnd = self.window._hWnd
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.1)

        self.hwindc = win32gui.GetWindowDC(self.hwnd)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()

        # self.stacked_frames_count = 4
        # self.stacked_frames = deque([np.zeros((480, 640, 3), dtype=np.uint8) for i in range(self.stacked_frames_count)], maxlen=4)
        # self.stacked_state = np.stack(self.stacked_frames, axis=2)

        self.client_left = None
        self.client_top = None
        self.client_right = None
        self.client_bottom = None
        client_rect = win32gui.GetClientRect(self.hwnd)  # Get client area dimensions relative to window
        self.client_left, self.client_top = win32gui.ClientToScreen(self.hwnd, (client_rect[0], client_rect[1]))
        self.client_right, self.client_bottom = win32gui.ClientToScreen(self.hwnd, (client_rect[2], client_rect[3]))

        self.width = self.client_right - self.client_left
        self.height = self.client_bottom - self.client_top

        self.score_addr = None
        self.progress_addr = None
        self.lives_addr = None
        self._get_addresses()

        self.score = None
        self.progress = None
        self.lives = None
        self.read_game_values()

        self.frog_x = None
        self.frog_y = None
        self.update_frog_coords(self.level)

    def _get_addresses(self):
        base_address = get_module(self.process, "game.exe")["base"]
        score_base_addr = base_address + 0x00133DA0
        score_offsets = [0x4, 0x0, 0x14, 0x34, 0x24, 0xC0, 0x60]
        self.score_addr = StateReader._read_offsets(self.process, score_base_addr, score_offsets)

        progress_base_addr = base_address + 0x0017FF54
        progress_offsets = [0x58, 0xC0, 0x8, 0x58, 0x48, 0x140]
        self.progress_addr = StateReader._read_offsets(self.process, progress_base_addr, progress_offsets)

        lives_base_addr = base_address + 0x0017FF54
        lives_offsets = [0x60, 0x88, 0x8, 0x98, 0x48, 0xC0]
        self.lives_addr = StateReader._read_offsets(self.process, lives_base_addr, lives_offsets)

    @staticmethod
    def _read_offsets(proc, base_addr, offsets):
        current_pointer = r_int(proc, base_addr)

        for offset in offsets[:-1]:
            current_pointer = r_int(proc, current_pointer + offset)
        return current_pointer + offsets[-1]

    def read_game_values(self):
        self.score = int(r_int(self.process, self.score_addr))
        self.progress = int(r_int(self.process, self.progress_addr))
        self.lives = int(r_int(self.process, self.lives_addr))

    def write_game_values(self):
        w_int(self.process, self.score_addr, self.score)
        w_int(self.process, self.progress_addr, self.progress)
        w_int(self.process, self.lives_addr, self.lives)

    def shoot_ball(self, angle_deg, radius=60):
        angle_rad = math.radians(angle_deg)

        dx = math.cos(angle_rad) * radius
        dy = math.sin(angle_rad) * radius
        pyautogui.leftClick(self.frog_x + dx, self.frog_y + dy)

    def update_frog_coords(self, level_index):

        self.frog_x = self.frog_positions_raw[level_index][0] + self.client_left
        self.frog_y = self.frog_positions_raw[level_index][1] + self.client_top

    def screenshot_process(self):

        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)
        self.memdc.SelectObject(bmp)

        # Adjust BitBlt to start from the client area
        self.memdc.BitBlt((0, 0),
                          (self.width, self.height),
                          self.srcdc,
                          (self.client_left - self.window.left, self.client_top - self.window.top),
                          win32con.SRCCOPY)

        # Convert the raw data to a PIL image
        bmpinfo = bmp.GetInfo()
        bmpstr = bmp.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # img = StateReader.transform_image_to_2bit_rgb(img)

        transformed_image = StateReader.transform_image_to_2bit_rgb(img)

        win32gui.DeleteObject(bmp.GetHandle())

        # preprocessed_frame = transform.resize(img, [84, 84])

        return transformed_image

    # def stack_frames(self, is_new_episode):
    #
    #     frame = self._screenshot_process()
    #
    #     if is_new_episode:
    #         self.stacked_frames = deque([np.zeros((480, 640, 3), dtype=np.uint8) for i in range(self.stacked_frames_count)], maxlen=4)
    #         self.stacked_frames.append(frame)
    #         self.stacked_frames.append(frame)
    #         self.stacked_frames.append(frame)
    #         self.stacked_frames.append(frame)
    #         self.stacked_state = np.stack(self.stacked_frames, axis=0)
    #
    #     else:
    #         self.stacked_frames.append(frame)
    #         self.stacked_state = np.stack(self.stacked_frames, axis=0)

    @staticmethod
    def transform_image_to_2bit_rgb(image) -> np.stack:

        # Convert the image to a NumPy array
        img_array = np.array(image)

        # Reduce each channel to 2 bits
        r_2bit = (img_array[:, :, 0] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range
        g_2bit = (img_array[:, :, 1] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range
        b_2bit = (img_array[:, :, 2] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range

        # Combine the channels back into an image
        transformed_array = np.stack((r_2bit, g_2bit, b_2bit), axis=-1)
        # transformed_image = Image.fromarray(transformed_array.astype('uint8'), mode="RGB")

        return transformed_array

    def __del__(self):
        # win32gui.ReleaseDC(self.hwnd, self.hwindc)
        # self.memdc.DeleteDC()
        self.srcdc.DeleteDC()

    @staticmethod
    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

