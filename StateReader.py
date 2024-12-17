import math
import time
from collections import deque
from threading import Thread
from time import sleep

import numpy as np
import psutil
import win32api
import win32gui
import win32process
import win32ui
import win32con
import pygetwindow as gw
import pyautogui
from PIL.Image import Resampling
from skimage import transform  # Help us to preprocess the frames
from PIL import Image, ImageDraw
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

from pyMeow import *
np.seterr(all='raise')


class StateReader:
    def __init__(self, env_index=0, level=0):
        print("Env index: ", str(env_index))
        self.level = level
        self.frog_positions_raw = [(242, 248)]
        self.level_score_limits = [2500]
        self.focus_lock_enable = True

        window_list = gw.getWindowsWithTitle("Zuma Deluxe 1.0")
        self.window = window_list[env_index]
        self.hwnd = self.window._hWnd
        win32process.GetWindowThreadProcessId(self.hwnd)
        # self.pid = game_process_list[env_index].pid
        self.pid = win32process.GetWindowThreadProcessId(self.hwnd)[1]
        self.process = open_process(self.pid)

        # win32gui.SetForegroundWindow(self.hwnd)
        # time.sleep(0.5)

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
        self.rotation_addr = None
        self.focus_loss_addr = None
        self._get_addresses()
        self._focus_loss_lock()

        self.score = None
        self.progress = None
        self.lives = None
        self.rotation = None
        self.read_game_values()

        self.frog_x = None
        self.frog_y = None
        self.update_frog_coords(self.level)

    def _get_addresses(self):
        base_address = self.process["base"]
        score_base_addr = base_address + 0x0017FF54
        score_offsets = [0x58, 0xC0, 0x0, 0xA0, 0x70, 0x8, 0xB8]
        self.score_addr = StateReader._read_offsets(self.process, score_base_addr, score_offsets)

        progress_base_addr = base_address + 0x0017FF54
        progress_offsets = [0x58, 0xC0, 0x8, 0x58, 0x48, 0x140]
        self.progress_addr = StateReader._read_offsets(self.process, progress_base_addr, progress_offsets)

        lives_base_addr = base_address + 0x0017FF54
        lives_offsets = [0x60, 0x88, 0x8, 0x98, 0x48, 0xC0]
        self.lives_addr = StateReader._read_offsets(self.process, lives_base_addr, lives_offsets)

        rotation_base_addr = base_address + 0x00133DA0
        rotation_offsets = [0x4, 0x4, 0x14, 0x34, 0x1FC, 0x4]
        self.rotation_addr = StateReader._read_offsets(self.process, rotation_base_addr, rotation_offsets)

        focus_loss_base_addr = base_address + 0x0017FF54
        focus_loss_offsets = [0x58, 0xC8, 0x8, 0x5C, 0x48, 0xCC]
        self.focus_loss_addr = StateReader._read_offsets(self.process, focus_loss_base_addr, focus_loss_offsets)

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
        # TODO
        # self.rotation = float(r_float(self.process, self.rotation_addr))

    def write_game_values(self):
        w_int(self.process, self.score_addr, self.score)
        w_int(self.process, self.progress_addr, self.progress)
        w_int(self.process, self.lives_addr, self.lives)

    def reset_rotation(self):
        w_float(self.process, self.rotation_addr, 0.0)

    def _focus_loss_thread(self):
        while True:
            if self.focus_lock_enable:
                w_int(self.process, self.focus_loss_addr, 0)
            time.sleep(0.1)

    def _focus_loss_lock(self):
        focus_loss_thread = Thread(target=self._focus_loss_thread)
        focus_loss_thread.start()
        # w_int(self.process, self.focus_loss_addr, 0)

    def shoot_ball(self, angle_deg, radius=60):
        angle_rad = math.radians(angle_deg)

        dx = math.cos(angle_rad) * radius
        dy = math.sin(angle_rad) * radius

        l_param = win32api.MAKELONG(int(self.frog_x + dx), int(self.frog_y + dy))

        win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, l_param)
        win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, l_param)

        # pyautogui.leftClick(self.frog_x + dx, self.frog_y + dy)

    def restart_level(self):
        time.sleep(0.1)
        self.score = 10
        self.lives = 3
        self.write_game_values()
        time.sleep(0.1)

        # x, y, delay after acton
        action_sequences = [(576, 15, 0.1),
                            (310, 308, 0.1),
                            (250, 268, 0.1),
                            (518, 103, 0.1),
                            (390, 322, 0.3),
                            (323, 326, 0.1)]
        for action_sequence in action_sequences:
            x = action_sequence[0]
            y = action_sequence[1]
            delay = action_sequence[2]
            l_param = win32api.MAKELONG(int(x), int(y))
            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, l_param)
            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, l_param)
            time.sleep(delay)

        self.shoot_ball(180)
        time.sleep(1.5)

    def life_lost_actions(self):
        self.score = 0
        self.progress = 0
        self.lives = 3
        self.write_game_values()
        time.sleep(3)
        self.shoot_ball(180)
        time.sleep(1)
        self.shoot_ball(180)
        time.sleep(1.5)

    def update_frog_coords(self, level_index):

        # self.frog_x = self.frog_positions_raw[level_index][0] + self.client_left
        # self.frog_y = self.frog_positions_raw[level_index][1] + self.client_top
        self.frog_x = self.frog_positions_raw[level_index][0]
        self.frog_y = self.frog_positions_raw[level_index][1]

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
        img = img.crop((15, 30, bmpinfo['bmWidth']-15, bmpinfo['bmHeight']-15))

        # img = StateReader.transform_image_to_2bit_rgb(img)

        # transformed_image = StateReader.transform_image_to_2bit_rgb(img)

        win32gui.DeleteObject(bmp.GetHandle())

        img = img.resize((40, 40))

        return img

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
        transformed_image = Image.fromarray(transformed_array.astype('uint8'), mode="RGB")
        resized_image = transformed_image.resize((84, 84))

        return resized_image

    # def __del__(self):
        # win32gui.ReleaseDC(self.hwnd, self.hwindc)
        # self.memdc.DeleteDC()
        # self.srcdc.DeleteDC()

