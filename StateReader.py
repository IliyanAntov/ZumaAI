import time
from time import sleep

import numpy as np
import win32gui
import win32ui
import win32con
import pygetwindow as gw
import pyautogui
from PIL import Image, ImageDraw

from pyMeow import *


def move_mouse_to(x, y, window_left, window_top):
    screen_x = window_left + x
    screen_y = window_top + y
    pyautogui.moveTo(screen_x, screen_y)


def read_offsets(proc, base_addr, offsets):
    current_pointer = r_int(proc, base_addr)

    for offset in offsets[:-1]:
        current_pointer = r_int(proc, current_pointer + offset)

    return current_pointer + offsets[-1]


def screenshot_process(process_name):

    window = next((win for win in gw.getWindowsWithTitle(process_name) if win.visible), None)

    if not window:
        print(f"No visible window found for process '{process_name}'.")
        return None

    hwnd = window._hWnd
    win32gui.SetForegroundWindow(hwnd)

    # Get the client area dimensions
    client_rect = win32gui.GetClientRect(hwnd)  # Get client area dimensions relative to window
    client_left, client_top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
    client_right, client_bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))

    # Calculate width and height of the client area
    width = client_right - client_left
    height = client_bottom - client_top

    move_mouse_to(*frog_positions_raw[0], client_left, client_top)

    # Capture the client area using a device context
    hwindc = win32gui.GetWindowDC(hwnd)  # Retrieves the device context for the window
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)

    # Adjust BitBlt to start from the client area
    memdc.BitBlt((0, 0), (width, height), srcdc,
                 (client_left - window.left, client_top - window.top),
                 win32con.SRCCOPY)

    # Convert the raw data to a PIL image
    bmpinfo = bmp.GetInfo()
    bmpstr = bmp.GetBitmapBits(True)
    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )
    draw = ImageDraw.Draw(img)

    draw.circle(frog_positions_raw[0], 60, outline='red')

    img = transform_image_to_2bit_rgb(img)
    # Cleanup
    win32gui.ReleaseDC(hwnd, hwindc)
    memdc.DeleteDC()
    # srcdc.DeleteDC()
    win32gui.DeleteObject(bmp.GetHandle())

    return img


def transform_image_to_2bit_rgb(image):

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Reduce each channel to 2 bits
    r_2bit = (img_array[:, :, 0] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range
    g_2bit = (img_array[:, :, 1] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range
    b_2bit = (img_array[:, :, 2] >> 7) << 7  # Top 1 bits, scaled back to 8-bit range

    # Combine the channels back into an image
    transformed_array = np.stack((r_2bit, g_2bit, b_2bit), axis=-1)
    transformed_image = Image.fromarray(transformed_array.astype('uint8'), mode="RGB")

    return transformed_image


frog_positions_raw = [(242, 248), (334, 247)]
# frog_positions_circle = [(x + 3, y + 26) for x, y in frog_positions_raw]
# frog_positions_mouse = [(int(x*1.295), int(y*1.295)) for x, y in frog_positions_raw]

if __name__ == "__main__":

    time.sleep(2)
    for i in range(4):
        img = screenshot_process("Zuma Deluxe 1.0")
        # img.show()
        img.save(str(i) + ".png")

    # process = open_process("game.exe")
    # base_address = get_module(process, "game.exe")["base"]
    #
    # score_base_addr = base_address + 0x00133DA0
    # score_offsets = [0x4, 0x0, 0x14, 0x34, 0x24, 0xC0, 0x60]
    # score_addr = read_offsets(process, score_base_addr, score_offsets)
    #
    # progress_base_addr = base_address + 0x0017FF54
    # progress_offsets = [0x58, 0xC0, 0x8, 0x58, 0x48, 0x140]
    # progress_addr = read_offsets(process, progress_base_addr, progress_offsets)
    #
    # lives_base_addr = base_address + 0x0017FF54
    # lives_offsets = [0x60, 0x88, 0x8, 0x98, 0x48, 0xC0]
    # lives_addr = read_offsets(process, lives_base_addr, lives_offsets)
    #
    # while 1:
    #     print("Score: " + str(r_int(process, score_addr)) +
    #           "  Progress: " + str(r_int(process, progress_addr)) + "/256" +
    #           "  Lives: " + str(r_int(process, lives_addr)) + "/3")
    #     sleep(1)


