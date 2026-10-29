"""tools/shot.py —— 开发期截图助手：启动界面、置前、按窗口裁切截图、退出。

用法：
    python tools/shot.py out.png [wait_seconds] [--keep] [--script 动作脚本]
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import ImageGrab

SRC = Path(__file__).resolve().parents[1]

# 截图进程必须自己也声明 DPI 感知，否则 GetWindowRect 拿到的是被虚拟化的坐标
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:  # noqa: BLE001
    pass
sys.path.insert(0, str(SRC))
from creative3d import config as _cfg  # noqa: E402

TITLE = f"{_cfg.APP_NAME} v{_cfg.APP_VERSION}"


def find_window(title: str) -> int:
    user32 = ctypes.windll.user32
    return int(user32.FindWindowW(None, title))


def focus(hwnd: int) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW = 0x0002, 0x0001, 0x0040
    user32.ShowWindow(hwnd, 9)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.8)
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "shot.png")
    wait = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    keep = "--keep" in sys.argv

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["C3D_NO_WELCOME"] = "1"
    proc = subprocess.Popen([sys.executable, "-m", "creative3d.app"], env=env, cwd=str(SRC))
    try:
        hwnd = 0
        deadline = time.time() + 20
        while time.time() < deadline:
            hwnd = find_window(TITLE)
            if hwnd:
                break
            if proc.poll() is not None:
                print("进程已退出，退出码", proc.returncode)
                return 1
            time.sleep(0.3)
        if not hwnd:
            print("没有找到窗口")
            return 1
        time.sleep(wait)
        box = focus(hwnd)
        time.sleep(0.8)
        img = ImageGrab.grab(bbox=box, all_screens=True)
        img.save(out)
        print(f"截图已保存 {out} 尺寸 {img.size}")
    finally:
        if not keep:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
