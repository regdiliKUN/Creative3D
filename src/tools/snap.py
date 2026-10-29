"""tools/snap.py —— 进程内截图：直接构造主窗口，等模型渲染好后截窗口区域。

比 tools/shot.py 更可靠（不走单实例互斥体、不依赖窗口标题），
用于改完界面后快速目视验收。

用法：
    python tools/snap.py out.png [等待秒数] [--concept cat] [--quality fine]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

os.environ.setdefault("C3D_NO_WELCOME", "1")

try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:  # noqa: BLE001
    pass

from PIL import ImageGrab  # noqa: E402

from creative3d.app import enable_dpi  # noqa: E402
from creative3d.settings import Settings  # noqa: E402
from creative3d.ui.mainwindow import MainWindow  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "snap.png")
    wait = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    enable_dpi()

    settings = Settings()
    if "--concept" in sys.argv:
        settings.set("concept", sys.argv[sys.argv.index("--concept") + 1])
        settings.set("mode", "concept")
    if "--quality" in sys.argv:
        settings.set("quality", sys.argv[sys.argv.index("--quality") + 1])

    win = MainWindow(settings)
    win.geometry(f"{min(1600, win.winfo_screenwidth() - 80)}x"
                 f"{min(1000, win.winfo_screenheight() - 120)}+40+30")
    win.deiconify()
    win.attributes("-topmost", True)
    win.lift()
    win.update()
    time.sleep(0.4)

    deadline = time.time() + wait
    while time.time() < deadline:
        win.update()
        time.sleep(0.05)

    win.update_idletasks()
    x, y = win.winfo_rootx(), win.winfo_rooty()
    w, h = win.winfo_width(), win.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
    img.save(out)
    print(f"截图已保存 {out} 尺寸 {img.size}")
    win.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
