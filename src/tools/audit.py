"""tools/audit.py —— 无头布局体检。

启动真实窗口（不弹欢迎页），遍历所有可见控件，找出：
  * 被父容器裁掉（请求宽度 > 实际宽度）的控件
  * 文字超过可用宽度而被截断的 Label
  * 尺寸为 0 的意外控件
输出一份简明报告，用来在改动布局后快速回归。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

os.environ.setdefault("C3D_NO_WELCOME", "1")

try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:  # noqa: BLE001
    pass

from creative3d.app import enable_dpi  # noqa: E402

enable_dpi()

import tkinter as tk  # noqa: E402

from creative3d.settings import Settings  # noqa: E402
from creative3d.ui.mainwindow import MainWindow  # noqa: E402

PROBLEMS: list[str] = []

from creative3d.ui.widgets import ScrollFrame  # noqa: E402


def _in_scrollframe(widget: tk.Widget) -> bool:
    """判断控件是否位于滚动容器内部（这类容器请求宽度大于可视宽度是正常的）。"""
    from creative3d.ui.widgets import ScrollFrame

    node = widget.master
    while node is not None:
        if isinstance(node, ScrollFrame):
            return True
        node = node.master
    return False


def walk(widget: tk.Widget, depth: int = 0) -> None:
    try:
        children = widget.winfo_children()
    except tk.TclError:
        return
    for child in children:
        try:
            if not child.winfo_ismapped():
                # 未映射（隐藏的标签页 / 尚未弹出的气泡）不算问题，也不必再往下遍历
                continue
            w = child.winfo_width()
            h = child.winfo_height()
            req_w = child.winfo_reqwidth()
            req_h = child.winfo_reqheight()
            name = child.winfo_class()
            text = ""
            if isinstance(child, (tk.Label, tk.Button, tk.Checkbutton, tk.Text)):
                try:
                    text = str(child.cget("text")).replace("\n", "⏎")
                except tk.TclError:
                    text = ""
            if w <= 1 or h <= 1:
                if name not in ("Frame", "Canvas", "TFrame"):
                    PROBLEMS.append(f"[零尺寸] {name} w={w} h={h} text={text[:30]!r}")
            elif req_w > w + 2 and req_w - w > 6:
                # 滚动容器内部的 Frame/Canvas 由 scrollregion 撑开，天然大于可视宽度
                if name in ("Frame", "Canvas", "TFrame") and (
                    isinstance(child, ScrollFrame) or _in_scrollframe(child)
                ):
                    pass
                else:
                    chain = []
                    node = child.master
                    while node is not None and len(chain) < 4:
                        chain.append(node.winfo_class())
                        node = node.master
                    PROBLEMS.append(
                        f"[被裁窄] {name} 需要 {req_w} 实际 {w} (差 {req_w - w}) "
                        f"text={text[:36]!r} 父链={'/'.join(chain)}"
                    )
            elif req_h > h + 2 and req_h - h > 6 and name in ("Label",):
                PROBLEMS.append(
                    f"[被裁高] {name} 需要 {req_h} 实际 {h} text={text[:36]!r}"
                )
        except tk.TclError:
            pass
        walk(child, depth + 1)


def main() -> int:
    settings = Settings()
    win = MainWindow(settings)

    def report() -> None:
        win.update_idletasks()
        print("=" * 70)
        print(f"窗口 {win.winfo_width()}x{win.winfo_height()}  缩放 {win.tk.call('tk', 'scaling'):.3f}")
        for attr in ("left", "right", "viewport"):
            w = getattr(win, attr, None)
            if w is not None:
                print(f"  {attr:9s} 宽 {w.winfo_width():4d} 高 {w.winfo_height():4d}")
        walk(win)
        print("-" * 70)
        if PROBLEMS:
            for line in PROBLEMS[:60]:
                print(line)
            print(f"共 {len(PROBLEMS)} 处疑似问题")
        else:
            print("未发现布局问题")
        print("=" * 70)
        win.destroy()

    win.after(1800, report)
    win.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
