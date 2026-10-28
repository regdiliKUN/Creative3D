"""ui.theme —— 全局样式：字体、ttk 皮肤、通用画笔。

所有颜色都来自 config.THEME，保证「界面 / 视口 / 导出图」是同一套色。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from .. import config

T = config.THEME


def init_root(root: tk.Tk) -> None:
    """初始化窗口级样式（必须在创建控件前调用）。"""
    config.ui_font_family()
    family = config.ui_font_family()

    root.configure(bg=T.bg)
    root.option_add("*Font", (family, 10))
    root.option_add("*tearOff", False)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("C3D.TEntry",
                    fieldbackground=T.panel_alt, background=T.panel_alt,
                    foreground=T.text, bordercolor=T.line, lightcolor=T.line,
                    darkcolor=T.line, insertcolor=T.accent, padding=(8, 5))
    style.map("C3D.TEntry",
              bordercolor=[("focus", T.accent)],
              lightcolor=[("focus", T.accent)],
              darkcolor=[("focus", T.accent)])

    style.configure("C3D.Vertical.TScrollbar",
                    background=T.panel_alt, troughcolor=T.bg, bordercolor=T.bg,
                    arrowcolor=T.text_dim, gripcount=0, arrowsize=10)
    style.map("C3D.Vertical.TScrollbar", background=[("active", T.line)])

    style.configure("C3D.TCombobox",
                    fieldbackground=T.panel_alt, background=T.panel_alt,
                    foreground=T.text, arrowcolor=T.text_dim,
                    bordercolor=T.line, padding=(6, 4))
    style.map("C3D.TCombobox",
              fieldbackground=[("readonly", T.panel_alt)],
              foreground=[("readonly", T.text)])
    root.option_add("*TCombobox*Listbox.background", T.panel_alt)
    root.option_add("*TCombobox*Listbox.foreground", T.text)
    root.option_add("*TCombobox*Listbox.selectBackground", T.accent_soft)

    # 允许把默认字体调大一点，中文更易读
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            f = tkfont.nametofont(name)
            f.configure(family=family, size=10)
        except tk.TclError:
            pass


def f(size: int = 10, bold: bool = False) -> tuple:
    return config.font(size, bold)


def mono(size: int = 9) -> tuple:
    return config.mono_font(size)


def rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, r=8, **kwargs):
    """Canvas 上的圆角矩形（用多边形近似）。"""
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


def mix(hex_a: str, hex_b: str, t: float) -> str:
    a = hex_a.lstrip("#")
    b = hex_b.lstrip("#")
    ca = tuple(int(a[i:i + 2], 16) for i in (0, 2, 4))
    cb = tuple(int(b[i:i + 2], 16) for i in (0, 2, 4))
    out = tuple(int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*out)


def lighten(hex_color: str, amount: float = 0.12) -> str:
    return mix(hex_color, "#ffffff", max(0.0, min(1.0, amount)))
