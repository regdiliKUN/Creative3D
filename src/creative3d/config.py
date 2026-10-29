"""config.py —— 全局常量、主题配色、路径解析。

设计约束：
1. 所有「写文件」的位置都必须可写：程序目录可能是只读的（例如放在
   Program Files 或只读 U 盘），因此日志 / 设置 / 缓存一律写到
   %LOCALAPPDATA%\\Creative3D；导出目录默认用程序目录下的 exports，
   若不可写则自动退回到 %LOCALAPPDATA%\\Creative3D\\exports。
2. 颜色只在这里定义一次，界面、视口、导出图、图标共用同一套，
   避免各模块「各自调色」导致观感割裂。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "3D 建模创意生成器"
APP_SHORT = "Creative3D"
APP_VERSION = "3.0.0"
APP_BUILD = "2026.10"
APP_TAGLINE = "任意提示词 · 高精度建模 · 打印体检 · 一键导出"

MIN_WINDOW_SIZE = (1200, 780)
DEFAULT_WINDOW_SIZE = (1500, 940)

# 模型统一归一化到该尺寸（毫米），界面、导出、创意卡共用同一口径
MODEL_TARGET_MM = 100.0
DEFAULT_WALL_MM = 1.4

MAX_IMAGE_PIXELS = 40_000_000  # 超过此像素量的图片先降采样，避免内存爆掉
MAX_EXPORT_PIXELS = 4096 * 4096


@dataclass(frozen=True)
class UITheme:
    """界面配色（深色）。"""

    bg: str = "#0c1017"
    bg_alt: str = "#11161f"
    panel: str = "#151b25"
    panel_alt: str = "#1a212d"
    card: str = "#181f2b"
    line: str = "#252e3d"
    line_soft: str = "#1e2532"
    text: str = "#e4eaf5"
    text_dim: str = "#9aa6ba"
    text_mute: str = "#6d7889"
    accent: str = "#4d9cff"
    accent_hover: str = "#6cb0ff"
    accent_soft: str = "#1b3355"
    accent2: str = "#8b5cff"
    ok: str = "#37d399"
    warn: str = "#f5b544"
    err: str = "#ff6b6b"
    sel: str = "#1d3452"
    thumb: str = "#39465c"  # 滚动指示条


THEME = UITheme()


@dataclass(frozen=True)
class ModelPalette:
    """模型着色使用的三段色 + 视口背景渐变。"""

    id: str
    label: str
    shadow: tuple[int, int, int]
    base: tuple[int, int, int]
    light: tuple[int, int, int]
    bg_top: tuple[int, int, int]
    bg_bottom: tuple[int, int, int]

    def hex(self, which: str = "base") -> str:
        r, g, b = getattr(self, which)
        return f"#{r:02x}{g:02x}{b:02x}"


PALETTES: dict[str, ModelPalette] = {
    "frost": ModelPalette(
        "frost", "霜蓝",
        (18, 38, 68), (92, 158, 226), (206, 234, 255),
        (16, 24, 38), (8, 11, 17),
    ),
    "magma": ModelPalette(
        "magma", "熔岩",
        (58, 16, 18), (204, 92, 54), (255, 214, 158),
        (34, 18, 18), (11, 9, 12),
    ),
    "moss": ModelPalette(
        "moss", "苔原",
        (20, 44, 34), (98, 168, 128), (208, 246, 214),
        (16, 30, 26), (8, 12, 12),
    ),
    "lunar": ModelPalette(
        "lunar", "月白",
        (48, 52, 66), (182, 190, 208), (255, 255, 255),
        (26, 28, 38), (10, 11, 16),
    ),
    "amethyst": ModelPalette(
        "amethyst", "紫晶",
        (42, 24, 72), (139, 92, 255), (226, 210, 255),
        (28, 20, 44), (10, 9, 16),
    ),
    "amber": ModelPalette(
        "amber", "琥珀",
        (58, 36, 12), (222, 158, 62), (255, 236, 186),
        (32, 24, 14), (12, 10, 9),
    ),
    "steel": ModelPalette(
        "steel", "铁灰",
        (32, 38, 48), (128, 142, 162), (226, 236, 248),
        (20, 24, 32), (8, 10, 14),
    ),
    "coral": ModelPalette(
        "coral", "珊瑚",
        (66, 24, 38), (232, 112, 132), (255, 218, 224),
        (34, 18, 26), (12, 9, 13),
    ),
}

DEFAULT_PALETTE = "frost"


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """包体根目录：打包后 = exe 所在目录；源码运行时 = 工程根目录。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    """内置资源（图标等）所在目录。

    打包后用 PyInstaller 的临时解包目录 _MEIPASS；
    源码运行时资源在 src/ 下（与包目录同级），而不是工程根。
    """
    if is_frozen():
        base = Path(getattr(sys, "_MEIPASS", app_dir()))
    else:
        base = Path(__file__).resolve().parents[1]
    return base.joinpath(*parts)


def user_data_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    base = Path(root) if root else Path.home() / ".local" / "share"
    return base / APP_SHORT


def log_dir() -> Path:
    return user_data_dir() / "logs"


def cache_dir() -> Path:
    return user_data_dir() / "cache"


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def default_export_dir() -> Path:
    """优先包体内 exports，不可写时退回用户目录。"""
    preferred = app_dir() / "exports"
    if _is_writable(preferred):
        return preferred
    return ensure_dir(user_data_dir() / "exports")


def exports_dir(configured: str | None = None) -> Path:
    if configured:
        candidate = Path(configured)
        if _is_writable(candidate):
            return candidate
    return default_export_dir()


# --------------------------------------------------------------------------
# 字体
# --------------------------------------------------------------------------

_UI_FONT_CACHE: str | None = None


def ui_font_family() -> str:
    """挑一个能正常显示中文的无衬线字体。"""
    global _UI_FONT_CACHE
    if _UI_FONT_CACHE:
        return _UI_FONT_CACHE
    candidates = ["Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑", "Segoe UI"]
    try:
        import tkinter as tk
        from tkinter import font as tkfont

        root = tk._default_root
        if root is not None:
            available = set(tkfont.families(root))
            for name in candidates:
                if name in available:
                    _UI_FONT_CACHE = name
                    return name
    except Exception:  # noqa: BLE001 - 字体探测失败不应中断启动
        pass
    _UI_FONT_CACHE = "Microsoft YaHei UI"
    return _UI_FONT_CACHE


def font(size: int = 10, bold: bool = False, family: str | None = None) -> tuple:
    return (family or ui_font_family(), size, "bold" if bold else "normal")


def mono_font(size: int = 9) -> tuple:
    return ("Consolas", size, "normal")


_SCALE_CACHE: float | None = None


def ui_scale() -> float:
    """界面缩放系数（相对 96 DPI）。

    Tk 的控件宽度是「物理像素」，而字体由 tk scaling 放大；
    高分屏上如果不按比例放大面板宽度，文字就会把面板撑爆。
    """
    global _SCALE_CACHE
    if _SCALE_CACHE is not None:
        return _SCALE_CACHE
    scale = 1.0
    try:
        import tkinter as tk

        root = tk._default_root
        if root is not None:
            scale = float(root.tk.call("tk", "scaling")) / 1.3333
    except Exception:  # noqa: BLE001 - 探测失败就按 1.0 处理
        scale = 1.0
    _SCALE_CACHE = max(1.0, min(2.6, scale))
    return _SCALE_CACHE


def sp(value: float) -> int:
    """把设计稿里的像素值换算成当前 DPI 下的像素。"""
    return int(round(value * ui_scale()))
