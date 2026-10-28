"""imaging.py —— 图像分析（多模态输入的核心）。

能力：
1. 读取 PNG / JPG / WEBP / BMP / GIF / TIFF（含 EXIF 方向修正、超大图自动降采样）；
2. 主色提取（k-means）→ 可套用到视口与导出图，也可直接驱动配色；
3. 主体轮廓提取 → 供「轮廓旋转体」使用；
4. 亮度高度场 → 供「图像浮雕」使用；
5. 统计量（对称性 / 对比度 / 边缘密度）→ 转成可读的「图像解读」提示。

所有函数都做了防御：任何异常都转成 ImageError，带中文说明与建议。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import config
from .errors import ImageError

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

_COLOR_NAMES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("红", (198, 62, 58)),
    ("橙", (214, 128, 48)),
    ("琥珀", (206, 158, 72)),
    ("黄", (216, 196, 74)),
    ("青柠", (150, 190, 86)),
    ("绿", (86, 168, 110)),
    ("松石", (62, 164, 158)),
    ("青", (74, 156, 190)),
    ("蓝", (70, 122, 208)),
    ("靛", (74, 82, 172)),
    ("紫", (132, 92, 190)),
    ("品红", (188, 82, 158)),
    ("粉", (218, 132, 152)),
    ("棕", (132, 96, 72)),
    ("灰", (132, 138, 148)),
    ("银", (190, 196, 206)),
    ("黑", (26, 28, 34)),
    ("白", (244, 246, 250)),
)


@dataclass
class Swatch:
    rgb: tuple[int, int, int]
    ratio: float

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)

    @property
    def name(self) -> str:
        return color_name(self.rgb)

    @property
    def label(self) -> str:
        return f"{self.name} {self.hex.upper()}"


def color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best, best_d = "灰", 1e9
    for name, (cr, cg, cb) in _COLOR_NAMES:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = name, d
    mx, mn = max(rgb), min(rgb)
    light = (r + g + b) / 3.0
    if mx - mn < 26:
        if light < 58:
            return "深" + best if best != "黑" else "黑"
        if light > 206:
            return "亮" + best if best != "白" else "白"
        return best
    if light < 72:
        return "深" + best
    if light > 200:
        return "浅" + best
    return best


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------


def load_image(path: str | os.PathLike, *, max_pixels: int = config.MAX_IMAGE_PIXELS) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise ImageError("找不到这张图片。", hint="可能已被移动或删除，请重新选择。", detail=str(p))
    if p.suffix.lower() not in SUPPORTED_EXT:
        raise ImageError(
            f"不支持 {p.suffix or '该'} 格式。",
            hint="请使用 PNG / JPG / WEBP / BMP / GIF / TIFF。",
            detail=str(p),
        )
    try:
        with Image.open(p) as raw:
            img = ImageOps.exif_transpose(raw)
            img = img.convert("RGBA")
            w, h = img.size
            if w * h > max_pixels:
                k = math.sqrt(max_pixels / float(w * h))
                img = img.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError(
            "这张图片读取失败，文件可能已损坏。",
            hint="用看图软件另存为 PNG 后再试一次。",
            detail=f"{p}: {exc}",
        ) from exc
    if img.width < 8 or img.height < 8:
        raise ImageError("图片太小，无法提取有效轮廓。", hint="请换一张长宽都大于 8 像素的图片。")
    return img


def from_clipboard() -> Image.Image | None:
    """从剪贴板拿图片；没有图片时返回 None（不抛异常）。"""
    try:
        from PIL import ImageGrab

        data = ImageGrab.grabclipboard()
        if isinstance(data, Image.Image):
            return ImageOps.exif_transpose(data).convert("RGBA")
        if isinstance(data, list) and data:
            for item in data:
                if Path(item).suffix.lower() in SUPPORTED_EXT:
                    return load_image(item)
    except Exception:  # noqa: BLE001 - 剪贴板不可用是完全正常的情况
        return None
    return None


# --------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------


@dataclass
class ImageFeatures:
    name: str
    width: int
    height: int
    swatches: list[Swatch]
    gray: np.ndarray  # 小尺寸灰度图 (h, w) float32 0..1
    alpha: np.ndarray | None = None  # 主体蒙版（若原图有透明通道）
    notes: list[str] = field(default_factory=list)
    source_path: str = ""

    # ------------------------------------------------------------ 派生
    @property
    def colors(self) -> list[str]:
        return [s.hex.upper() for s in self.swatches]

    @property
    def dominant(self) -> tuple[int, int, int]:
        return self.swatches[0].rgb if self.swatches else (120, 150, 190)

    @property
    def contrast(self) -> float:
        return float(self.gray.std())

    @property
    def symmetry(self) -> float:
        g = self.gray
        return float(1.0 - np.abs(g - np.fliplr(g)).mean())

    @property
    def edge_density(self) -> float:
        gx = np.abs(np.diff(self.gray, axis=1)).mean() if self.gray.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(self.gray, axis=0)).mean() if self.gray.shape[0] > 1 else 0.0
        return float((gx + gy) * 0.5)

    # ------------------------------------------------------------ 给生成器用
    def heightmap(self, size: int) -> np.ndarray:
        """返回 (size, size) 的归一化高度场，等比放置（含黑边），并做轻度去噪。"""
        src = Image.fromarray((np.clip(self.gray, 0, 1) * 255).astype(np.uint8), "L")
        canvas = Image.new("L", (size, size), 0)
        scale = min(size / src.width, size / src.height)
        w = max(1, int(src.width * scale))
        h = max(1, int(src.height * scale))
        resized = src.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(max(0.4, size / 220.0)))
        canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo)
        return arr

    def radial_profile(self, count: int) -> np.ndarray:
        """按「从上到下」的行主角宽度提取旋转体轮廓，返回 0..1 的半径序列。"""
        mask = self._subject_mask()
        h, w = mask.shape
        rows = np.linspace(0, h - 1, count).astype(int)
        radii = np.zeros(count, dtype=np.float64)
        for i, y in enumerate(rows):
            xs = np.nonzero(mask[y])[0]
            if len(xs) >= 2:
                radii[i] = (xs[-1] - xs[0] + 1)
            else:
                radii[i] = 0.0
        # 行顺序：图像顶部 → 底部；旋转体需要底部 → 顶部，因此翻转
        radii = radii[::-1]
        peak = float(radii.max())
        if peak <= 1e-6:
            return np.linspace(0.35, 0.35, count)
        radii = radii / peak
        # 轻度平滑，去掉单行噪声
        k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        k /= k.sum()
        padded = np.pad(radii, 2, mode="edge")
        return np.convolve(padded, k, mode="valid")

    def _subject_mask(self) -> np.ndarray:
        if self.alpha is not None:
            return self.alpha > 0.5
        g = self.gray
        # 以四边中位亮度为背景估计，差异超过阈值的视为主体
        border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
        bg = float(np.median(border))
        diff = np.abs(g - bg)
        thr = max(0.10, float(diff.std()) * 1.2)
        mask = diff > thr
        # 若检出面积过大/过小，退回 Otsu 风格阈值
        ratio = mask.mean()
        if ratio > 0.92 or ratio < 0.04:
            thr = float(np.quantile(g, 0.45))
            mask = np.abs(g - bg) > abs(thr - bg) * 0.8
        return mask

    def summary(self) -> str:
        sym = "左右对称" if self.symmetry > 0.82 else ("略不对称" if self.symmetry > 0.62 else "明显不对称")
        con = "高对比" if self.contrast > 0.26 else ("中等对比" if self.contrast > 0.14 else "低对比")
        edge = "细节密集" if self.edge_density > 0.035 else ("细节适中" if self.edge_density > 0.015 else "大面积平涂")
        return f"{self.width}×{self.height} · {sym} · {con} · {edge}"


def _kmeans(pixels: np.ndarray, k: int, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    n = len(pixels)
    k = max(1, min(k, n))
    # 均匀抽样做确定性初始化，避免每次结果跳动
    idx = np.linspace(0, n - 1, k).astype(int)
    centers = pixels[idx].astype(np.float64)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d, axis=1)
        for c in range(k):
            sel = labels == c
            if sel.any():
                centers[c] = pixels[sel].mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return centers, counts / max(counts.sum(), 1.0)


def analyze(img: Image.Image, *, name: str = "图片", path: str = "", palette_size: int = 5) -> ImageFeatures:
    """把 PIL 图变成可用的特征对象。"""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    # 太小的图提取不出轮廓也提不出配色，早点给一句人话，别让用户对着空白发呆
    if min(w, h) < 16 or max(w, h) < 24:
        raise ImageError(
            f"这张图片只有 {w}×{h} 像素，太小了。",
            hint="换一张长宽都大于 100 像素的图片，轮廓和配色才提得准。",
            detail=f"image too small: {w}x{h}",
        )

    # 灰色小图（长边 ≤ 256）用于高度场 / 轮廓 / 统计
    scale = 256.0 / max(w, h)
    small = rgba.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.LANCZOS)

    rgb = Image.new("RGB", small.size, (14, 18, 26))
    rgb.paste(small, mask=small.split()[3])
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    alpha_arr = None
    a = small.split()[3]
    if a.getextrema()[0] < 250:  # 存在透明像素才当作蒙版
        alpha_arr = np.asarray(a, dtype=np.float32) / 255.0

    # 主色：忽略过暗/过白的像素，避免背景吃掉配色
    flat = arr.reshape(-1, 3)
    lum = flat @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    keep = (lum > 0.06) & (lum < 0.97)
    sample = flat[keep] if keep.sum() >= 32 else flat
    if len(sample) > 20000:
        step = len(sample) // 20000 + 1
        sample = sample[::step]
    centers, ratios = _kmeans(sample, palette_size)
    order = np.argsort(-ratios)
    swatches = [
        Swatch(tuple(int(round(v * 255)) for v in np.clip(centers[i], 0, 1)), float(ratios[i]))
        for i in order
    ]

    feats = ImageFeatures(
        name=name,
        width=w,
        height=h,
        swatches=swatches,
        gray=gray,
        alpha=alpha_arr,
        source_path=path,
    )

    notes: list[str] = []
    if feats.symmetry > 0.82:
        notes.append("左右对称")
    if feats.contrast > 0.26:
        notes.append("高对比")
    elif feats.contrast < 0.14:
        notes.append("低对比")
    if feats.edge_density > 0.035:
        notes.append("细节密集")
    if alpha_arr is not None:
        notes.append("带透明通道")
    if not notes:
        notes.append("构图均衡")
    feats.notes = notes
    return feats


def analyze_path(path: str | os.PathLike) -> ImageFeatures:
    p = Path(path)
    img = load_image(p)
    return analyze(img, name=p.name, path=str(p))


# --------------------------------------------------------------------------
# 配色 → 模型配色
# --------------------------------------------------------------------------


def _mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _luma(rgb) -> float:
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0


def palette_from_features(feats: ImageFeatures) -> config.ModelPalette:
    """从图片主色派生一套模型配色（三段色 + 背景渐变）。"""
    if not feats.swatches:
        return config.PALETTES[config.DEFAULT_PALETTE]

    # 选一个「够亮、够饱和」的颜色作为主体色，避免全黑/全白
    def score(s: Swatch) -> float:
        r, g, b = s.rgb
        sat = (max(r, g, b) - min(r, g, b)) / 255.0
        lum = _luma(s.rgb)
        balance = 1.0 - abs(lum - 0.55) * 1.4
        return sat * 0.8 + balance * 0.6 + s.ratio * 0.5

    base = max(feats.swatches[:4], key=score).rgb
    dark = min(feats.swatches, key=lambda s: _luma(s.rgb)).rgb
    light = max(feats.swatches, key=lambda s: _luma(s.rgb)).rgb

    shadow = _mix(base, (8, 10, 16), 0.62)
    lit = _mix(base, (255, 255, 255), 0.45)
    if _luma(light) > 0.75:
        lit = _mix(lit, light, 0.35)
    bg_top = _mix(dark, (18, 22, 30), 0.45)
    if _luma(bg_top) > 0.35:
        bg_top = _mix(bg_top, (10, 12, 18), 0.6)
    bg_bottom = _mix(bg_top, (4, 5, 8), 0.68)
    return config.ModelPalette(
        id="image",
        label="取自图片",
        shadow=shadow,
        base=tuple(base),
        light=lit,
        bg_top=bg_top,
        bg_bottom=bg_bottom,
    )


def make_thumbnail(img: Image.Image, size: int = 96) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    return thumb
