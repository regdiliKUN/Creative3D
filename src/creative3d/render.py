"""render.py —— 统一光栅化渲染管线（视口预览与导出预览图共用）。

为什么不用 OpenGL：
    目标机器可能没有可用的显卡驱动、也可能在远程桌面里跑；软件光栅化 + PIL
    在任何 Windows 上都能稳定出图，且「预览」与「导出」走同一条管线，
    不会出现「屏幕上好看、导出变样」的问题。

性能策略：
* 顶点变换 / 背面剔除 / 深度排序全部用 numpy 向量化；
* 绘制用 PIL 的 C 实现多边形填充；
* 交互时降采样 + 限制面数，静止后再出高质量帧（见 ui/viewport.py）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config
from .mesh import Mesh

# 视图朝向（俯仰角限制，避免翻到模型下面）
PITCH_LIMIT = math.radians(86)


@dataclass
class Camera:
    yaw: float = math.radians(-34.0)
    pitch: float = math.radians(22.0)
    zoom: float = 1.0

    def clamped(self) -> "Camera":
        return Camera(
            yaw=self.yaw,
            pitch=max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch)),
            zoom=max(0.55, min(4.0, self.zoom)),
        )


@dataclass
class RenderStats:
    tris: int = 0
    ms: float = 0.0
    width: int = 0
    height: int = 0
    clipped: int = 0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms if self.ms > 0 else 0.0


@dataclass
class RenderResult:
    image: Image.Image
    stats: RenderStats
    extra: dict = field(default_factory=dict)


def _ramp(palette: config.ModelPalette, steps: int = 96) -> np.ndarray:
    shadow = np.array(palette.shadow, dtype=np.float64)
    base = np.array(palette.base, dtype=np.float64)
    light = np.array(palette.light, dtype=np.float64)
    t = np.linspace(0.0, 1.0, steps)
    out = np.empty((steps, 3), dtype=np.float64)
    mid = 0.52
    lo = t < mid
    out[lo] = shadow + (base - shadow)[None, :] * (t[lo] / mid)[:, None]
    out[~lo] = base + (light - base)[None, :] * ((t[~lo] - mid) / (1.0 - mid))[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


_BG_CACHE: dict[tuple, Image.Image] = {}
_BG_CACHE_ORDER: list[tuple] = []
_BG_CACHE_MAX = 12


def _background(size: tuple[int, int], palette: config.ModelPalette) -> Image.Image:
    """竖向渐变 + 中央提亮的背景。

    背景只取决于「尺寸 + 配色」，跟模型无关，而逐像素算一张 1600×1600 的渐变要 70ms
    —— 这在拖拽时是压倒性的开销。做一层小 LRU 缓存后，同一视口的连续帧几乎零成本。
    """
    key = (size[0], size[1], palette.id)
    hit = _BG_CACHE.get(key)
    if hit is not None:
        return hit.copy()

    w, h = size
    top = np.array(palette.bg_top, dtype=np.float64)
    bottom = np.array(palette.bg_bottom, dtype=np.float64)
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    col = top[None, :] * (1 - t) + bottom[None, :] * t
    # 轻微的中央提亮，避免大面积纯色显得死板
    xs = np.linspace(-1.0, 1.0, w)[None, :]
    ys = np.linspace(-1.0, 1.0, h)[:, None]
    radial = np.clip(1.0 - np.sqrt(xs ** 2 + ys ** 2) * 0.75, 0.0, 1.0)[:, :, None]
    col = col + radial * np.array([10.0, 12.0, 16.0])[None, None, :]
    # 底部「打光地面」：在画面下方中央放一团柔和的暖白光，
    # 让物体的落地阴影有可对比的衬底（否则在深底上暗影完全看不出来）。
    # 这团光同时充当「地面」——物体在它上面“坐”着，而不是悬在空中。
    fx = (np.linspace(-1.0, 1.0, w) / 1.05)[None, :]
    fy = ((np.linspace(-1.0, 1.0, h) - 0.40) / 0.70)[:, None]
    floor = np.exp(-(fx ** 2 + fy ** 2) * 1.35)[:, :, None]
    col = col + floor * np.array([42.0, 48.0, 58.0])[None, None, :]
    arr = np.clip(col, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.broadcast_to(arr, (h, w, 3)), "RGB")

    _BG_CACHE[key] = img
    _BG_CACHE_ORDER.append(key)
    while len(_BG_CACHE_ORDER) > _BG_CACHE_MAX:
        _BG_CACHE.pop(_BG_CACHE_ORDER.pop(0), None)
    return img.copy()


class _CameraSpace:
    """一次渲染内的相机参数与投影矩阵（避免重复三角运算）。"""

    def __init__(self, mesh: Mesh, cam: Camera, width: int, height: int) -> None:
        lo, hi = mesh.bounds()
        center = (lo + hi) * 0.5
        radius = float(np.linalg.norm(hi - lo)) * 0.5
        radius = max(radius, 1e-3)
        self.target = center
        self.width = width
        self.height = height
        fov = math.radians(42.0)
        focal = 1.0 / math.tan(fov * 0.5)
        self.focal = focal
        self.dist = max(radius * 2.6, radius * focal / 0.90) / max(cam.zoom, 0.05)

        cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
        cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
        direction = np.array([cp * sy, -cp * cy, sp], dtype=np.float64)
        self.eye = center + direction * self.dist

        forward = -direction
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.basis = np.stack([right, up, forward], axis=0)  # (3,3)

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (points - self.eye) @ self.basis.T

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = self.to_camera(points)
        z = np.maximum(cam[:, 2], 1e-4)
        scale = self.focal * (self.height * 0.5)
        sx = self.width * 0.5 + cam[:, 0] / z * scale
        sy = self.height * 0.5 - cam[:, 1] / z * scale
        return np.stack([sx, sy], axis=1), cam[:, 2]


_LIGHT_A = np.array([-0.45, -0.62, 0.72])
_LIGHT_B = np.array([0.62, 0.34, 0.42])
_LIGHT_A /= np.linalg.norm(_LIGHT_A)
_LIGHT_B /= np.linalg.norm(_LIGHT_B)


def _shade(
    mesh: Mesh, cam: Camera, space: _CameraSpace, palette: config.ModelPalette
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (可见面索引, 面颜色索引, 面深度)，按深度从远到近排序。

    默认使用顶点法线插值（近似 Gouraud），有机形态不会出现横向色带；
    晶体等需要硬朗棱面的形态在 mesh.meta["flat"] 里声明走平面着色。
    """
    fn = mesh.face_normals(normalize=True)
    centroids = mesh.verts[mesh.faces].mean(axis=1)

    view = space.eye - centroids
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    facing = np.einsum("ij,ij->i", fn, view)
    visible = facing > 0.004

    flat = bool(mesh.meta.get("flat"))
    if flat:
        lambert = np.clip(fn @ _LIGHT_A, 0.0, 1.0) * 0.72
        fill = np.clip(fn @ _LIGHT_B, 0.0, 1.0) * 0.30
        rim = np.power(np.clip(1.0 - facing, 0.0, 1.0), 2.4) * 0.30
        intensity = np.clip(0.16 + lambert + fill + rim, 0.0, 1.0)
    else:
        vn = mesh.vertex_normals()
        vview = space.eye - mesh.verts
        vview /= np.maximum(np.linalg.norm(vview, axis=1, keepdims=True), 1e-9)
        v_facing = np.einsum("ij,ij->i", vn, vview)
        v_int = (
            0.16
            + np.clip(vn @ _LIGHT_A, 0.0, 1.0) * 0.72
            + np.clip(vn @ _LIGHT_B, 0.0, 1.0) * 0.30
            + np.power(np.clip(1.0 - np.abs(v_facing), 0.0, 1.0), 2.6) * 0.26
        )
        intensity = np.clip(v_int[mesh.faces].mean(axis=1), 0.0, 1.0)

    idx = np.nonzero(visible)[0]
    if idx.size == 0:
        return idx, idx, idx
    depth = space.to_camera(centroids[idx])[:, 2]
    order = np.argsort(-depth)
    idx = idx[order]
    ramp_idx = np.clip((intensity[idx] * 95.0).astype(np.int32), 0, 95)
    return idx, ramp_idx, depth[order]


def render_scene(
    mesh: Mesh,
    cam: Camera,
    width: int,
    height: int,
    palette: config.ModelPalette,
    *,
    supersample: int = 1,
    background: bool = True,
    shadow: bool = True,
    wire: bool = False,
    transparent: bool = False,
    max_tris: int | None = None,
    bg_image: Image.Image | None = None,
) -> RenderResult:
    """把网格渲染成 PIL 图像。width/height 为逻辑尺寸。"""
    t0 = time.perf_counter()
    width = max(16, int(width))
    height = max(16, int(height))
    ss = max(1, min(3, int(supersample)))
    w, h = width * ss, height * ss

    cam = cam.clamped()
    space = _CameraSpace(mesh, cam, w, h)

    if transparent:
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    elif bg_image is not None:
        base = bg_image.convert("RGB").resize((w, h), Image.LANCZOS)
    elif background:
        base = _background((w, h), palette)
    else:
        base = Image.new("RGB", (w, h), palette.bg_bottom)

    # 注意：ImageDraw.Draw 一旦被强制指定为 "RGBA" 模式，在 RGB 底图上会创建
    # 一个内部副本，导致后续对 base 的 paste 操作“失效”（底座阴影不显示）。
    # 因此这里走底图自己的模式；只有在真正需要半透明描边（wireframe）时才转 RGBA。
    if shadow and not transparent:
        _draw_ground_shadow(base, space, mesh, palette)

    if wire and base.mode != "RGBA":
        base = base.convert("RGBA")

    draw = ImageDraw.Draw(base)
    ramp = _ramp(palette)
    colors = [tuple(int(v) for v in row) for row in ramp]

    idx, ramp_idx, depth = _shade(mesh, cam, space, palette)
    clipped = 0
    if max_tris is not None and idx.size > max_tris:
        clipped = int(idx.size - max_tris)
        idx = idx[:max_tris]
        ramp_idx = ramp_idx[:max_tris]

    # 一次投影所有顶点，然后按面索引取用
    screen, vz = space.project(mesh.verts)
    faces = mesh.faces[idx]
    polys = screen[faces]  # (K, 3, 2)
    behind = (vz[faces] < 0.05).any(axis=1)
    if behind.any():
        polys = polys[~behind]
        ramp_idx = ramp_idx[~behind]
        clipped += int(behind.sum())

    # PIL 需要 python float 元组
    poly_list = polys.tolist()
    for poly, ci in zip(poly_list, ramp_idx.tolist()):
        col = colors[ci]
        draw.polygon([(poly[0][0], poly[0][1]), (poly[1][0], poly[1][1]), (poly[2][0], poly[2][1])],
                     fill=col, outline=col)

    if wire and len(poly_list) <= 4200:
        edge = (min(255, palette.light[0]), min(255, palette.light[1]), min(255, palette.light[2]), 38)
        for poly in poly_list:
            draw.line(
                [
                    (poly[0][0], poly[0][1]),
                    (poly[1][0], poly[1][1]),
                    (poly[2][0], poly[2][1]),
                    (poly[0][0], poly[0][1]),
                ],
                fill=edge,
                width=max(1, ss // 2),
            )

    if ss > 1:
        base = base.resize((width, height), Image.LANCZOS)
    elif base.size != (width, height):
        base = base.resize((width, height), Image.LANCZOS)

    stats = RenderStats(
        tris=int(len(poly_list)),
        ms=(time.perf_counter() - t0) * 1000.0,
        width=width,
        height=height,
        clipped=clipped,
    )
    return RenderResult(base, stats)


def _draw_ground_shadow(img: Image.Image, space: _CameraSpace, mesh: Mesh, palette) -> None:
    """在模型底部画一层柔和的落地阴影（多层椭圆近似高斯模糊，成本可控）。"""
    lo, hi = mesh.bounds()
    pts = np.array(
        [
            [0.0, 0.0, float(lo[2])],
            [float(hi[0] - lo[0]) * 0.45, 0.0, float(lo[2])],
            [0.0, float(hi[1] - lo[1]) * 0.45, float(lo[2])],
        ]
    )
    scr, _ = space.project(pts)
    cx, cy = float(scr[0][0]), float(scr[0][1])
    rx = max(8.0, abs(float(scr[1][0]) - cx))
    ry = max(5.0, abs(float(scr[2][1]) - cy) * 0.6)
    if not all(math.isfinite(v) for v in (cx, cy, rx, ry)):
        return
    # 在 1/3 分辨率的灰度蒙版上画椭圆并模糊，再放大合成。
    # 高斯模糊的代价随像素数线性增长，全尺寸模糊 1600×1600 要 45ms，缩小后只要 5ms。
    ds = 3
    small_w = max(8, img.size[0] // ds)
    small_h = max(8, img.size[1] // ds)
    mask = Image.new("L", (small_w, small_h), 0)
    md = ImageDraw.Draw(mask)
    layers = 6
    # 必须从「最大最淡」画到「最小最亮」：后画的盖在先画的上面，
    # 这样中心叠到最高 alpha、外圈只保留最淡一层，形成柔和渐变。
    # 若顺序反过来，亮心会被外层淡椭圆盖掉，整片只剩 alpha≈2 的微影。
    # 深色背景上纯暗影对比很弱，所以中心峰值给到 ~175，并保证阴影明显比
    # 周围「打光地面」更暗，才能让人看出物体“坐”在地上。
    for i in reversed(range(layers)):
        k = 1.0 + i * 0.30
        alpha = int(175 * (1.0 - i / layers) ** 1.5)
        md.ellipse(
            [
                (cx - rx * k) / ds, (cy - ry * k) / ds,
                (cx + rx * k) / ds, (cy + ry * k) / ds,
            ],
            fill=alpha,
        )
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.2, small_w / 300.0)))
    mask = mask.resize(img.size, Image.BILINEAR)

    # 用蒙版把暗色叠到底图上，避免整幅图来回 convert
    shade = Image.new("RGB", img.size, (0, 0, 0))
    img.paste(shade, (0, 0), mask)


def draw_gizmo(draw: ImageDraw.ImageDraw, space: _CameraSpace, rect: tuple[int, int, int, int], palette) -> None:
    """右下角画一个 XYZ 方位指示器。"""
    x0, y0, x1, y1 = rect
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    radius = min(x1 - x0, y1 - y0) * 0.34
    origin = np.array([[0.0, 0.0, 0.0]])
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    labels = ("X", "Y", "Z")
    colors = ((255, 108, 108, 235), (108, 220, 160, 235), (110, 170, 255, 235))
    cam_pts = space.to_camera(axes)  # 仅取方向
    for i in range(3):
        v = cam_pts[i]
        px = cx + v[0] * radius
        py = cy - v[1] * radius
        draw.line([(cx, cy), (px, py)], fill=colors[i], width=2)
        draw.ellipse([px - 3.5, py - 3.5, px + 3.5, py + 3.5], fill=colors[i])
        draw.text((px + 5, py - 6), labels[i], fill=colors[i])
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(230, 236, 245, 220))
