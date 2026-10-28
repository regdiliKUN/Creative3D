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


def _background(size: tuple[int, int], palette: config.ModelPalette) -> Image.Image:
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
    arr = np.clip(col, 0, 255).astype(np.uint8)
    return Image.fromarray(np.broadcast_to(arr, (h, w, 3)), "RGB")


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
        self.dist = max(radius * 2.6, radius * focal / 0.80) / max(cam.zoom, 0.05)

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

    draw = ImageDraw.Draw(base, "RGBA")
    ramp = _ramp(palette)
    colors = [tuple(int(v) for v in row) for row in ramp]

    idx, ramp_idx, depth = _shade(mesh, cam, space, palette)
    clipped = 0
    if max_tris is not None and idx.size > max_tris:
        clipped = int(idx.size - max_tris)
        idx = idx[:max_tris]
        ramp_idx = ramp_idx[:max_tris]

    if shadow and not transparent:
        _draw_ground_shadow(base, space, mesh, palette)

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
    base_center = np.array([[0.0, 0.0, float(lo[2])]])
    pts = np.array(
        [
            [0.0, 0.0, float(lo[2])],
            [float(hi[0] - lo[0]) * 0.45, 0.0, float(lo[2])],
            [0.0, float(hi[1] - lo[1]) * 0.45, float(lo[2])],
        ]
    )
    scr, _ = space.project(pts)
    cx, cy = float(scr[0][0]), float(scr[0][1])
    rx = max(6.0, abs(float(scr[1][0]) - cx))
    ry = max(4.0, abs(float(scr[2][1]) - cy) * 0.55)
    if not all(math.isfinite(v) for v in (cx, cy, rx, ry)):
        return
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    layers = 7
    for i in range(layers):
        k = 1.0 + i * 0.55
        alpha = int(46 * (1.0 - i / layers) ** 1.6)
        od.ellipse([cx - rx * k, cy - ry * k, cx + rx * k, cy + ry * k], fill=(0, 0, 0, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(max(1.0, img.size[0] / 340.0)))
    rgba = img.convert("RGBA")
    rgba.alpha_composite(overlay)
    img.paste(rgba.convert("RGB"), (0, 0))


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
