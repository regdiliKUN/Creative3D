"""generators.py —— 参数化 3D 形态生成器（numpy 向量化）。

统一签名: gen(ctx: GenContext) -> Mesh

约定：
* 局部坐标，最长边随后统一归一化到 100mm（见 Session）；
* 全部形态都保证「封闭 + 法线朝外」，导出前还会再校验一次；
* res 只影响分段数，不影响造型逻辑，因此低精度预览与高精度导出「长得一样」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import noise as N
from .errors import GenerationError
from .mesh import Mesh

TAU = 2.0 * math.pi


@dataclass
class GenContext:
    """生成上下文。image 为 imaging.ImageFeatures（此处按鸭子类型使用）。"""

    seed: int = 0
    res: int = 60
    a: float = 50.0
    b: float = 50.0
    image: object | None = None
    extra: dict = field(default_factory=dict)

    def norm(self, value: float) -> float:
        return max(0.0, min(100.0, float(value))) / 100.0


# --------------------------------------------------------------------------
# 基础构件
# --------------------------------------------------------------------------


def _loft(rings: np.ndarray, wrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """把 (K, M, 3) 的环序列缝合成管状表面，环内首尾相连。

    wrap=True 时把最后一环与第一环缝合（用于闭合曲线，如环面纽结），
    不会产生重复顶点，因此网格保持封闭。
    """
    k, m = rings.shape[0], rings.shape[1]
    verts = rings.reshape(-1, 3).astype(np.float32)
    idx = np.arange(k * m).reshape(k, m)
    if wrap:
        a = idx
        b = np.roll(idx, -1, axis=0)
    else:
        a = idx[:-1]
        b = idx[1:]
    a2 = np.roll(a, -1, axis=1)
    b2 = np.roll(b, -1, axis=1)
    f1 = np.stack([a, a2, b2], axis=-1).reshape(-1, 3)
    f2 = np.stack([a, b2, b], axis=-1).reshape(-1, 3)
    return verts, np.concatenate([f1, f2], axis=0).astype(np.int32)


def _cap_faces(ring_indices: np.ndarray, center_index: int, up: bool) -> np.ndarray:
    """用扇形封口。up=True 时法线朝 +z。"""
    j = np.arange(len(ring_indices))
    jn = (j + 1) % len(ring_indices)
    if up:
        tri = np.stack([np.full_like(j, center_index), ring_indices[j], ring_indices[jn]], axis=1)
    else:
        tri = np.stack([np.full_like(j, center_index), ring_indices[jn], ring_indices[j]], axis=1)
    return tri.astype(np.int32)


def _hollow_revolve(profile_r: np.ndarray, profile_z: np.ndarray, seg: int, wall: float) -> Mesh:
    """把一条外轮廓旋转成「有壁厚的中空实体」（花瓶 / 海螺类）。

    profile_r / profile_z 长度 K，从底到顶；wall 为局部单位下的壁厚。
    """
    k = len(profile_r)
    theta = np.linspace(0.0, TAU, seg, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def ring(radius: np.ndarray, z: np.ndarray, twist: np.ndarray | None = None) -> np.ndarray:
        ang = theta[None, :] + (twist[:, None] if twist is not None else 0.0)
        r = radius[:, None]
        return np.stack([r * np.cos(ang), r * np.sin(ang), np.repeat(z[:, None], seg, axis=1)], axis=-1)

    outer = ring(profile_r, profile_z)
    r_in = np.maximum(profile_r - wall, profile_r * 0.35)
    z_in = np.minimum(profile_z + wall * 1.8, profile_z[-1])
    z_in = np.maximum(z_in, profile_z[0] + wall * 1.2)
    inner = ring(r_in, z_in)

    rings = np.concatenate([outer, inner[::-1]], axis=0)
    verts, faces = _loft(rings)
    faces = [faces]

    outer_ring_idx = np.arange(seg)
    inner_ring_idx = (2 * k - 1) * seg + np.arange(seg)
    c_bottom = len(verts)
    c_floor = len(verts) + 1
    verts = np.concatenate(
        [verts, np.array([[0.0, 0.0, profile_z[0]], [0.0, 0.0, z_in[0]]], dtype=np.float32)], axis=0
    )
    faces.append(_cap_faces(outer_ring_idx, c_bottom, up=False))
    faces.append(_cap_faces(inner_ring_idx, c_floor, up=True))
    return Mesh(verts, np.concatenate(faces, axis=0))


def _tube(
    curve: np.ndarray,
    radii: np.ndarray,
    seg: int,
    closed: bool,
    caps: bool,
    elliptic: tuple[float, float] = (1.0, 1.0),
) -> Mesh:
    """沿曲线扫掠圆截面（平行移动标架，避免扭转跳变）。"""
    k = len(curve)
    tangents = np.empty_like(curve)
    if closed:
        tangents = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    else:
        tangents[1:-1] = curve[2:] - curve[:-2]
        tangents[0] = curve[1] - curve[0]
        tangents[-1] = curve[-1] - curve[-2]
    norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norm, 1e-9)

    # 初始法向
    up = np.array([0.0, 0.0, 1.0])
    n0 = np.cross(tangents[0], up)
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(tangents[0], np.array([1.0, 0.0, 0.0]))
    n0 /= np.linalg.norm(n0)

    normals = np.empty_like(curve)
    normals[0] = n0
    for i in range(1, k):
        prev_t = tangents[i - 1]
        t = tangents[i]
        v = np.cross(prev_t, t)
        s = np.linalg.norm(v)
        if s < 1e-8:
            normals[i] = normals[i - 1]
        else:
            axis = v / s
            angle = math.atan2(s, float(np.dot(prev_t, t)))
            c, sn = math.cos(angle), math.sin(angle)
            nprev = normals[i - 1]
            normals[i] = nprev * c + np.cross(axis, nprev) * sn + axis * np.dot(axis, nprev) * (1 - c)
            normals[i] /= max(np.linalg.norm(normals[i]), 1e-9)

    theta = np.linspace(0.0, TAU, seg, endpoint=False)
    binormals = np.cross(tangents, normals)

    # 闭合曲线：平行移动会累积一个小的「和乐角」，把它均摊到整圈，
    # 否则首尾接缝会出现一条突然扭转的窄带。
    shift = np.zeros(k)
    if closed and k > 2:
        prev_t, t = tangents[-1], tangents[0]
        v = np.cross(prev_t, t)
        s = float(np.linalg.norm(v))
        if s > 1e-9:
            axis = v / s
            ang = math.atan2(s, float(np.dot(prev_t, t)))
            c, sn = math.cos(ang), math.sin(ang)
            n_prev = normals[-1]
            n_end = n_prev * c + np.cross(axis, n_prev) * sn + axis * np.dot(axis, n_prev) * (1 - c)
            denom = max(float(np.linalg.norm(n_end) * np.linalg.norm(normals[0])), 1e-9)
            delta = math.atan2(float(np.dot(np.cross(n_end, normals[0]), t)),
                               float(np.dot(n_end, normals[0])) / denom)
            shift = -delta * np.arange(k) / float(k)

    ang = theta[None, :] + shift[:, None]
    cos_a = np.cos(ang)[:, :, None]
    sin_a = np.sin(ang)[:, :, None]
    rx, rz = elliptic
    rings = curve[:, None, :] + radii[:, None, None] * (
        (cos_a * rx) * normals[:, None, :] + (sin_a * rz) * binormals[:, None, :]
    )
    verts, faces = _loft(rings, wrap=closed)
    faces = [faces]
    if caps and not closed:
        c0 = len(verts)
        c1 = len(verts) + 1
        verts = np.concatenate(
            [verts, curve[0][None, :].astype(np.float32), curve[-1][None, :].astype(np.float32)], axis=0
        )
        faces.append(_cap_faces(np.arange(seg), c0, up=False))
        last = (len(rings) - 1) * seg + np.arange(seg)
        faces.append(_cap_faces(last, c1, up=True))
    return Mesh(verts, np.concatenate(faces, axis=0))


def _solid_from_grid(xs: np.ndarray, ys: np.ndarray, ztop: np.ndarray, zbot: float) -> Mesh:
    """把高度场做成封闭地块：顶面 + 底面 + 四周侧壁。"""
    gx, gy = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([xx, yy, ztop], axis=-1).reshape(-1, 3)
    bottom = np.stack([xx, yy, np.full_like(ztop, zbot)], axis=-1).reshape(-1, 3)
    verts = np.concatenate([top, bottom], axis=0).astype(np.float32)
    n = gx * gy
    idx = np.arange(n).reshape(gx, gy)

    i = idx[:-1, :-1]
    j = idx[1:, :-1]
    k = idx[1:, 1:]
    l = idx[:-1, 1:]
    top_faces = [np.stack([i, j, k], -1).reshape(-1, 3), np.stack([i, k, l], -1).reshape(-1, 3)]
    b = idx + n
    bi, bj, bk, bl = b[:-1, :-1], b[1:, :-1], b[1:, 1:], b[:-1, 1:]
    bottom_faces = [np.stack([bi, bk, bj], -1).reshape(-1, 3), np.stack([bi, bl, bk], -1).reshape(-1, 3)]

    # 侧壁：沿周界做一圈 loft（底 → 顶），顶点复用顶/底网格，保证完全封闭
    perimeter = np.concatenate(
        [
            idx[:, 0],
            idx[-1, 1:],
            idx[::-1, -1][1:],
            idx[0, ::-1][1:-1],
        ]
    )
    peri_bottom = perimeter + n
    rings = np.stack([bottom[perimeter], top[perimeter]], axis=0)
    _, sf = _loft(rings)
    side_faces = np.empty_like(sf)
    p_count = len(perimeter)
    for face_no in range(sf.shape[0]):
        for k_no in range(3):
            vi = int(sf[face_no, k_no])
            side_faces[face_no, k_no] = peri_bottom[vi] if vi < p_count else perimeter[vi - p_count]

    faces = np.concatenate(
        [np.concatenate(top_faces), np.concatenate(bottom_faces), side_faces],
        axis=0,
    )
    return Mesh(verts, faces)


def _finalize(mesh: Mesh) -> Mesh:
    """统一收尾：保证法线整体朝外（有向体积为正）。"""
    if mesh.is_empty():
        return mesh
    if mesh.signed_volume() < 0:
        mesh.faces = mesh.faces[:, ::-1]
    return mesh


def _icosphere(levels: int) -> Mesh:
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )
    for _ in range(max(0, levels)):
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        edges = np.concatenate(
            [np.stack([v0, v1], 1), np.stack([v1, v2], 1), np.stack([v2, v0], 1)], axis=0
        )
        und = np.sort(edges, axis=1)
        uniq, inverse = np.unique(und, axis=0, return_inverse=True)
        mids = (verts[uniq[:, 0]] + verts[uniq[:, 1]]) * 0.5
        mids /= np.linalg.norm(mids, axis=1, keepdims=True)
        mid_index = len(verts) + inverse.reshape(3, -1).T
        verts = np.concatenate([verts, mids], axis=0)
        m01, m12, m20 = mid_index[:, 0], mid_index[:, 1], mid_index[:, 2]
        faces = np.concatenate(
            [
                np.stack([v0, m01, m20], 1),
                np.stack([v1, m12, m01], 1),
                np.stack([v2, m20, m12], 1),
                np.stack([m01, m12, m20], 1),
            ],
            axis=0,
        )
    return Mesh(verts.astype(np.float32), faces.astype(np.int32))


# --------------------------------------------------------------------------
# 1. 波纹花瓶
# --------------------------------------------------------------------------


def gen_vase(ctx: GenContext) -> Mesh:
    k = 16 + int(ctx.res * 0.32)
    seg = 28 + int(ctx.res * 0.60)
    t = np.linspace(0.0, 1.0, k)
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)

    profile = 0.17 + 0.30 * np.exp(-(((t - 0.33) / 0.30) ** 2)) + 0.075 * t
    profile *= 1.0 + 0.05 * (N.fbm1(t * 2.4 + 5.0, seed=ctx.seed) - 0.5)

    rows = 3.0 + round(b * 9.0)
    amp = 0.010 + b * 0.055
    theta = np.linspace(0.0, TAU, seg, endpoint=False)
    th = theta[None, :]
    tt = t[:, None]
    ripple = 1.0 + amp * np.cos(rows * th + 5.0 * tt + ctx.seed * 0.013)
    ripple *= 1.0 + 0.018 * np.sin(9.0 * tt + ctx.seed * 0.007)
    radius = profile[:, None] * ripple
    twist = a * 0.75 * TAU * t
    ang = th + twist[:, None]
    z = tt * np.ones_like(th)

    wall = 0.022
    r_in = np.maximum(radius - wall, radius * 0.4)
    z_in = np.minimum(z + wall * 1.6, z[-1])
    z_in = np.maximum(z_in, z[0] + wall * 1.4)

    outer = np.stack([radius * np.cos(ang), radius * np.sin(ang), z], axis=-1)
    inner = np.stack([r_in * np.cos(ang), r_in * np.sin(ang), z_in], axis=-1)
    rings = np.concatenate([outer, inner[::-1]], axis=0)

    verts, faces = _loft(rings)
    faces = [faces]
    c_bottom = len(verts)
    c_floor = len(verts) + 1
    verts = np.concatenate(
        [
            verts,
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(z_in.flat[0])]], dtype=np.float32),
        ],
        axis=0,
    )
    faces.append(_cap_faces(np.arange(seg), c_bottom, up=False))
    faces.append(_cap_faces((2 * k - 1) * seg + np.arange(seg), c_floor, up=True))
    return Mesh(verts, np.concatenate(faces, axis=0))


# --------------------------------------------------------------------------
# 2. 晶簇尖晶
# --------------------------------------------------------------------------


def _prism(radius: float, shaft: float, sides: int, tip: float, phase: float = 0.0) -> Mesh:
    """一根封闭的棱柱 + 锥形尖端（石英式晶柱）。"""
    theta = np.linspace(0.0, TAU, sides, endpoint=False) + phase
    base = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(sides)], axis=-1)
    top = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full(sides, shaft)], axis=-1
    )
    verts = np.concatenate([base, top, np.array([[0.0, 0.0, shaft + tip]])], axis=0)
    ring_b = np.arange(sides)
    ring_t = np.arange(sides) + sides
    apex = 2 * sides
    # 底面用中心点封口
    verts = np.concatenate([verts, np.array([[0.0, 0.0, 0.0]])], axis=0)
    centre = len(verts) - 1
    faces = [_cap_faces(ring_b, centre, up=False)]
    # 侧面
    j = np.arange(sides)
    jn = (j + 1) % sides
    quads1 = np.stack([ring_b[j], ring_b[jn], ring_t[jn]], axis=1)
    quads2 = np.stack([ring_b[j], ring_t[jn], ring_t[j]], axis=1)
    faces += [quads1, quads2]
    # 尖端
    tips = np.stack([ring_t[j], ring_t[jn], np.full(sides, apex)], axis=1)
    faces.append(tips)
    return Mesh(verts.astype(np.float32), np.concatenate(faces).astype(np.int32))


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float64,
    )


def gen_crystal(ctx: GenContext) -> Mesh:
    """晶簇：一丛朝向各异、长短不一的六棱晶柱，底部融在同一块基座上。

    构型比「球面噪声位移」更接近真实的晶簇，也天然是封闭网格
    （每根晶柱各自封闭，整体可被切片软件正确识别）。
    """
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    rng = np.random.default_rng((int(ctx.seed) * 2654435761) & 0xFFFFFFFF)
    # 数量随「生长强度」和细节密度一起增长：否则密度滑杆对晶体完全无效
    count = 4 + int(round(a * 9 + ctx.res * 0.07))
    sides = 4 + int(round(b * 4))  # 4..8 棱

    meshes: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for i in range(count):
        golden = 2.399963229728653
        ang = i * golden + rng.random() * 0.6
        radial = 0.34 * math.sqrt((i + 0.6) / count) * (0.55 + rng.random() * 0.8)
        is_center = i == 0
        height = (1.05 if is_center else 0.55 + rng.random() * 0.75) * (0.65 + a * 0.75)
        radius = (0.16 if is_center else 0.09 + rng.random() * 0.075) * (0.8 + rng.random() * 0.5)
        tip = height * (0.30 - b * 0.16) * (0.8 + rng.random() * 0.5)
        shaft = max(height - tip, height * 0.35)

        prism = _prism(radius, shaft, sides, tip, phase=rng.random() * TAU)
        # 越靠外倾斜越大，形成放射状晶簇
        tilt = 0.0 if is_center else math.radians(10 + rng.random() * 32) * (0.6 + a * 0.7)
        if tilt > 0:
            radial_dir = np.array([math.cos(ang), math.sin(ang), 0.0])
            axis = np.cross(np.array([0.0, 0.0, 1.0]), radial_dir)
            rot = _rot_axis(axis, tilt)
            prism.verts = (prism.verts.astype(np.float64) @ rot.T).astype(np.float32)

        pos = np.array([radial * math.cos(ang), radial * math.sin(ang), -0.04 - rng.random() * 0.05])
        verts = prism.verts + pos
        meshes.append(verts)
        faces.append(prism.faces + offset)
        offset += len(verts)

    all_verts = np.concatenate(meshes, axis=0)
    all_faces = np.concatenate(faces, axis=0)
    mesh = Mesh(all_verts, all_faces)
    mesh.meta["flat"] = True  # 晶体保留硬朗的平面着色
    return mesh


# --------------------------------------------------------------------------
# 3. 环面纽结
# --------------------------------------------------------------------------


def gen_knot(ctx: GenContext) -> Mesh:
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    p = 2 + int(round(a * 5))
    q = 3 + int(round(b * 6))
    while math.gcd(p, q) != 1:
        q += 1
    steps = 70 + int(ctx.res * 1.2)
    seg = 10 + int(ctx.res * 0.16)

    t = np.linspace(0.0, TAU, steps, endpoint=False)
    r_major = 2.0 + np.cos(q * t)
    curve = np.stack(
        [r_major * np.cos(p * t), r_major * np.sin(p * t), np.sin(q * t) * 0.9], axis=-1
    )
    curve *= 0.34
    mod = 1.0 + 0.22 * (N.fbm1(t / TAU * 3.0, seed=ctx.seed) - 0.5)
    radii = (0.052 + 0.018 * a) * mod
    return _tube(curve, radii, seg, closed=True, caps=False)


# --------------------------------------------------------------------------
# 4. 螺旋塔
# --------------------------------------------------------------------------


def gen_tower(ctx: GenContext) -> Mesh:
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    sides = 3 + int(round(b * 7))  # 3..10
    k = 14 + int(ctx.res * 0.30)
    t = np.linspace(0.0, 1.0, k)
    radius = (0.34 * (1.0 - 0.42 * t)) * (1.0 + 0.05 * (N.fbm1(t * 3.2 + 7.0, seed=ctx.seed) - 0.5))
    radius *= 1.0 + 0.10 * np.sin(t * math.pi * (2 + round(a * 4)))
    theta = np.linspace(0.0, TAU, sides, endpoint=False)
    twist = a * TAU * 0.9 * t
    ang = theta[None, :] + twist[:, None]
    rings = np.stack(
        [
            radius[:, None] * np.cos(ang),
            radius[:, None] * np.sin(ang),
            np.repeat((t * 1.25)[:, None], sides, axis=1),
        ],
        axis=-1,
    )
    verts, faces = _loft(rings)
    faces = [faces]
    c0, c1 = len(verts), len(verts) + 1
    verts = np.concatenate(
        [verts, np.array([[0, 0, 0.0], [0, 0, float(t[-1] * 1.25)]], dtype=np.float32)], axis=0
    )
    faces.append(_cap_faces(np.arange(sides), c0, up=False))
    faces.append(_cap_faces((k - 1) * sides + np.arange(sides), c1, up=True))
    return Mesh(verts, np.concatenate(faces, axis=0))


# --------------------------------------------------------------------------
# 5. 起伏地块
# --------------------------------------------------------------------------


def gen_terrain(ctx: GenContext) -> Mesh:
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    g = 26 + int(ctx.res * 0.45)
    xs = np.linspace(-0.55, 0.55, g)
    ys = np.linspace(-0.55, 0.55, g)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    freq = 1.6 + b * 4.2
    h = N.fbm(np.stack([xx * freq, yy * freq, np.full_like(xx, ctx.seed * 0.01)], -1),
              seed=ctx.seed, octaves=5).reshape(g, g)
    h = (h - h.min()) / max(float(h.max() - h.min()), 1e-9)
    h = h ** 1.15
    amp = 0.10 + a * 0.55
    ztop = h * amp
    return _solid_from_grid(xs, ys, ztop, zbot=-0.06)


# --------------------------------------------------------------------------
# 6. 螺旋海螺
# --------------------------------------------------------------------------


def gen_shell(ctx: GenContext) -> Mesh:
    """海螺：对数螺线 + 逐圈堆叠的圆管 + 末圈外扩的壳口。

    z 方向按「当前管径」累计抬升，使相邻两圈恰好相切，
    既不会互相穿模（保证可切片），又能看出层层叠压的螺纹。
    """
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    turns = 3.0 + b * 3.0
    steps = 110 + int(ctx.res * 1.4)
    seg = 12 + int(ctx.res * 0.16)
    u = np.linspace(0.0, 1.0, steps)
    growth = 0.90 + a * 0.95
    theta = turns * TAU * u
    r = 0.060 * np.exp(growth * u)
    tube = 0.70 * (growth / turns) * r
    tube = tube * (1.0 + 0.32 * np.clip((u - 0.80) / 0.20, 0.0, 1.0) ** 2)
    dz = tube * 1.85 * (turns / steps)
    z = np.cumsum(dz)
    curve = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)
    mod = 1.0 + 0.05 * (N.fbm1(u * 4.0, seed=ctx.seed) - 0.5)
    return _tube(curve, tube * mod, seg, closed=False, caps=True)


# --------------------------------------------------------------------------
# 7. 图像浮雕（多模态）
# --------------------------------------------------------------------------


def gen_relief(ctx: GenContext) -> Mesh:
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    g = 32 + int(ctx.res * 0.50)
    xs = np.linspace(-0.5, 0.5, g)
    ys = np.linspace(-0.5, 0.5, g)
    if ctx.image is not None:
        h = ctx.image.heightmap(g)
    else:
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        h = N.fbm(
            np.stack([xx * 4.0, yy * 4.0, np.full_like(xx, 1.7)], -1), seed=ctx.seed, octaves=4
        ).reshape(g, g)
    h = np.clip(h, 0.0, 1.0)
    depth = 0.08 + a * 0.50
    gamma = 0.65 + b * 1.4
    ztop = (h ** gamma) * depth
    return _solid_from_grid(xs, ys, ztop, zbot=-0.05)


# --------------------------------------------------------------------------
# 8. 轮廓旋转体（多模态）
# --------------------------------------------------------------------------


def gen_profile(ctx: GenContext) -> Mesh:
    a, b = ctx.norm(ctx.a), ctx.norm(ctx.b)
    k = 22 + int(ctx.res * 0.35)
    seg = 28 + int(ctx.res * 0.55)
    t = np.linspace(0.0, 1.0, k)
    if ctx.image is not None:
        prof = ctx.image.radial_profile(k)
    else:
        prof = 0.20 + 0.26 * np.sin(np.pi * t) ** 0.9
        # 没有参考图时也要响应种子：否则换种子看不出任何变化
        prof = prof + 0.07 * (N.fbm1(t * 3.2, seed=ctx.seed, octaves=3) - 0.5)
    prof = 0.06 + 0.94 * np.clip(prof, 0.0, 1.0)
    prof *= 0.34 * (0.65 + a * 0.7)
    smooth = 0.12 + b * 0.75
    if smooth > 0.01:
        kernel = np.ones(5) / 5.0
        prof = np.convolve(np.pad(prof, 2, mode="edge"), kernel, mode="valid")
    prof = np.maximum(prof, 0.02)
    z = t * 1.0
    return _hollow_revolve(prof, z, seg, wall=0.024)


GENERATORS = {
    "vase": gen_vase,
    "crystal": gen_crystal,
    "knot": gen_knot,
    "tower": gen_tower,
    "terrain": gen_terrain,
    "shell": gen_shell,
    "relief": gen_relief,
    "profile": gen_profile,
}


def build(form_id: str, ctx: GenContext) -> Mesh:
    """生成形态。任何失败都抛出 GenerationError，方便上层统一翻译成人话。"""
    fn = GENERATORS.get(form_id)
    if fn is None:
        raise GenerationError(
            f"没有「{form_id}」这个形态。",
            hint="在左侧形态列表里重新选一个。",
            detail=f"unknown form id: {form_id}",
        )
    try:
        mesh = fn(ctx)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(
            "生成这个形态时出错了。",
            hint="试着换一个形态，或降低细节密度。",
            detail=f"{form_id}: {type(exc).__name__}: {exc}",
        ) from exc
    if mesh.is_empty():
        raise GenerationError(
            "这组参数没有生成出任何实体。",
            hint="试着提高细节密度，或把两个参数往中间调。",
            detail=f"{form_id} produced empty mesh at res={ctx.res}",
        )
    if not np.isfinite(mesh.verts).all():
        raise GenerationError(
            "这组参数算出了无效坐标。",
            hint="把细节密度或参数往回调一些再试。",
            detail=f"{form_id} produced NaN/Inf vertices at res={ctx.res}",
        )
    _finalize(mesh)
    mesh.meta["form"] = form_id
    mesh.meta["seed"] = ctx.seed
    mesh.meta["res"] = ctx.res
    mesh.meta["image"] = bool(ctx.image is not None)
    return mesh
