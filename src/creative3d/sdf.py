"""sdf.py —— 有符号距离场（SDF）建模内核。

为什么换成 SDF：
    v2 的 8 个形态各自手写三角网格，能做的形状被写死在代码里；
    SDF 把「形状」表达成函数 d(p)（点到表面的带符号距离），
    于是任意图元都能用布尔 / 平滑并集 / 噪声位移自由拼装，
    再由统一的网格化器（sdfmesh.py）转成三角网格 —— 只要场是良态的，
    出来的网格天然封闭、法线朝外，不再需要每个形态自己保证水密。

约定：
    * 所有函数接收 (N, 3) 的 float64 点集，返回 (N,) 的距离；
      负数在内部，正数在外部，单位是「局部单位」（模型大致落在 ±1 内）。
    * 图元默认以原点为中心、沿 z 轴摆放；用 at()/rotz() 等算子摆位。
    * 组合算子是「接收函数、返回函数」的高阶函数，因此可以任意嵌套。
    * 距离场只保证近表面处近似正确（这是 SDF 的通用做法），
      对网格化足够，不要拿它做精确碰撞。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

from . import noise as _noise

SDF = Callable[[np.ndarray], np.ndarray]

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _p(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 3)
    return p


def _len(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _v2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x, y], axis=-1)


def _len2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y)


def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """多项式平滑最小值（k 越大过渡越圆润）。"""
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -_smin(-a, -b, k)


def _mix(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    return a + (b - a) * t


# --------------------------------------------------------------------------
# 3D 图元（以原点为中心）
# --------------------------------------------------------------------------


def sphere(r: float = 1.0) -> SDF:
    return lambda p: _len(_p(p)) - r


def ellipsoid(rx: float, ry: float, rz: float) -> SDF:
    rad = np.array([rx, ry, rz], dtype=np.float64)

    def f(p):
        q = _p(p) / rad
        k0 = _len(q)
        k1 = _len(q / rad)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    return f


def box(hx: float, hy: float, hz: float) -> SDF:
    b = np.array([hx, hy, hz], dtype=np.float64)

    def f(p):
        q = np.abs(_p(p)) - b
        outside = _len(np.maximum(q, 0.0))
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    return f


def rbox(hx: float, hy: float, hz: float, r: float = 0.05) -> SDF:
    r = float(max(r, 0.0))
    inner = box(max(hx - r, 1e-4), max(hy - r, 1e-4), max(hz - r, 1e-4))
    return lambda p: inner(p) - r


def cyl(r: float, h: float) -> SDF:
    """沿 z 轴的圆柱（含上下平盖）。"""

    def f(p):
        q = _p(p)
        dx = _len2(q[:, 0], q[:, 1]) - r
        dz = np.abs(q[:, 2]) - h * 0.5
        outside = _len2(np.maximum(dx, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dx, dz), 0.0)
        return outside + inside

    return f


def cone(r1: float, r2: float, h: float) -> SDF:
    """沿 z 轴的圆台：底半径 r1（z=-h/2）、顶半径 r2（z=+h/2）。"""
    half = max(h * 0.5, 1e-6)
    k2x = r2 - r1
    k2y = 2.0 * half
    k2_sq = max(k2x * k2x + k2y * k2y, 1e-12)

    def f(p):
        q = _p(p)
        x = _len2(q[:, 0], q[:, 1])
        y = q[:, 2]
        # iq 的 capped cone：取「侧面垂足」与「上下盖」两条候选距离中更近者
        cax = x - np.minimum(x, np.where(y < 0.0, r1, r2))
        cay = np.abs(y) - half
        t = np.clip(((r2 - x) * r2 + (half - y) * half) / k2_sq, 0.0, 1.0)
        cbx = x - r2 + k2x * t
        cby = y - half + k2y * t
        s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
        return s * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))

    return f


def lathe(profile: Sequence[Sequence[float]], blend: float = 0.0) -> SDF:
    """按 (z, 半径) 剖面生成旋转体（花瓶、棋子、陀螺、瓶子）。

    实现的是**精确**距离场：把剖面看成 (半径, 高度) 平面上的一条折线，
    对每个点求到折线的最短距离，再用「半径是否小于该高度的剖面半径」定内外。
    这比「一堆圆台求并集」贵一点，但内外都准确 —— 于是抽壳、布尔差、
    平滑这些操作不会在实体内部冒出假表面。
    """
    raw = sorted(((float(z), max(float(r), 1e-4)) for z, r in profile), key=lambda p: p[0])
    if len(raw) < 2:
        raise ValueError("剖面至少需要两个点")
    zs = np.array([p[0] for p in raw], dtype=np.float64)
    rs = np.array([p[1] for p in raw], dtype=np.float64)
    pts = np.stack([rs, zs], axis=1)  # (半径, 高度)
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    # 两端封盖（从轴心到边缘），让「到边界的最短距离」在底面/顶面也正确
    segs.append((np.array([0.0, zs[0]]), pts[0]))
    segs.append((pts[-1], np.array([0.0, zs[-1]])))
    segs = [(np.asarray(a), np.asarray(b)) for a, b in segs]
    z_lo, z_hi = float(zs[0]), float(zs[-1])

    def f(p):
        q = _p(p)
        rad = _len2(q[:, 0], q[:, 1])
        z = q[:, 2]
        best = np.full(q.shape[0], 1e18)
        for a, b in segs:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            l2 = dx * dx + dy * dy
            t = np.clip(((rad - a[0]) * dx + (z - a[1]) * dy) / max(l2, 1e-12), 0.0, 1.0)
            ex = rad - (a[0] + t * dx)
            ey = z - (a[1] + t * dy)
            best = np.minimum(best, ex * ex + ey * ey)
        d = np.sqrt(best)
        limit = np.interp(z, zs, rs)
        inside = (z >= z_lo) & (z <= z_hi) & (rad <= limit)
        return np.where(inside, -d, d)

    return f


def capsule(a: Sequence[float], b: Sequence[float], r: float) -> SDF:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    ab2 = float(ab @ ab)

    def f(p):
        q = _p(p) - a
        t = np.clip((q @ ab) / max(ab2, 1e-12), 0.0, 1.0)
        return _len(q - t[:, None] * ab) - r

    return f


def torus(R: float, r: float) -> SDF:
    def f(p):
        q = _p(p)
        dx = _len2(q[:, 0], q[:, 1]) - R
        return _len2(dx, q[:, 2]) - r

    return f


def hexprism(r: float, h: float, sides: int = 6, round_r: float = 0.0) -> SDF:
    """正 n 棱柱（沿 z 轴）。sides=6 即六棱柱，4 是方柱，3 是三棱柱。"""
    n = max(3, int(sides))
    # 用「n 个半空间求交」的近似写法，几何直观且对任意边数都稳定
    ang = np.linspace(0.0, TAU, n, endpoint=False) + math.pi / n
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    apothem = r * math.cos(math.pi / n)

    def f(p):
        q = _p(p)
        d = np.full(q.shape[0], -1e9)
        for nx_, ny_ in normals:
            d = np.maximum(d, q[:, 0] * nx_ + q[:, 1] * ny_ - apothem)
        dz = np.abs(q[:, 2]) - h * 0.5
        outside = _len2(np.maximum(d, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(d, dz), 0.0)
        return outside + inside - round_r

    return f


def plane(normal: Sequence[float], offset: float = 0.0) -> SDF:
    """半空间：dot(p, n) - offset < 0 的一侧为实体。"""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    return lambda p: _p(p) @ n - offset


def link(length: float, r: float, tube: float) -> SDF:
    """链环（两个圆环拼成的长圆孔环，打印链条用）。"""
    half = max(length * 0.5 - r, 1e-4)
    seg = capsule([-half, 0, 0], [half, 0, 0], r)

    def f(p):
        q = _p(p)
        d = seg(q)
        return np.abs(d) - tube

    return f


# --------------------------------------------------------------------------
# 2D 图元（在 xy 平面上，返回 2D 距离；用于挤出 / 旋转）
# --------------------------------------------------------------------------


def circle2(r: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda q: _len2(_p2(q)[:, 0], _p2(q)[:, 1]) - r


def _p2(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(-1, 2)
    return q


def box2(hx: float, hy: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(q):
        v = np.abs(_p2(q)) - np.array([hx, hy])
        outside = _len2(np.maximum(v[:, 0], 0.0), np.maximum(v[:, 1], 0.0))
        inside = np.minimum(np.max(v, axis=1), 0.0)
        return outside + inside

    return f


def poly2(points: Iterable[Sequence[float]]) -> Callable[[np.ndarray], np.ndarray]:
    """任意简单多边形的精确 2D 距离（外部为正）。"""
    pts = np.asarray(list(points), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError("多边形至少需要 3 个点")
    a = pts
    b = np.roll(pts, -1, axis=0)
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)

    def f(q):
        v = _p2(q)
        n = v.shape[0]
        best = np.full(n, 1e9)
        inside = np.zeros(n, dtype=bool)
        px, py = v[:, 0], v[:, 1]
        for i in range(a.shape[0]):
            ax, ay = a[i]
            dx, dy = ab[i]
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / ab2[i], 0.0, 1.0)
            ex = px - (ax + t * dx)
            ey = py - (ay + t * dy)
            best = np.minimum(best, ex * ex + ey * ey)
            # 射线法：向 +x 方向发一条水平射线
            cond = (ay > py) != (b[i][1] > py)
            xint = ax + (py - ay) * dx / (dy if abs(dy) > 1e-12 else 1e-12)
            inside ^= cond & (px < xint)
        d = np.sqrt(best)
        return np.where(inside, -d, d)

    return f


def star2(r_out: float, r_in: float, points: int = 5, phase: float = 0.0) -> Callable:
    n = max(3, int(points))
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = phase + i * math.pi / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return poly2(pts)


def heart2(size: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """心形（参数曲线采样成多边形，比隐函数更可控）。"""
    pts = []
    for i in range(48):
        t = TAU * i / 48
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2 * t) - 2.0 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x / 17.0 * size, y / 17.0 * size))
    return poly2(pts)


def polygon_from_profile(points: Iterable[Sequence[float]]) -> Callable:
    """旋转体用的剖面多边形（x = 半径，y = 高度）。"""
    return poly2(points)


def at2(fn: Callable, dx: float, dy: float) -> Callable:
    off = np.array([dx, dy], dtype=np.float64)
    return lambda q: fn(_p2(q) - off)


def union2(*fns: Callable) -> Callable:
    fns = tuple(fns)

    def f(q):
        d = fns[0](q)
        for g in fns[1:]:
            d = np.minimum(d, g(q))
        return d

    return f


def sub2(a: Callable, *bs: Callable) -> Callable:
    def f(q):
        d = a(q)
        for g in bs:
            d = np.maximum(d, -g(q))
        return d

    return f


def inter2(*fns: Callable) -> Callable:
    fns = tuple(fns)

    def f(q):
        d = fns[0](q)
        for g in fns[1:]:
            d = np.maximum(d, g(q))
        return d

    return f


def rot2(fn: Callable, angle: float) -> Callable:
    c, s = math.cos(-angle), math.sin(-angle)

    def f(q):
        v = _p2(q)
        return fn(np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c], axis=1))

    return f


# --------------------------------------------------------------------------
# 2D → 3D
# --------------------------------------------------------------------------


def extrude(d2: Callable, half_h: float, round_r: float = 0.0) -> SDF:
    """把 2D 距离沿 z 挤出成柱体（可选圆角）。"""

    def f(p):
        q = _p(p)
        d2v = d2(q[:, :2])
        dz = np.abs(q[:, 2]) - half_h
        outside = _len2(np.maximum(d2v, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(d2v, dz), 0.0)
        return outside + inside - round_r

    return f


def revolve(d2: Callable) -> SDF:
    """把 2D 剖面绕 z 轴旋转成实体（x 为半径，y 为高度）。"""

    def f(p):
        q = _p(p)
        r = _len2(q[:, 0], q[:, 1])
        return d2(np.stack([r, q[:, 2]], axis=1))

    return f


# --------------------------------------------------------------------------
# 布尔与修饰
# --------------------------------------------------------------------------


def union(*fns: SDF) -> SDF:
    fns = tuple(fns)
    if len(fns) == 1:
        return fns[0]

    def f(p):
        d = fns[0](p)
        for g in fns[1:]:
            d = np.minimum(d, g(p))
        return d

    return f


def inter(*fns: SDF) -> SDF:
    fns = tuple(fns)

    def f(p):
        d = fns[0](p)
        for g in fns[1:]:
            d = np.maximum(d, g(p))
        return d

    return f


def sub(a: SDF, *bs: SDF) -> SDF:
    def f(p):
        d = a(p)
        for g in bs:
            d = np.maximum(d, -g(p))
        return d

    return f


def smooth_union(k: float, *fns: SDF) -> SDF:
    fns = tuple(fns)
    if len(fns) == 1:
        return fns[0]

    def f(p):
        d = fns[0](p)
        for g in fns[1:]:
            d = _smin(d, g(p), k)
        return d

    return f


def smooth_sub(k: float, a: SDF, *bs: SDF) -> SDF:
    def f(p):
        d = a(p)
        for g in bs:
            d = _smax(d, -g(p), k)
        return d

    return f


def smooth_inter(k: float, a: SDF, b: SDF) -> SDF:
    return lambda p: _smax(a(p), b(p), k)


def shell(fn: SDF, thickness: float) -> SDF:
    """抽壳：保留距表面 thickness/2 以内的一层皮。

    注意：抽壳会把零等值面推到实体内部，而「并集」的距离场只在表面附近准确
    （内部会被图元自身的内部面污染）。因此抽壳只对**单一图元**或**精确距离场**
    安全；由多个图元拼出来的形体请改用 sub(外壳, 内芯) 的写法（见 hollow_lathe）。
    """
    t = max(thickness, 1e-4) * 0.5
    return lambda p: np.abs(fn(p)) - t


def hollow_lathe(profile: Sequence[Sequence[float]], wall: float,
                 blend: float = 0.0, open_top: bool = True) -> SDF:
    """把 (z, 半径) 剖面做成「有壁厚的中空旋转体」。

    用 sub(外壳, 内芯) 而不是 shell()：两者在数学上都对，但内芯是由剖面
    内缩后重新旋转出来的，距离场在内部依然干净，不会在壁中间冒出一层假表面。
    """
    pts = [(float(z), max(float(r), 1e-4)) for z, r in profile]
    outer = lathe(pts, blend=blend)
    w = max(wall, 1e-4)
    inner = [(z + w * 0.90, max(r - w, 0.008)) for z, r in pts]
    if open_top:
        z_top, r_top = pts[-1]
        inner.append((z_top + w * 2.0 + 0.02, max(r_top - w, 0.008)))
    core = lathe(inner, blend=blend)
    return sub(outer, core)


def drill(fn: SDF, holes: SDF, count: int, radius: float, *, phase: float = 0.0) -> SDF:
    """沿径向钻一圈孔（镂空 / 装饰穿孔）。孔是实体，用它去减。"""
    parts = []
    for i in range(max(1, int(count))):
        a = phase + TAU * i / count
        # 孔轴沿径向：把竖直圆柱转到水平，再推到半径处
        h = roty(holes, math.pi * 0.5)
        parts.append(rotz(at(h, 0.0, 0.0, 0.0), a))
    return sub(fn, *parts)


def offset(fn: SDF, d: float) -> SDF:
    return lambda p: fn(p) - d


def round_edges(fn: SDF, r: float) -> SDF:
    return lambda p: fn(p) - r


# --------------------------------------------------------------------------
# 变换
# --------------------------------------------------------------------------


def at(fn: SDF, x: float, y: float, z: float) -> SDF:
    off = np.array([x, y, z], dtype=np.float64)
    return lambda p: fn(_p(p) - off)


def _rot_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
    x, y, z = ax
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def rotated(fn: SDF, axis: Sequence[float], angle: float) -> SDF:
    """把实体绕 axis 旋转 angle 弧度（对点做逆变换）。"""
    M = _rot_matrix(axis, angle)
    Mt = M.T
    return lambda p: fn(_p(p) @ Mt.T)


def rotx(fn: SDF, angle: float) -> SDF:
    return rotated(fn, (1.0, 0.0, 0.0), angle)


def roty(fn: SDF, angle: float) -> SDF:
    return rotated(fn, (0.0, 1.0, 0.0), angle)


def rotz(fn: SDF, angle: float) -> SDF:
    return rotated(fn, (0.0, 0.0, 1.0), angle)


def scaled(fn: SDF, s: float | Sequence[float]) -> SDF:
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 0:
        s = np.repeat(float(s), 3)
    inv = 1.0 / np.maximum(s, 1e-9)
    scale_back = float(np.min(s))

    def f(p):
        return fn(_p(p) * inv) * scale_back

    return f


def mirror_x(fn: SDF) -> SDF:
    return lambda p: fn(_p(p) * np.array([-1.0, 1.0, 1.0]))


def mirror_y(fn: SDF) -> SDF:
    return lambda p: fn(_p(p) * np.array([1.0, -1.0, 1.0]))


def mirror_z(fn: SDF) -> SDF:
    return lambda p: fn(_p(p) * np.array([1.0, 1.0, -1.0]))


def symmetric_x(fn: SDF) -> SDF:
    """左右对称：对 |x| 求值（比镜像并集便宜且绝对对称）。"""

    def f(p):
        q = _p(p)
        q = np.stack([np.abs(q[:, 0]), q[:, 1], q[:, 2]], axis=1)
        return fn(q)

    return f


def twist(fn: SDF, amount: float, height: float = 1.0) -> SDF:
    """绕 z 轴按高度扭绞（amount 为整段高度的总扭转弧度）。"""
    half = max(height * 0.5, 1e-6)

    def f(p):
        q = _p(p)
        t = np.clip(q[:, 2] / half, -1.0, 1.0)
        a = amount * 0.5 * t
        c, s = np.cos(a), np.sin(a)
        x = q[:, 0] * c + q[:, 1] * s
        y = -q[:, 0] * s + q[:, 1] * c
        return fn(np.stack([x, y, q[:, 2]], axis=1))

    return f


def taper(fn: SDF, k: float, height: float = 1.0, axis_z: bool = True) -> SDF:
    """沿 z 线性缩放 xy（k>0 顶部收窄，k<0 顶部变宽）。"""
    half = max(height * 0.5, 1e-6)

    def f(p):
        q = _p(p)
        t = np.clip(q[:, 2] / half, -1.0, 1.0)
        s = np.maximum(1.0 + k * t, 0.12)
        if axis_z:
            return fn(np.stack([q[:, 0] / s, q[:, 1] / s, q[:, 2]], axis=1)) * s
        return fn(np.stack([q[:, 0] / s, q[:, 1], q[:, 2] / s], axis=1)) * s

    return f


def bend_z(fn: SDF, k: float) -> SDF:
    """让实体沿 z 轴弯成弓形（k 为曲率）。"""

    def f(p):
        q = _p(p)
        c = np.cos(k * q[:, 2])
        s = np.sin(k * q[:, 2])
        x = q[:, 0] * c - q[:, 1] * s
        y = q[:, 0] * s + q[:, 1] * c
        return fn(np.stack([x, y, q[:, 2]], axis=1))

    return f


# --------------------------------------------------------------------------
# 噪声 / 纹理
# --------------------------------------------------------------------------


def displace(fn: SDF, amp: float, freq: float = 3.0, seed: int = 0, octaves: int = 3) -> SDF:
    """沿法线做噪声位移：低频大起伏 + 高频细颗粒，是「有机感」的主要来源。"""
    if amp <= 1e-6:
        return fn
    off = np.array([seed * 0.37, seed * 0.11, seed * 0.73], dtype=np.float64)

    def f(p):
        q = _p(p) * freq + off
        n = _noise.fbm(q, seed=seed, octaves=octaves)
        return fn(_p(p)) - amp * (n - 0.5) * 2.0

    return f


def ridged_displace(fn: SDF, amp: float, freq: float = 4.0, seed: int = 0) -> SDF:
    """脊状位移：做出岩石 / 晶体的棱状肌理。"""
    if amp <= 1e-6:
        return fn
    off = np.array([seed * 0.21, seed * 0.53, seed * 0.37], dtype=np.float64)

    def f(p):
        q = _p(p) * freq + off
        n = _noise.ridged(q, seed=seed, octaves=3)
        return fn(_p(p)) - amp * (n - 0.5) * 2.0

    return f


def warp(fn: SDF, amp: float = 0.06, freq: float = 2.0, seed: int = 0) -> SDF:
    """域扭曲：把坐标本身推歪，得到黏土 / 融化的手感。"""
    if amp <= 1e-6:
        return fn
    off = np.array([seed * 0.19, seed * 0.41, seed * 0.29], dtype=np.float64)

    def f(p):
        q = _p(p)
        w = q * freq + off
        dx = _noise.fbm(w, seed=seed, octaves=2) - 0.5
        dy = _noise.fbm(w + 11.0, seed=seed + 7, octaves=2) - 0.5
        dz = _noise.fbm(w + 23.0, seed=seed + 13, octaves=2) - 0.5
        return fn(q + np.stack([dx, dy, dz], axis=1) * (amp * 2.0))

    return f


# --------------------------------------------------------------------------
# 阵列 / 重复
# --------------------------------------------------------------------------


def repeat_polar(fn: SDF, count: int, phase: float = 0.0) -> SDF:
    """极坐标域重复：把 0..2π/count 的扇形复制一圈（齿轮、花瓣、立柱环）。"""
    n = max(1, int(count))
    sector = TAU / n

    def f(p):
        q = _p(p)
        ang = np.arctan2(q[:, 1], q[:, 0]) - phase
        # 折叠到单个扇区，并保持到扇区边界的距离正确
        a = np.mod(ang + sector * 0.5, sector) - sector * 0.5
        r = _len2(q[:, 0], q[:, 1])
        x = r * np.cos(a)
        y = r * np.sin(a)
        return fn(np.stack([x, y, q[:, 2]], axis=1))

    return f


def ring_of(fn: SDF, count: int, radius: float, *, phase: float = 0.0,
            face_out: bool = True) -> SDF:
    """把实体绕 z 轴摆一圈（每份都朝向径向外侧）。"""
    n = max(1, int(count))
    parts = []
    for i in range(n):
        a = phase + TAU * i / n
        piece = rotz(fn, a) if face_out else fn
        parts.append(at(piece, radius * math.cos(a), radius * math.sin(a), 0.0))
    return union(*parts)


def steps(count: int, rise: float, run: float, width: float) -> SDF:
    """一段楼梯（沿 +y 上行）。"""
    n = max(1, int(count))
    parts = []
    for i in range(n):
        h = rise * (i + 1)
        parts.append(at(box(width * 0.5, run * 0.5, h * 0.5), 0.0, run * (i + 0.5), h * 0.5))
    return union(*parts)


def tube_path(points: Sequence[Sequence[float]], radius: float, *, sides: int = 1) -> SDF:
    """沿折线扫掠一串胶囊（触手、尾巴、树枝的通用构件）。"""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    if len(pts) < 2:
        raise ValueError("路径至少需要两个点")
    parts = [capsule(pts[i], pts[i + 1], radius) for i in range(len(pts) - 1)]
    return union(*parts) if len(parts) > 1 else parts[0]


def tapered_path(points: Sequence[Sequence[float]], r0: float, r1: float) -> SDF:
    """沿折线扫掠、半径线性变化的「锥形管」（尾巴 / 角 / 树枝）。"""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    n = len(pts) - 1
    parts = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = r0 + (r1 - r0) * t
        parts.append(capsule(pts[i], pts[i + 1], max(r, 1e-3)))
    return smooth_union(min(r0, r1) * 0.6, *parts)


def tube(points: Sequence[Sequence[float]], r0: float, r1: float | None = None,
         radii: Sequence[float] | None = None) -> SDF:
    """沿折线的锥形管，但用**一次求值内取分段最小**实现。

    比 tapered_path 快很多（后者是几十个胶囊嵌套求并集），
    适合纽结、海螺、触手这类上百段的曲线。
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    if len(pts) < 2:
        raise ValueError("路径至少需要两个点")
    r1 = r0 if r1 is None else r1
    n = len(pts) - 1
    segs = []
    for i in range(n):
        if radii is not None:
            r = float(radii[min(i, len(radii) - 1)])
        else:
            t = i / max(n - 1, 1)
            r = r0 + (r1 - r0) * t
        segs.append((pts[i], pts[i + 1], max(r, 1e-3)))

    def f(p):
        q = _p(p)
        best = np.full(q.shape[0], 1e18)
        best_r = np.full(q.shape[0], r0)
        for a, b, r in segs:
            ab = b - a
            ab2 = max(float(ab @ ab), 1e-12)
            d = q - a
            t = np.clip((d @ ab) / ab2, 0.0, 1.0)
            ex = d - t[:, None] * ab
            d2 = ex[:, 0] ** 2 + ex[:, 1] ** 2 + ex[:, 2] ** 2
            closer = d2 < best
            best = np.where(closer, d2, best)
            best_r = np.where(closer, r, best_r)
        return np.sqrt(best) - best_r

    return f


# --------------------------------------------------------------------------
# 刻面 / 切割
# --------------------------------------------------------------------------


def cut_planes(fn: SDF, count: int, seed: int = 0, radius: float = 1.0,
               depth: float = 0.18) -> SDF:
    """用若干随机半空间切出平面 —— 宝石 / 低多边形 / 硬朗棱面的来源。"""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    cuts = [fn]
    for _ in range(max(0, int(count))):
        v = rng.normal(size=3)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        d = radius * (1.0 - depth * float(rng.random()))
        cuts.append(plane(v, d))
    return inter(*cuts)


def flute(fn: SDF, lobes: int, amp: float = 0.06, *,
          fade_from: float | None = None, fade_to: float | None = None) -> SDF:
    """竖棱：按角度调制半径（比「减一圈圆柱」干净得多，不会在壁上留刀口）。

    amp 是半径的相对起伏；距离场会被同比例拉伸，所以再除回去做近似校正。
    fade_from / fade_to 让竖棱在某段高度内渐隐 —— 口沿太薄时不要在上面刻棱，
    否则薄环会被刻成锯齿。
    """
    n = max(2, int(lobes))
    if amp <= 1e-4:
        return fn

    def f(p):
        q = _p(p)
        rad = _len2(q[:, 0], q[:, 1])
        ang = np.arctan2(q[:, 1], q[:, 0])
        w = 1.0
        if fade_from is not None and fade_to is not None and fade_to > fade_from:
            t = (q[:, 2] - fade_from) / (fade_to - fade_from)
            t = np.clip(t, 0.0, 1.0)
            w = 1.0 - (t * t * (3.0 - 2.0 * t))
        k = 1.0 + amp * np.cos(n * ang) * w
        x = rad * k * np.cos(ang)
        y = rad * k * np.sin(ang)
        return fn(np.stack([x, y, q[:, 2]], axis=1)) / np.maximum(k, 0.2)

    return f


def facet(fn: SDF, count: int, seed: int = 0, depth: float = 0.16) -> SDF:
    """刻面：用一组随机平面把圆润表面切成宝石 / 低多边形的硬棱面。"""
    return cut_planes(fn, count, seed=seed, radius=1.0, depth=depth)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def blend_all(k: float, parts: Iterable[SDF]) -> SDF:
    parts = tuple(parts)
    if not parts:
        raise ValueError("至少需要一个部件")
    return smooth_union(k, *parts)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def polar_path(radius_fn, z_fn, turns: float, steps_n: int) -> list[tuple[float, float, float]]:
    """生成螺旋 / 环形路径点（把手、触手、螺纹用）。"""
    out = []
    for i in range(max(2, steps_n)):
        t = i / (steps_n - 1)
        a = TAU * turns * t
        r = radius_fn(t)
        out.append((r * math.cos(a), r * math.sin(a), z_fn(t)))
    return out
