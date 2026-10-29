"""sdfmesh.py —— 把 SDF 变成三角网格（Surface Nets 多边形化）。

为什么不用 Marching Cubes：
    * MC 需要 256 项查找表，代码长、易写错；
    * MC 在细薄结构上容易产生退化三角形，切片软件会报「非流形」。
    Surface Nets（对偶等值面）每个跨越表面的体素只放一个顶点，
    再把这些顶点按网格边连成四边形 —— 天然封闭、天然流形、顶点数更省，
    对有机造型尤其友好。

精度怎么保证：
    1. 两遍采样：先粗采样定位表面包围盒，再把网格分辨率全部花在有效区域上；
    2. 顶点不放在体素中心，而是放在「12 条棱与表面的交点」的质心；
    3. Taubin 平滑（λ/μ 交替）去阶梯感且不缩体积；
    4. 牛顿投影：沿 SDF 梯度把顶点推到真正的零等值面上。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .errors import GenerationError
from .mesh import Mesh

# 体素 8 个角在 (i, j, k) 上的偏移
CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)
# 12 条棱（连接有符号差异的两个角）
EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

MAX_GRID_CELLS = 48_000_000  # 约 190MB float32，超过就自动降分辨率


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------


def _eval_grid(fn: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
               chunk: int = 3_000_000) -> np.ndarray:
    """在规则网格上求值，按 x 方向切片以限制峰值内存。"""
    nx, ny, nz = len(xs), len(ys), len(zs)
    out = np.empty((nx, ny, nz), dtype=np.float32)
    slab = max(1, int(chunk // max(ny * nz, 1)))
    for i0 in range(0, nx, slab):
        i1 = min(nx, i0 + slab)
        gx, gy, gz = np.meshgrid(xs[i0:i1], ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = np.asarray(fn(pts), dtype=np.float64).reshape(-1)
        vals = np.nan_to_num(vals, nan=1e3, posinf=1e3, neginf=-1e3)
        out[i0:i1] = vals.reshape(i1 - i0, ny, nz).astype(np.float32)
    return out


def _surface_bbox(fn: Callable, domain: float, res: int = 44) -> tuple[np.ndarray, np.ndarray] | None:
    """粗采样定位表面所在范围（返回 None 表示这个尺度下没有表面）。"""
    step = 2.0 * domain / res
    axis = np.linspace(-domain, domain, res + 1)
    F = _eval_grid(fn, axis, axis, axis)
    inside = F < 0.0
    if not inside.any() or inside.all():
        return None
    # 只要有一个角在内部、另一个在外部，这个体素就跨越表面
    cross = np.zeros((res, res, res), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                sl = F[dx:dx + res, dy:dy + res, dz:dz + res] < 0.0
                cross |= sl
    both = cross & ~np.all(
        np.stack([F[dx:dx + res, dy:dy + res, dz:dz + res] < 0.0
                  for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]), axis=0
    )
    if not both.any():
        return None
    idx = np.argwhere(both)
    # 表面碰到采样盒边界 = 这个尺度装不下，换更大的盒子再试
    if (idx.min(axis=0) <= 0).any() or (idx.max(axis=0) >= res - 1).any():
        return None
    lo = -domain + idx.min(axis=0) * step
    hi = -domain + (idx.max(axis=0) + 1) * step
    margin = (hi - lo) * 0.12 + step
    return lo - margin, hi + margin


def auto_bounds(fn: Callable) -> tuple[np.ndarray, np.ndarray]:
    """自动找一个能完整包住实体的包围盒（容忍任意局部尺度）。"""
    for domain in (0.08, 0.35, 1.35, 5.0, 22.0, 90.0):
        box = _surface_bbox(fn, domain)
        if box is not None:
            return box
    raise GenerationError(
        "这个描述没有生成出实体。",
        hint="换一种说法，或换一个模型；也可以试试「随机灵感」。",
        detail="SDF 在 0.08–90 单位范围内都没有找到表面",
    )


# --------------------------------------------------------------------------
# Surface Nets
# --------------------------------------------------------------------------


def _surface_nets(F: np.ndarray, step: float, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = F.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    if cx < 1 or cy < 1 or cz < 1:
        raise GenerationError("分辨率太低，无法生成网格。", hint="把精度调高一档。")

    corners = [F[o[0]:o[0] + cx, o[1]:o[1] + cy, o[2]:o[2] + cz] for o in CORNERS]
    any_in = np.zeros((cx, cy, cz), dtype=bool)
    all_in = np.ones((cx, cy, cz), dtype=bool)
    for c in corners:
        m = c < 0.0
        any_in |= m
        all_in &= m
    active = any_in & ~all_in
    idx = np.argwhere(active)
    if idx.size == 0:
        raise GenerationError("表面没有落在采样范围内。", hint="换一个模型再试。")

    ai, aj, ak = idx[:, 0], idx[:, 1], idx[:, 2]
    acc = np.zeros((len(idx), 3), dtype=np.float64)
    cnt = np.zeros(len(idx), dtype=np.float64)
    for a, b in EDGES:
        fa = corners[a][ai, aj, ak].astype(np.float64)
        fb = corners[b][ai, aj, ak].astype(np.float64)
        cross = (fa < 0.0) != (fb < 0.0)
        denom = np.where(cross, fa - fb, 1.0)
        t = np.clip(np.where(cross, fa / denom, 0.0), 0.0, 1.0)
        oa = np.array(CORNERS[a], dtype=np.float64)
        ob = np.array(CORNERS[b], dtype=np.float64)
        pos = oa + (ob - oa) * t[:, None]
        acc += np.where(cross[:, None], pos, 0.0)
        cnt += cross
    local = acc / np.maximum(cnt, 1.0)[:, None]
    verts = origin + (idx + local) * step

    cell_index = np.full((cx, cy, cz), -1, dtype=np.int64)
    cell_index[ai, aj, ak] = np.arange(len(idx))

    quads: list[np.ndarray] = []

    # ---- 沿 x 的网格边：相邻两个采样点跨过表面
    a = F[0:cx, 1:cy, 1:cz]
    b = F[1:nx, 1:cy, 1:cz]
    cross = (a < 0.0) != (b < 0.0)
    if cross.any():
        c0 = cell_index[0:cx, 0:cy - 1, 0:cz - 1]
        c1 = cell_index[0:cx, 1:cy, 0:cz - 1]
        c2 = cell_index[0:cx, 1:cy, 1:cz]
        c3 = cell_index[0:cx, 0:cy - 1, 1:cz]
        ok = cross & (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        q = np.stack([c0[ok], c1[ok], c2[ok], c3[ok]], axis=1)
        flip = (a[ok] > 0.0)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)

    # ---- 沿 y
    a = F[1:cx, 0:cy, 1:cz]
    b = F[1:cx, 1:ny, 1:cz]
    cross = (a < 0.0) != (b < 0.0)
    if cross.any():
        c0 = cell_index[0:cx - 1, 0:cy, 0:cz - 1]
        c1 = cell_index[0:cx - 1, 0:cy, 1:cz]
        c2 = cell_index[1:cx, 0:cy, 1:cz]
        c3 = cell_index[1:cx, 0:cy, 0:cz - 1]
        ok = cross & (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        q = np.stack([c0[ok], c1[ok], c2[ok], c3[ok]], axis=1)
        flip = (a[ok] > 0.0)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)

    # ---- 沿 z
    a = F[1:cx, 1:cy, 0:cz]
    b = F[1:cx, 1:cy, 1:nz]
    cross = (a < 0.0) != (b < 0.0)
    if cross.any():
        c0 = cell_index[0:cx - 1, 0:cy - 1, 0:cz]
        c1 = cell_index[1:cx, 0:cy - 1, 0:cz]
        c2 = cell_index[1:cx, 1:cy, 0:cz]
        c3 = cell_index[0:cx - 1, 1:cy, 0:cz]
        ok = cross & (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        q = np.stack([c0[ok], c1[ok], c2[ok], c3[ok]], axis=1)
        flip = (a[ok] > 0.0)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)

    if not quads:
        raise GenerationError("表面没有生成任何面片。", hint="把精度调高一点再试。")

    q = np.concatenate(quads, axis=0)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=0).astype(np.int32)
    return verts.astype(np.float32), faces


# --------------------------------------------------------------------------
# 后处理
# --------------------------------------------------------------------------


def _smooth(verts: np.ndarray, faces: np.ndarray, iterations: int,
            lam: float = 0.55, mu: float = -0.58) -> np.ndarray:
    """Taubin 平滑：λ 收缩、μ 反向补偿，去掉体素台阶又不缩小体积。"""
    if iterations <= 0:
        return verts
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    deg = np.maximum(np.bincount(e[:, 0], minlength=n).astype(np.float64), 1.0)
    v = verts.astype(np.float64)
    for i in range(int(iterations) * 2):
        acc = np.zeros_like(v)
        np.add.at(acc, e[:, 0], v[e[:, 1]])
        avg = acc / deg[:, None]
        v = v + (avg - v) * (lam if i % 2 == 0 else mu)
    return v.astype(np.float32)


def _project(verts: np.ndarray, fn: Callable, step: float, iterations: int,
             clamp: float = 0.75) -> np.ndarray:
    """沿 SDF 梯度把顶点推到零等值面上（牛顿一步近似）。"""
    if iterations <= 0:
        return verts
    eps = step * 0.5
    v = verts.astype(np.float64)
    limit = step * clamp
    for _ in range(int(iterations)):
        f0 = np.asarray(fn(v), dtype=np.float64)
        gx = (np.asarray(fn(v + [eps, 0, 0]), dtype=np.float64) - f0) / eps
        gy = (np.asarray(fn(v + [0, eps, 0]), dtype=np.float64) - f0) / eps
        gz = (np.asarray(fn(v + [0, 0, eps]), dtype=np.float64) - f0) / eps
        g = np.stack([gx, gy, gz], axis=1)
        gl2 = np.einsum("ij,ij->i", g, g)
        t = f0 / np.maximum(gl2, 1e-9)
        t = np.clip(t, -limit, limit)
        v = v - g * t[:, None]
    return v.astype(np.float32)


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------


def polygonize(
    fn: Callable,
    res: int = 96,
    *,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    refine: int = 2,
    smooth: int = 1,
    pad: float = 2.0,
    grid_shift: int = 0,
    erode: float = 0.0,
    progress: Callable[[str, float], None] | None = None,
) -> Mesh:
    """把 SDF 转成封闭三角网格。

    res      —— 最长边方向上的采样数（精度档位就是它）
    refine   —— 牛顿投影迭代次数（把顶点贴到真实表面上）
    smooth   —— Taubin 平滑迭代次数（去体素台阶）
    pad      —— 包围盒外扩的体素数，保证表面不会碰到网格边界
    grid_shift —— 采样网格的亚体素偏移；用来躲开「表面正好落在采样平面上」
                  造成的退化（0 为不偏移，1/2 为两组错开的偏移）
    """
    res = int(max(14, min(512, res)))

    def tick(label: str, frac: float) -> None:
        if progress:
            try:
                progress(label, frac)
            except Exception:  # noqa: BLE001 - 进度回调不该影响建模
                pass

    tick("定位模型范围", 0.02)
    lo, hi = bounds if bounds is not None else auto_bounds(fn)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    ext = np.maximum(hi - lo, 1e-6)
    step = float(ext.max()) / res

    lo = lo - step * pad
    hi = hi + step * pad
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    cells = float(dims[0]) * dims[1] * dims[2]
    if cells > MAX_GRID_CELLS:
        # 网格太大：放大步长而不是硬扛，避免吃满内存
        step *= (cells / MAX_GRID_CELLS) ** (1.0 / 3.0) * 1.02
        dims = np.ceil((hi - lo) / step).astype(int) + 1

    # 亚体素偏移：让采样平面避开 0、整数、模型自身的对称平面
    shifts = ((0.0, 0.0, 0.0), (0.211, 0.377, 0.463), (0.523, 0.129, 0.717))
    origin = lo + np.asarray(shifts[int(grid_shift) % len(shifts)], dtype=np.float64) * step

    xs = origin[0] + step * np.arange(dims[0])
    ys = origin[1] + step * np.arange(dims[1])
    zs = origin[2] + step * np.arange(dims[2])

    # 亚体素侵蚀：把「刚好相切」的零宽度颈部轻轻拉开，避免等值面在一条线上退化
    field = (lambda q: fn(q) - erode * step) if erode > 0.0 else fn

    tick("采样距离场", 0.08)
    F = _eval_grid(field, xs, ys, zs)
    # 边界强制为外部，保证等值面不与采样盒相交 —— 网格一定封闭
    F[0, :, :] = np.maximum(F[0, :, :], step)
    F[-1, :, :] = np.maximum(F[-1, :, :], step)
    F[:, 0, :] = np.maximum(F[:, 0, :], step)
    F[:, -1, :] = np.maximum(F[:, -1, :], step)
    F[:, :, 0] = np.maximum(F[:, :, 0], step)
    F[:, :, -1] = np.maximum(F[:, :, -1], step)

    tick("提取表面", 0.45)
    verts, faces = _surface_nets(F, step, origin)

    if smooth > 0:
        tick("平滑表面", 0.62)
        verts = _smooth(verts, faces, smooth)

    if refine > 0:
        tick("贴合真实表面", 0.74)
        verts = _project(verts, field, step, refine)

    mesh = Mesh(verts, faces)
    if mesh.signed_volume() < 0.0:
        mesh.faces = mesh.faces[:, ::-1]
    mesh.meta["sdf"] = True
    mesh.meta["grid"] = int(res)
    mesh.meta["step"] = float(step)
    return mesh


def mesh_from_sdf(fn: Callable, quality, *, progress=None) -> Mesh:
    """按精度档位生成网格；若出现非流形边就换一组网格偏移重试。

    「两个面正好重合」这类建模巧合会让等值面在一条线上退化，
    换一个亚体素偏移通常就能躲开。最多试三次，取问题最少的一版。
    """
    best: Mesh | None = None
    best_bad = 1 << 30
    # 先试「不侵蚀」的三组网格偏移；仍有退化时再轻微侵蚀（几何上只是把
    # 零宽度的接触点拉开半个体素，肉眼不可见）
    plans = ((0, 0.0), (1, 0.45), (2, 0.85))
    for attempt, erode in plans:
        mesh = polygonize(
            fn,
            quality.grid,
            refine=quality.refine,
            smooth=quality.smooth,
            grid_shift=attempt,
            erode=erode,
            progress=progress,
        )
        ok, boundary, nonmanifold = mesh.check_watertight()
        bad = boundary + nonmanifold
        if bad == 0:
            return mesh
        if bad < best_bad:
            best, best_bad = mesh, bad
    assert best is not None
    best.meta["nonmanifold"] = best_bad
    return best
