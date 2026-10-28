"""noise.py —— numpy 向量化的三维值噪声 / fBm / 脊状噪声。

全部函数接收 (N, 3) 的坐标数组并返回 (N,) 的噪声值，便于一次性处理
几万个顶点，避免 v1 逐点循环带来的卡顿。
"""

from __future__ import annotations

import numpy as np

_MASK = np.uint32(0xFFFFFFFF)


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """整数格点哈希 → [0, 1) 的伪随机值。"""
    x = ix.astype(np.uint64) * np.uint64(374761393)
    y = iy.astype(np.uint64) * np.uint64(668265263)
    z = iz.astype(np.uint64) * np.uint64(2147483647)
    h = (x + y + z + np.uint64(seed * 144665 + 1013904223)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """三线性插值的值噪声，范围 [0, 1]。"""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 3)
    i = np.floor(p)
    f = p - i
    ix, iy, iz = i[:, 0].astype(np.int64), i[:, 1].astype(np.int64), i[:, 2].astype(np.int64)
    u, v, w = _fade(f[:, 0]), _fade(f[:, 1]), _fade(f[:, 2])

    def corner(dx: int, dy: int, dz: int) -> np.ndarray:
        return _hash3(ix + dx, iy + dy, iz + dz, seed)

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)

    x00 = c000 + (c100 - c000) * u
    x10 = c010 + (c110 - c010) * u
    x01 = c001 + (c101 - c001) * u
    x11 = c011 + (c111 - c011) * u
    y0 = x00 + (x10 - x00) * v
    y1 = x01 + (x11 - x01) * v
    return y0 + (y1 - y0) * w


def fbm(
    points: np.ndarray,
    seed: int = 0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """分形叠加噪声，范围约 [0, 1]。"""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    total = np.zeros(p.shape[0], dtype=np.float64)
    amp = 1.0
    norm = 0.0
    freq = 1.0
    for o in range(max(1, octaves)):
        total += amp * value_noise(p * freq, seed + o * 131)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-9)


def ridged(points: np.ndarray, seed: int = 0, octaves: int = 4) -> np.ndarray:
    """脊状噪声：把 fBm 折成尖锐山脊，适合晶体/岩石质感。"""
    n = fbm(points, seed=seed, octaves=octaves)
    return 1.0 - np.abs(2.0 * n - 1.0)


def fbm1(t: np.ndarray, seed: int = 0, octaves: int = 3) -> np.ndarray:
    """一维 fBm（用于轮廓曲线），输入 (N,) 返回 (N,)。"""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    pts = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    return fbm(pts, seed=seed, octaves=octaves)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 == edge0:
        return np.clip((x - edge0) * 1e9, 0.0, 1.0)
    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
