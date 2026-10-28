"""mesh.py —— 三角网格数据结构、法线计算、拓扑自检与 STL / OBJ 导出。

与 v1 的差异：
* 顶点/面改用 numpy 数组，批量运算（变换、法线、着色）快 1~2 个数量级；
* 新增 `fix_winding()` / `check_watertight()`，保证「封闭实体 + 法线朝外」不再靠
  生成器自己写对，而是导出前统一校验、必要时自动修正；
* 导出改为原子写（先写 .tmp 再替换），导出中断不会留下半截文件。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    verts: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32
    meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.verts = np.ascontiguousarray(self.verts, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if self.verts.ndim != 2 or self.verts.shape[1] != 3:
            raise ValueError("顶点数组必须是 (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("面数组必须是 (M, 3)")

    # ---------------------------------------------------------------- 基本信息
    @property
    def n_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def is_empty(self) -> bool:
        return self.n_faces == 0 or self.n_verts == 0

    def stats_text(self) -> str:
        return f"{self.n_faces:,} 面 / {self.n_verts:,} 顶点"

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) * 0.5

    # ---------------------------------------------------------------- 变换
    def normalize(self, target: float = 100.0, *, floor: bool = True) -> "Mesh":
        """居中并缩放到最长边 = target（毫米）。floor=True 时底面落在 z=0。"""
        if self.is_empty():
            return self
        lo, hi = self.bounds()
        size = hi - lo
        longest = float(np.max(size))
        if not math.isfinite(longest) or longest <= 1e-9:
            return self
        k = target / longest
        center = (lo + hi) * 0.5
        v = (self.verts - center) * k
        if floor:
            v[:, 2] -= float(v[:, 2].min())
        self.verts = v.astype(np.float32)
        self._cache.clear()
        return self

    def translated(self, offset) -> "Mesh":
        return Mesh(self.verts + np.asarray(offset, dtype=np.float32), self.faces, dict(self.meta))

    # ---------------------------------------------------------------- 法线
    def face_normals(self, normalize: bool = True) -> np.ndarray:
        key = "fn" if normalize else "fnn"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        v = self.verts
        f = self.faces
        a = v[f[:, 0]]
        b = v[f[:, 1]]
        c = v[f[:, 2]]
        n = np.cross(b - a, c - a)
        if normalize:
            length = np.linalg.norm(n, axis=1, keepdims=True)
            np.divide(n, np.maximum(length, 1e-12), out=n)
        else:
            # 未归一化时长度信息有用（面积），仍然缓存
            pass
        n = n.astype(np.float32)
        self._cache[key] = n
        return n

    def vertex_normals(self) -> np.ndarray:
        cached = self._cache.get("vn")
        if cached is not None:
            return cached
        n = self.face_normals(normalize=False)
        out = np.zeros_like(self.verts, dtype=np.float32)
        for i in range(3):
            np.add.at(out, self.faces[:, i], n)
        length = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, np.maximum(length, 1e-12), out=out)
        out = out.astype(np.float32)
        self._cache["vn"] = out
        return out

    # ---------------------------------------------------------------- 拓扑
    def check_watertight(self) -> tuple[bool, int, int]:
        """返回 (是否封闭, 边界边数量, 非流形边数量)。

        封闭 = 每条无向边恰好被两个三角形共享。
        """
        if self.is_empty():
            return False, 0, 0
        f = self.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        und = np.sort(edges, axis=1)
        uniq, counts = np.unique(und, axis=0, return_counts=True)
        boundary = int(np.sum(counts == 1))
        nonmanifold = int(np.sum(counts > 2))
        return (boundary == 0 and nonmanifold == 0), boundary, nonmanifold

    def signed_volume(self) -> float:
        v = self.verts.astype(np.float64)
        f = self.faces
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)

    def fix_winding(self) -> "Mesh":
        """让整张网格的三角形绕向一致，并整体朝外（体积为正）。

        做法：以共享边为邻接关系做 BFS，相邻面若沿共享边方向相同则翻转；
        最后用有向体积判断整体是否需要翻转。
        """
        if self.is_empty():
            return self
        f = self.faces.copy()
        m = f.shape[0]
        # 边 -> [(face, dir)]，dir=+1 表示该面沿 (lo->hi) 方向
        edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for fi in range(m):
            tri = f[fi]
            for k in range(3):
                a, b = int(tri[k]), int(tri[(k + 1) % 3])
                key = (a, b) if a < b else (b, a)
                edge_map.setdefault(key, []).append((fi, 1 if a < b else -1))

        # 面邻接
        adj: list[list[tuple[int, int]]] = [[] for _ in range(m)]
        for key, items in edge_map.items():
            if len(items) == 2:
                (f1, d1), (f2, d2) = items
                # 若方向相同，说明两者绕向冲突，需要翻转其中一个
                need_flip = d1 == d2
                adj[f1].append((f2, need_flip))
                adj[f2].append((f1, need_flip))

        flip = np.zeros(m, dtype=bool)
        seen = np.zeros(m, dtype=bool)
        for start in range(m):
            if seen[start]:
                continue
            seen[start] = True
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt, conflict in adj[cur]:
                    if seen[nxt]:
                        continue
                    seen[nxt] = True
                    flip[nxt] = flip[cur] ^ conflict
                    stack.append(nxt)

        f[flip] = f[flip][:, ::-1]
        self.faces = f
        if self.signed_volume() < 0:
            self.faces = self.faces[:, ::-1]
        self._cache.clear()
        return self

    def validate(self) -> list[str]:
        """返回问题列表（空列表表示健康）。"""
        problems: list[str] = []
        if self.is_empty():
            problems.append("网格为空")
            return problems
        if self.faces.min() < 0 or self.faces.max() >= self.n_verts:
            problems.append("面索引越界")
        if not np.all(np.isfinite(self.verts)):
            problems.append("顶点含 NaN / Inf")
        ok, boundary, nonmanifold = self.check_watertight()
        if boundary:
            problems.append(f"{boundary} 条边界边（模型未封闭，切片可能报错）")
        if nonmanifold:
            problems.append(f"{nonmanifold} 条非流形边")
        if self.n_faces < 8:
            problems.append("面数过少")
        return problems

    # ---------------------------------------------------------------- 导出
    def stl_bytes(self, name: str = "creative3d") -> bytes:
        """二进制 STL（法线 + 3 顶点 + 属性，共 50 字节/面）。"""
        normals = self.face_normals(normalize=True)
        tri = self.verts[self.faces].astype(np.float32)

        rec = np.zeros(
            self.n_faces,
            dtype=[
                ("n", "<f4", (3,)),
                ("v", "<f4", (3, 3)),
                ("attr", "<u2"),
            ],
        )
        rec["n"] = normals
        rec["v"] = tri
        header = f"Creative3D | {name} | {self.n_faces} tris".encode("ascii", "replace")[:79]
        header = header.ljust(80, b" ")
        return header + np.uint32(self.n_faces).tobytes() + rec.tobytes()

    def obj_text(self, name: str = "creative3d", palette: str | None = None) -> str:
        lines: list[str] = [
            f"# Creative3D v2 · {name}",
            f"# {self.stats_text()}",
            "# 单位：毫米（最长边 100mm，底面位于 z=0）",
            f"o {name}",
        ]
        if palette:
            lines.append(f"# palette: {palette}")
        v = self.verts
        lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v)
        f = self.faces + 1
        lines.extend(f"f {a} {b} {c}" for a, b, c in f)
        return "\n".join(lines) + "\n"


_SAFE_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def safe_filename(text: str, fallback: str = "creative3d", max_len: int = 60) -> str:
    """把任意文本变成 Windows 合法文件名。"""
    cleaned = _SAFE_RE.sub("_", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """先写临时文件再原子替换，避免中断留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
        import os

        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))
