"""printcheck.py —— 打印体检：把网格翻译成「这台机器能不能打、怎么打」。

检查项都是切片前真正会踩坑的地方：
    * 水密 / 非流形 —— 有破面切片会乱补；
    * 包围盒与体积 —— 是否超出热床、要多少料；
    * 悬垂面积占比 —— 决定要多少支撑；
    * 底部接触面 —— 决定要不要加底座（raft/brim）；
    * 最佳摆放 —— 试几个正交朝向，挑悬垂最少、高度最低的那个。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh

PLA_DENSITY = 1.24   # g/cm³
RESIN_DENSITY = 1.10
OVERHANG_LIMIT = math.cos(math.radians(45.0))  # 法线 z 分量低于 -cos45° 视为悬垂


@dataclass
class PrintReport:
    faces: int = 0
    verts: int = 0
    watertight: bool = False
    boundary_edges: int = 0
    nonmanifold_edges: int = 0
    size_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_cm3: float = 0.0
    weight_pla_g: float = 0.0
    weight_resin_g: float = 0.0
    overhang_ratio: float = 0.0
    footprint_ratio: float = 0.0
    grade: str = "—"
    notes: list[str] = field(default_factory=list)
    orientation: str = ""

    def bbox_text(self) -> str:
        x, y, z = self.size_mm
        return f"{x:.0f} × {y:.0f} × {z:.0f} mm"

    def summary(self) -> str:
        return f"{self.bbox_text()} · {self.volume_cm3:.1f} cm³ · {self.weight_pla_g:.0f} g"

    def to_lines(self) -> list[str]:
        lines = [
            f"网格：{self.faces:,} 面 / {self.verts:,} 顶点",
            f"水密性：{'封闭 ✓' if self.watertight else '有缺陷 ✗'}"
            f"（边界边 {self.boundary_edges}，非流形边 {self.nonmanifold_edges}）",
            f"包围盒：{self.bbox_text()}",
            f"体积：{self.volume_cm3:.2f} cm³（约 PLA {self.weight_pla_g:.1f} g / 树脂 {self.weight_resin_g:.1f} g）",
            f"悬垂面积：{self.overhang_ratio * 100:.0f}%（45° 判据）",
            f"底部接触：{self.footprint_ratio * 100:.0f}% 投影面积",
        ]
        if self.orientation:
            lines.append(f"建议摆放：{self.orientation}")
        lines.extend(self.notes)
        return lines


def _face_areas(mesh: Mesh) -> np.ndarray:
    v = mesh.verts[mesh.faces].astype(np.float64)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return 0.5 * np.linalg.norm(n, axis=1)


def _overhang(mesh: Mesh) -> tuple[float, float]:
    """返回 (悬垂面积占比, 底部投影接触比)。"""
    if mesh.is_empty():
        return 0.0, 0.0
    areas = _face_areas(mesh)
    total = float(areas.sum())
    if total <= 1e-9:
        return 0.0, 0.0
    normals = mesh.face_normals(normalize=True)
    lo, hi = mesh.bounds()
    z_min = float(lo[2])
    centroid_z = mesh.verts[mesh.faces].mean(axis=1)[:, 2]
    # 贴床面（最低 2% 高度内、法线朝下）不算悬垂
    height = float(hi[2] - lo[2])
    near_bed = centroid_z < z_min + max(height * 0.02, 1e-9)
    down = normals[:, 2] < -OVERHANG_LIMIT
    overhang = float(areas[down & ~near_bed].sum()) / total

    # 底部接触：朝下且贴床的面的投影面积 / 整体投影面积
    foot = float(areas[down & near_bed].sum())
    footprint = foot / total if total > 0 else 0.0
    return overhang, footprint


def analyze(mesh: Mesh) -> PrintReport:
    if mesh is None or mesh.is_empty():
        return PrintReport(notes=["没有可检查的模型"])
    ok, boundary, nonmanifold = mesh.check_watertight()
    lo, hi = mesh.bounds()
    size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
    vol_mm3 = abs(mesh.signed_volume())
    vol_cm3 = vol_mm3 / 1000.0
    overhang, footprint = _overhang(mesh)

    report = PrintReport(
        faces=mesh.n_faces,
        verts=mesh.n_verts,
        watertight=ok,
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        size_mm=size,
        volume_cm3=vol_cm3,
        weight_pla_g=vol_cm3 * PLA_DENSITY,
        weight_resin_g=vol_cm3 * RESIN_DENSITY,
        overhang_ratio=overhang,
        footprint_ratio=footprint,
    )

    notes: list[str] = []
    if not ok:
        notes.append("模型有非流形/破面：切片软件通常会自动修复，但建议先换一档精度重生成。")
    if max(size) > 256.0:
        notes.append("最长边超过 256mm，多数桌面机放不下：请缩小尺寸或分件打印。")
    elif max(size) > 200.0:
        notes.append("尺寸偏大，注意热床尺寸与翘边风险。")
    if overhang > 0.35:
        notes.append(f"悬垂面积 {overhang * 100:.0f}%，需要较多支撑；可试「自动摆正」。")
    elif overhang > 0.18:
        notes.append(f"悬垂面积 {overhang * 100:.0f}%，局部需要支撑。")
    else:
        notes.append("悬垂很少，基本可以免支撑打印。")
    if footprint < 0.04:
        notes.append("底部接触面很小：建议加 brim 或加一个底座。")
    if vol_cm3 < 0.2:
        notes.append("体积很小，注意最小特征尺寸不要低于喷嘴直径。")

    if ok and overhang <= 0.18 and footprint >= 0.04:
        report.grade = "优"
    elif ok or (boundary == 0 and nonmanifold <= 20):
        report.grade = "良"
    else:
        report.grade = "需注意"
    report.notes = notes
    return report


def _rotate(mesh: Mesh, rx: int, ry: int) -> Mesh:
    v = mesh.verts.astype(np.float64)
    for _ in range(rx % 4):
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    for _ in range(ry % 4):
        v = np.stack([v[:, 2], v[:, 1], -v[:, 0]], axis=1)
    return Mesh(v.astype(np.float32), mesh.faces.copy(), dict(mesh.meta))


def auto_orient(mesh: Mesh) -> tuple[Mesh, PrintReport, str]:
    """在 8 个正交朝向里挑一个最适合打印的，并落回 z=0。"""
    best: tuple[float, Mesh, PrintReport, str] | None = None
    labels = {(0, 0): "原朝向", (0, 1): "绕 Y 转 90°", (1, 0): "绕 X 转 90°",
              (1, 1): "侧放", (2, 0): "倒置", (0, 3): "绕 Y 转 270°",
              (3, 0): "绕 X 转 270°", (2, 2): "倒置侧放"}
    for rx in range(4):
        for ry in range(2):
            cand = _rotate(mesh, rx, ry)
            lo, hi = cand.bounds()
            cand.verts = (cand.verts - np.array([0, 0, float(lo[2])], dtype=np.float32)).astype(np.float32)
            rep = analyze(cand)
            # 评分：悬垂越少越好、底部越稳越好、高度越低越省时
            score = rep.overhang_ratio * 3.0 - rep.footprint_ratio * 1.2 + (hi[2] - lo[2]) / 400.0
            if best is None or score < best[0]:
                best = (score, cand, rep, labels.get((rx, ry), f"旋转 {rx}/{ry}"))
    assert best is not None
    _, mesh_out, report, label = best
    report.orientation = label
    return mesh_out, report, label


def report_text(report: PrintReport, *, title: str = "") -> str:
    head = f"打印体检{f'：{title}' if title else ''}"
    return "\n".join([head, "=" * 24] + report.to_lines())
