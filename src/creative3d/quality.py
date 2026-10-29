"""quality.py —— 精度档位（用户可选的「要多细」）。

精度不是滑杆，而是几档有明确代价的预设：每档都知道自己大概多少面、
多少内存、大概要等多久。这样用户在「打印级」上花的时间是可预期的，
而不是拖一根不知道结果的滑杆。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quality:
    id: str
    label: str
    grid: int          # 最长边采样数
    refine: int        # 牛顿投影迭代
    smooth: int        # Taubin 平滑迭代
    note: str

    @property
    def approx_faces(self) -> int:
        """经验估算：封闭曲面的面数约与 grid² 成正比（球体 5.1、一般形体 2~4，取 3.6）。"""
        return int(round(3.6 * self.grid * self.grid))

    def face_text(self) -> str:
        n = self.approx_faces
        if n >= 10000:
            return f"约 {n / 10000:.0f} 万面"
        return f"约 {n} 面"


QUALITIES: tuple[Quality, ...] = (
    Quality("draft", "草稿", 56, 1, 1, "秒出，用来看大形"),
    Quality("standard", "标准", 88, 2, 1, "日常预览与打印的平衡点"),
    Quality("fine", "精细", 128, 2, 2, "细节清楚，适合成品"),
    Quality("ultra", "超精细", 176, 3, 2, "手办级，导出前用"),
    Quality("print", "打印级", 240, 3, 2, "最细腻，导出会慢一些"),
)

QUALITY_MAP: dict[str, Quality] = {q.id: q for q in QUALITIES}
QUALITY_IDS: tuple[str, ...] = tuple(q.id for q in QUALITIES)
DEFAULT_QUALITY = "fine"

# 交互预览用的档位：够看清形状，又要跟手
PREVIEW = Quality("preview", "预览", 60, 1, 1, "交互预览")


def get(quality_id: str) -> Quality:
    return QUALITY_MAP.get(quality_id) or QUALITY_MAP[DEFAULT_QUALITY]


def index_of(quality_id: str) -> int:
    return QUALITY_IDS.index(get(quality_id).id)


def at(index: int) -> Quality:
    return QUALITIES[max(0, min(len(QUALITIES) - 1, int(index)))]


def upgrade(quality_id: str) -> Quality:
    return at(index_of(quality_id) + 1)


def downgrade(quality_id: str) -> Quality:
    return at(index_of(quality_id) - 1)


def face_budget(quality_id: str) -> int:
    """导出前用来提示「这个精度大概多大」的面数上限。"""
    return get(quality_id).approx_faces


def label(quality_id: str) -> str:
    return get(quality_id).label
