"""registry.py —— 形态登记表：全应用的唯一数据源。

界面上的形态按钮、参数滑杆名称与范围、提示词解析、随机灵感、预设、
导出文件名、自检脚本，全部从这里读取，避免「界面改了、解析没改」的错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

RES_SPEC = {
    "label": "细节密度",
    "hint": "越高越细腻、面数越多、导出文件越大；预览会自动降精度以保证流畅。",
    "lo": 0.0,
    "hi": 100.0,
    "default": 62.0,
}


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    hint: str
    lo: float = 0.0
    hi: float = 100.0
    default: float = 50.0
    step: float = 1.0
    suffix: str = ""


@dataclass(frozen=True)
class Preset:
    name: str
    a: float
    b: float
    note: str = ""


@dataclass(frozen=True)
class FormSpec:
    id: str
    label: str
    glyph: str
    desc: str
    detail: str
    params: tuple[ParamSpec, ParamSpec]
    needs_image: bool = False
    keywords: tuple[str, ...] = ()
    tips: str = ""
    default_palette: str = "frost"
    presets: tuple[Preset, ...] = ()

    def param(self, key: str) -> ParamSpec:
        return self.params[0] if key == "a" else self.params[1]


FORMS: tuple[FormSpec, ...] = (
    FormSpec(
        id="vase",
        label="波纹花瓶",
        glyph="◍",
        desc="旋转体 · 有机轮廓",
        detail="一条有机轮廓线绕轴旋转成中空器皿，波纹与扭曲带来手工陶艺的呼吸感。",
        params=(
            ParamSpec("a", "扭曲程度", "轮廓绕轴旋转的扭转角，越大越像被拧过的陶坯。", default=42),
            ParamSpec("b", "波纹强度", "器壁径向波纹的幅度与瓣数，越大越起伏。", default=38),
        ),
        keywords=("花瓶", "瓶", "器皿", "容器", "陶", "罐", "vase", "vessel", "pot", "bottle", "urn"),
        tips="中空壁厚约 1.8mm，可直接打印；想插花可把瓶口内径再放大。",
        default_palette="frost",
        presets=(
            Preset("素坯", 12, 18, "近乎对称的极简陶器"),
            Preset("海流", 58, 62, "明显扭曲 + 密集波纹"),
            Preset("沙丘", 30, 88, "强波纹，像被风堆出的沙层"),
        ),
    ),
    FormSpec(
        id="crystal",
        label="晶簇尖晶",
        glyph="◆",
        desc="细分二十面体 · 脊状位移",
        detail="用脊状噪声挤压细分球面，再量化半径形成晶面，介于矿物与建筑之间。",
        params=(
            ParamSpec("a", "生长强度", "晶体外凸的幅度，越大越尖锐。", default=58),
            ParamSpec("b", "棱角锐度", "半径量化档位，越大晶面越硬、越像切割宝石。", default=46),
        ),
        keywords=("晶体", "晶簇", "宝石", "矿石", "尖晶", "钻石", "crystal", "gem", "mineral", "diamond", "rock"),
        tips="尖锐处建议 0.1mm 层高；树脂打印后喷哑光透明漆会很像矿石。",
        default_palette="amethyst",
        presets=(
            Preset("原石", 24, 18, "柔和起伏，像未切割的矿石"),
            Preset("切割", 62, 78, "强量化，明显晶面"),
            Preset("尖峰", 92, 40, "细长晶柱群"),
        ),
    ),
    FormSpec(
        id="knot",
        label="环面纽结",
        glyph="∞",
        desc="数学纽结 · 管状扫掠",
        detail="(p, q) 环面纽结沿曲线扫掠圆管，p / q 互质保证它真的是一个结而不是几个圈。",
        params=(
            ParamSpec("a", "绕轴数 p", "绕大圆缠绕的圈数。", default=40),
            ParamSpec("b", "绕管数 q", "绕小圆缠绕的圈数，与 p 自动保持互质。", default=46),
        ),
        keywords=("纽结", "结", "环", "拓扑", "缠绕", "knot", "torus", "loop", "topology"),
        tips="管径较细，建议加底座或放大到 120mm 打印，避免共振晃动。",
        default_palette="lunar",
        presets=(
            Preset("三叶结", 8, 22, "经典 (2,3) 三叶结"),
            Preset("密绕", 72, 58, "高密度缠绕"),
            Preset("大圈", 22, 88, "宽松的大环"),
        ),
    ),
    FormSpec(
        id="tower",
        label="螺旋塔",
        glyph="▲",
        desc="分层放样 · 扭转塔身",
        detail="多边形截面逐层收缩并扭转，形成有建构感的塔体。",
        params=(
            ParamSpec("a", "扭转角度", "塔身从底到顶累计扭转的圈数。", default=38),
            ParamSpec("b", "侧面数量", "多边形边数，3 为三棱塔，10 接近圆柱。", default=46),
        ),
        keywords=("螺旋塔", "塔", "楼", "尖塔", "柱", "建筑", "tower", "spire", "column", "pagoda", "building"),
        tips="三棱塔打印时侧面悬空较大，建议开启支撑或改用 5 边以上。",
        default_palette="steel",
        presets=(
            Preset("方尖碑", 12, 22, "四棱直塔"),
            Preset("螺旋塔", 52, 46, "明显扭转"),
            Preset("圆锥塔", 26, 96, "接近圆锥"),
        ),
    ),
    FormSpec(
        id="terrain",
        label="起伏地块",
        glyph="▤",
        desc="分形高度场 · 封闭地块",
        detail="fBm 高度场压成一块可打印的地形切片，像从某个星球上切下来的一角。",
        params=(
            ParamSpec("a", "起伏幅度", "山峰与谷底的高差。", default=52),
            ParamSpec("b", "地形频率", "地形特征的疏密，越大山脊越细碎。", default=44),
        ),
        keywords=("地形", "地块", "山", "等高线", "地貌", "荒野", "terrain", "landscape", "mountain", "map"),
        tips="平放打印最省支撑；想做壁挂浮雕请把厚度方向朝外。",
        default_palette="moss",
        presets=(
            Preset("丘陵", 30, 22, "舒缓的丘陵"),
            Preset("山脉", 72, 52, "高耸山脊"),
            Preset("群岛", 44, 82, "细碎岛群"),
        ),
    ),
    FormSpec(
        id="shell",
        label="螺旋海螺",
        glyph="◎",
        desc="指数螺旋 · 渐扩管体",
        detail="沿对数螺线扫掠、管径指数增长的天然壳体，是最容易打印出「生物感」的形态。",
        params=(
            ParamSpec("a", "生长速率", "螺线张开的快慢，越大越张扬。", default=54),
            ParamSpec("b", "盘旋圈数", "海螺的圈数，越多越紧致。", default=40),
        ),
        keywords=("海螺", "螺", "贝壳", "螺旋", "nautilus", "shell", "snail", "spiral", "conch"),
        tips="尖端起翘处需要支撑，也可竖立打印让螺旋自身支撑。",
        default_palette="amber",
        presets=(
            Preset("菊石", 36, 62, "紧致多圈"),
            Preset("鹦鹉螺", 58, 40, "经典比例"),
            Preset("unicum", 84, 26, "张开的异形螺"),
        ),
    ),
    FormSpec(
        id="relief",
        label="图像浮雕",
        glyph="▦",
        desc="图片亮度 → 高度场",
        detail="把图片的明暗直接翻译成高度：亮处凸起、暗处凹陷，做成可打印的浮雕牌。",
        params=(
            ParamSpec("a", "浮雕深度", "最高点与最低点的高差。", default=55),
            ParamSpec("b", "对比曲线", "亮度曲线指数，越大中间调越平、明暗对比越硬。", default=45),
        ),
        needs_image=True,
        keywords=("浮雕", "图片", "照片", "logo", "印章", "relief", "photo", "image", "emboss", "lithophane"),
        tips="建议用主体清晰、背景干净的图片；深度别超过 4mm，否则容易拉丝。",
        default_palette="lunar",
        presets=(
            Preset("浅浮雕", 28, 40, "细节柔和"),
            Preset("深浮雕", 72, 55, "立体感强"),
            Preset("硬边", 48, 82, "高对比，像剪纸"),
        ),
    ),
    FormSpec(
        id="profile",
        label="轮廓旋转体",
        glyph="◐",
        desc="图片轮廓 → 旋转体",
        detail="提取图片主体的外轮廓，绕轴旋转成器皿——把一张剪影直接变成可打印的瓶子。",
        params=(
            ParamSpec("a", "外扩比例", "轮廓整体放大/收窄的程度。", default=50),
            ParamSpec("b", "轮廓平滑", "轮廓的平滑强度，越大越接近几何形体。", default=55),
        ),
        needs_image=True,
        keywords=("轮廓", "剪影", "形状", "silhouette", "profile", "outline", "shape"),
        tips="主体与背景对比越强、越居中，提取的轮廓越准确。",
        default_palette="coral",
        presets=(
            Preset("顺形", 50, 30, "尽量贴合原图轮廓"),
            Preset("几何", 50, 88, "强力平滑，接近几何体"),
            Preset("丰盈", 78, 55, "轮廓整体外扩"),
        ),
    ),
)

FORM_MAP: dict[str, FormSpec] = {f.id: f for f in FORMS}
FORM_IDS: tuple[str, ...] = tuple(f.id for f in FORMS)


def get(form_id: str) -> FormSpec:
    return FORM_MAP.get(form_id) or FORMS[0]


def params_by_key(form_id: str) -> dict[str, ParamSpec]:
    spec = get(form_id)
    return {p.key: p for p in spec.params}


def default_params(form_id: str) -> dict[str, float]:
    spec = get(form_id)
    return {p.key: p.default for p in spec.params}


def random_params(form_id: str, rng) -> dict[str, float]:
    """随机灵感用：在保守区间内取值，避免每次随机都得到极端造型。"""
    spec = get(form_id)
    out: dict[str, float] = {}
    for p in spec.params:
        span = p.hi - p.lo
        out[p.key] = round(rng.uniform(p.lo + span * 0.12, p.hi - span * 0.10), 1)
    return out


def clamp_params(form_id: str, values: dict[str, float]) -> dict[str, float]:
    spec = get(form_id)
    out = {}
    for p in spec.params:
        raw = values.get(p.key, p.default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = p.default
        if v != v or v in (float("inf"), float("-inf")):
            v = p.default
        out[p.key] = max(p.lo, min(p.hi, v))
    return out


def param_label(form_id: str, key: str) -> str:
    return get(form_id).param(key).label


def keyword_index() -> list[tuple[str, str]]:
    """(关键词, 形态 id)，长词优先匹配。"""
    pairs: list[tuple[str, str]] = []
    for f in FORMS:
        for kw in f.keywords:
            pairs.append((kw.lower(), f.id))
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    return pairs
