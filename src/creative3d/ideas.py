"""ideas.py —— 创意文案生成器。

由 (形态, 种子, 参数, 配色) 确定性派生命名 / 故事 / 材质 / 用途 / 打印提示，
相同输入永远得到相同文案，方便复现与归档。v2 增加了：
* 与配色联动的描述；
* 文本灵感、图片来源会作为「灵感来源」写进创意卡；
* 标签系统更细，便于检索。
"""

from __future__ import annotations

import random
import zlib
from dataclasses import asdict, dataclass, field

from . import concepts, registry

_TITLES_A = (
    "潮汐回响", "荒原生灵", "深海化石", "赛博祭坛", "苔原遗物", "极光褶皱",
    "月壤标本", "熔岩肌理", "静默几何", "风蚀记忆", "盐晶祷文", "星尘残片",
    "铁锈黎明", "玻璃内脏", "深空孢子", "海沟低语", "折光残卷", "灰烬纹章",
)
_TITLES_B = {
    "vase": ("之器", "水语容器", "呼吸瓶", "液态纪念碑", "潮汐罐", "空腔手稿"),
    "crystal": ("晶核", "碎光矿脉", "凝固闪电", "棱镜种子", "盐之骨", "切割星"),
    "knot": ("无解之结", "缠绕回路", "永恒纽结", "时间线圈", "拓扑护符", "自缚之环"),
    "tower": ("向上之城", "螺旋尖塔", "生长塔", "扭转碑", "层积之塔", "未完成的塔"),
    "terrain": ("微缩地貌", "掌心地块", "折叠高原", "方寸荒野", "沉积切片", "等高线残片"),
    "shell": ("螺旋腔体", "海之声", "钙质回廊", "卷曲居所", "鹦鹉螺残响", "生长的漏斗"),
    "relief": ("明暗拓片", "光影化石", "压印的记忆", "亮度剖面", "二维之骨", "像素浮雕"),
    "profile": ("剪影实体", "轮廓转生", "侧影器皿", "边界之内", "二手形状", "轮廓的容器"),
}
_MATERIALS = (
    "哑光白树脂 + 珠光喷涂",
    "PLA 哑光米白 + 丙烯做旧",
    "光敏树脂 + 局部金属箔",
    "木质 PLA + 蜂蜡打磨",
    "半透明 PETG + 内嵌暖光灯带",
    "青铜色金属 PLA + 干刷",
    "半透明白树脂 + 内部渐变染色",
    "回收 PLA 灰 + 石墨粉填缝",
)
_USAGES = (
    "桌面摆件", "灯具罩体", "首饰托架", "香薰扩香器", "建筑概念模型",
    "花器 / 笔筒", "纪念奖杯", "灯箱浮雕", "香座", "书桌压纸",
)
_TIPS = (
    "建议层高 0.16mm、填充 15%、壁厚 ≥1.2mm；竖直摆放可显著减少支撑。",
    "建议加 3mm 实心底座提升稳定性；悬空角 <45° 时可免支撑打印。",
    "薄壁处建议 0.12mm 层高；树脂打印后二次固化 2 分钟，避免变形。",
    "建议整体缩放至 100mm 内；大尺寸可拆分为 2 段，用 Ø4mm 圆柱销对接。",
    "先用 0.28mm 草稿层高试打一次，确认比例后再上细层高。",
)
_SIMILES = (
    "被海浪打磨过的骨头", "一枚正在冷却的星体", "潮间带遗落的钙质壳",
    "老厂房里锈穿的齿轮", "冰川内部的气泡层", "某种从未被记录的孢子",
    "一颗被风掏空的果实", "被反复折叠又展开的纸", "深海热泉口的沉积物",
)
_METAPHORS = (
    "一座会呼吸的塔", "被折叠的等高线", "一条自己咬住尾巴的河",
    "凝固在半空的水花", "一封写给重力的信", "一段被拉长的沉默",
    "被时间咬了一口的形状", "一具正在结晶的骨架",
)
_TEXTURES = (
    "半透明的冷光", "颗粒感的哑光", "湿润的金属反射", "干燥的粉质表面",
    "油亮的釉面", "粗糙的岩层断面", "被摩挲过的温润光泽",
)
_LIGHTS = ("低角度侧光", "清晨的散射光", "单点聚光", "背光轮廓", "黄昏的斜射光")
_TAGS = (
    "有机曲线", "参数化", "单件成型", "免装配", "可打印", "对称破缺",
    "分层肌理", "拓扑连续", "负空间", "轻量化结构", "单色雕塑", "可缩放",
)
_PALETTE_PHRASES = {
    "frost": "像被冷水冲过的蓝",
    "magma": "从炉膛里取出的暗红",
    "moss": "雨后苔藓的湿润绿",
    "lunar": "月面尘埃般的灰白",
    "amethyst": "有雾的紫水晶",
    "amber": "树脂凝固时的暖橙",
    "steel": "机床表面的冷灰",
    "coral": "退潮后珊瑚的粉",
}


@dataclass
class Idea:
    title: str
    story: str
    material: str
    usage: str
    tip: str
    tags: list[str]
    palette_label: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def generate_idea(
    form_id: str,
    seed: int,
    *,
    res: int = 60,
    params: dict | None = None,
    palette_id: str = "frost",
    palette_label: str = "",
    stats: str = "",
    source: str = "",
    image_notes: list[str] | None = None,
    concept_id: str | None = None,
    size_mm: float = 0.0,
    quality_label: str = "",
) -> Idea:
    concept = concepts.get(concept_id) if concept_id else None
    spec = registry.get(form_id)
    label = concept.label if concept else spec.label
    detail = concept.note if concept else spec.detail
    tips = (concept.hint if concept and concept.hint else spec.tips)
    # 用 crc32 而不是内置 hash：后者在每个进程里随机化，会破坏「同种子同文案」
    key = concept_id or form_id
    rng = random.Random((int(seed) * 7919) ^ (zlib.crc32(key.encode("utf-8")) & 0xFFFF))

    head = rng.choice(_TITLES_A)
    if concept:
        title = f"{head} · {label}"
    else:
        title = head + rng.choice(_TITLES_B.get(form_id, ("之作",)))

    simile = rng.choice(_SIMILES)
    metaphor = rng.choice(_METAPHORS)
    texture = rng.choice(_TEXTURES)
    light = rng.choice(_LIGHTS)
    palette_phrase = _PALETTE_PHRASES.get(palette_id, "一层薄薄的色雾")

    story = (
        f"它像{simile}，又像{metaphor}。在{light}下显出{texture}，"
        f"整体是{palette_phrase}。{detail}"
    )
    if source:
        story += f" 这一件的起点是：{source}。"

    tags = rng.sample(_TAGS, 3)
    tags.append(label)
    if image_notes:
        tags.extend(n for n in image_notes[:2] if n)

    return Idea(
        title=title,
        story=story,
        material=rng.choice(_MATERIALS),
        usage=rng.choice(_USAGES),
        tip=tips or rng.choice(_TIPS),
        tags=tags,
        palette_label=palette_label,
        source=source,
        extra={
            "stats": stats, "res": res, "seed": int(seed), "params": params or {},
            "concept": concept_id or "", "size_mm": size_mm, "quality": quality_label,
        },
    )


def idea_card_text(
    idea: Idea,
    *,
    form_id: str,
    seed: int,
    res: int,
    params: dict,
    palette_id: str,
    image_info: dict | None = None,
    concept_id: str | None = None,
    size_mm: float = 0.0,
    quality_label: str = "",
    report=None,
) -> str:
    """导出为可归档的 Markdown 创意卡。"""
    concept = concepts.get(concept_id) if concept_id else None
    spec = registry.get(form_id)
    head_a = registry.param_label(form_id, "a")
    head_b = registry.param_label(form_id, "b")
    if concept:
        head_a, head_b = concept.pa[0], concept.pb[0]

    lines = [
        f"# {idea.title}",
        "",
        f"> {idea.story}",
        "",
        "## 设计参数",
        "",
    ]
    if concept:
        lines.append(f"- 模型：{concept.label}（{concept.category} · {concept.note}）")
    else:
        lines.append(f"- 形态族：{spec.label}（{spec.desc}）")
    lines.append(f"- 随机种子：{seed}")
    if size_mm:
        lines.append(f"- 目标尺寸：最长边 {size_mm:.0f} mm")
    if quality_label:
        lines.append(f"- 精度档位：{quality_label}")
    lines.append(f"- 细节密度：{int(res)} / 100")
    lines.append(f"- {head_a}：{params.get('a', 0):.0f} / 100")
    lines.append(f"- {head_b}：{params.get('b', 0):.0f} / 100")
    lines.append(f"- 配色：{idea.palette_label or palette_id}")
    if idea.source:
        lines.append(f"- 灵感来源：{idea.source}")
    if image_info:
        name = image_info.get("name")
        if name:
            lines.append(f"- 参考图片：{name}")
        colors = image_info.get("colors") or []
        if colors:
            lines.append("- 提取色板：" + " / ".join(colors))
    if idea.extra.get("stats"):
        lines.append(f"- 网格规模：{idea.extra['stats']}")
    lines += [
        "",
        "## 落地建议",
        "",
        f"- 推荐用途：{idea.usage}",
        f"- 建议材质：{idea.material}",
        f"- 打印提示：{idea.tip}",
        f"- 关键词：{' / '.join(idea.tags)}",
    ]
    if report is not None:
        lines += ["", "## 打印体检", ""]
        lines += [f"- {line}" for line in report.to_lines()]
    lines += [
        "",
        "## 打印备注",
        "",
        "模型单位毫米、底面贴合 z=0；STL 可直接进切片软件，3MF 额外记录了单位与对象名。",
        "",
        "---",
        "",
        f"由「3D 建模创意生成器 v3」生成 · {concept_id or form_id} · seed {seed}",
        "",
    ]
    return "\n".join(lines)
