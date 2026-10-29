"""nlp.py —— 本地文本理解：把一句话变成「形态 + 参数 + 配色 + 种子」。

完全离线、零依赖的规则引擎：
1. 先匹配形态关键词（中英文都支持，长词优先，避免「花瓶」被「瓶」抢先）；
2. 再识别配色词、风格词、数量词（细节 / 种子）；
3. 最后把风格词折算成参数微调，并给出「我读到了什么」的可解释说明。

之所以不用大模型：这个工具承诺离线可用；而且可解释性比「黑箱生成」更适合
调参场景——用户能确切看到某个词改变了哪个旋钮。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import concepts, quality, registry

# 配色关键词（含中英文）
_PALETTE_WORDS: dict[str, tuple[str, ...]] = {
    "frost": ("冷", "冰", "霜", "雪", "蓝", "天青", "cold", "ice", "frost", "blue", "cyan"),
    "magma": ("暖", "火", "熔岩", "岩浆", "橙红", "赤", "magma", "lava", "fire", "warm", "red"),
    "moss": ("苔", "森", "绿", "植物", "草", "moss", "forest", "green", "plant"),
    "lunar": ("月", "白", "银灰", "素", "骨白", "lunar", "moon", "white", "ivory"),
    "amethyst": ("紫", "水晶", "魔法", "amethyst", "purple", "violet"),
    "amber": ("琥珀", "金", "黄", "木", "蜂蜜", "amber", "gold", "yellow", "wood", "honey"),
    "steel": ("钢", "铁", "金属", "机械", "赛博", "工业", "steel", "iron", "metal", "cyber", "industrial"),
    "coral": ("珊瑚", "粉", "玫瑰", "桃", "coral", "pink", "rose", "peach"),
}

# 形态关键词带来的「附加风格倾向」
_FORM_STYLE_HINT = {
    "vase": ("器皿", "handmade"),
    "crystal": ("矿物", "faceted"),
    "knot": ("数学", "topological"),
    "tower": ("建筑", "architectural"),
    "terrain": ("地貌", "landscape"),
    "shell": ("生物", "biological"),
    "relief": ("图像", "image"),
    "profile": ("剪影", "silhouette"),
}

# (关键词组, 作用描述, 参数增量)
_STYLE_RULES: tuple[tuple[tuple[str, ...], str, dict[str, float], str], ...] = (
    (("尖锐", "锋利", "硬朗", "棱角", "切割", "sharp", "spiky", "angular"), "棱角更锐利", {"b": 18}, ""),
    (("柔和", "圆润", "柔软", "光滑", "流线", "soft", "smooth", "round"), "线条更柔和", {"b": -12}, ""),
    (("扭曲", "缠绕", "拧", "扭转", "twisted", "twist"), "更强的扭曲", {"a": 16}, ""),
    (("高", "修长", "瘦", "细长", "tall", "slender"), "更修长的比例", {"a": 8}, ""),
    (("矮", "敦实", "厚重", "胖", "short", "stout"), "更敦实的比例", {"a": -8}, ""),
    (("极简", "简单", "克制", "干净", "minimal", "simple", "clean"), "更克制的细节", {"b": -10}, "res-14"),
    (("高精度", "高细节", "拉满", "精细", "超清", "高模", "high detail", "ultra"), "更高的细节密度", {"b": 8}, "res+16"),
    (("低模", "低精度", "粗略", "lowpoly", "low poly", "draft"), "更低的细节密度", {"b": -6}, "res-14"),
    (("繁复", "复杂", "密集", "精雕", "intricate", "complex", "detailed"), "更密集的细节", {"b": 10}, "res+16"),
    (("对称", "symmetric", "symmetry"), "保持对称", {}, ""),
    (("随机", "惊喜", "随便", "random", "surprise"), "随机种子", {}, "random"),
)

# 概念模式的风格词：直接改「形体旋钮」而不是某个形态的参数
_STYLE_FLAGS: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (("圆润", "柔和", "光滑", "可爱", "卡通", "胖乎乎", "soft", "cute", "smooth"), "organic", 0.28),
    (("尖锐", "锋利", "棱面", "棱角", "宝石", "切割", "sharp", "faceted", "gem"), "sharp", 0.34),
    (("粗糙", "风化", "石头", "岩石", "做旧", "斑驳", "rough", "stone", "weathered"), "rough", 0.32),
    (("扭曲", "缠绕", "拧", "螺旋", "twisted", "twist", "spiral"), "twist", 0.30),
    (("繁复", "复杂", "密集", "精雕", "细节", "intricate", "detailed"), "detail", 0.22),
    (("极简", "简单", "干净", "minimal", "simple", "clean"), "detail", -0.24),
)

_HOLLOW_WORDS = ("中空", "空心", "镂空", "薄壁", "hollow", "thin wall")

_QUALITY_WORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("草稿", "快速", "看看大形", "draft"), "draft"),
    (("标准", "普通", "standard", "normal"), "standard"),
    (("精细", "高清", "高精度", "高细节", "fine", "good", "high detail"), "fine"),
    (("超精细", "超高精度", "超清", "手办级", "ultra"), "ultra"),
    (("打印级", "最高", "极致", "print", "max"), "print"),
)

# 分隔符要包含 "="：很多人会照着命令行习惯写 seed=42
_NUM_SEED = re.compile(r"(?:种子|seed|编号)\s*[:：#=]?\s*(\d{1,5})", re.I)
_NUM_RES = re.compile(r"(?:细节|密度|精度|面数|分辨率|detail|quality)\s*[:：=]?\s*(\d{1,3})", re.I)
# 中文数字：「细节八十」「密度 三十五」
_NUM_RES_CN = re.compile(
    r"(?:细节|密度|精度|面数|分辨率|detail|quality)\s*[:：=]?\s*"
    r"([零一二三四五六七八九十百]+)"
)
_STANDALONE_NUM = re.compile(r"(?<![\d.])(\d{3,4})(?![\d.])")
# 尺寸：高 120mm / 8 厘米 / size=90
_NUM_SIZE_MM = re.compile(
    r"(?:高|长|宽|尺寸|大小|高度|直径|size|height)\s*[:：=]?\s*(\d{1,4}(?:\.\d+)?)\s*(?:mm|毫米|㎜)", re.I
)
_NUM_SIZE_CM = re.compile(
    r"(?:高|长|宽|尺寸|大小|高度|直径|size|height)\s*[:：=]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:cm|厘米|㎝)", re.I
)
# 裸写尺寸：8厘米 / 90mm（前面没有「高/尺寸」这类词）
_BARE_SIZE = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d+)?)\s*(mm|毫米|㎜|cm|厘米|㎝)", re.I)
_CN_NUM = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(token: str) -> int | None:
    """把「八」「十」「八十」「三十五」「一百」这类中文数字转成整数（0–999）。"""
    token = token.strip()
    if not token or not set(token) <= set("零一二三四五六七八九十百"):
        return None
    if "百" in token:
        head, _, tail = token.partition("百")
        hundreds = _cn_to_int(head) if head else 1
        rest = _cn_to_int(tail) if tail else 0
        if hundreds is None or rest is None:
            return None
        return hundreds * 100 + rest
    if "十" in token:
        head, _, tail = token.partition("十")
        tens = _cn_to_int(head) if head else 1
        ones = _cn_to_int(tail) if tail else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    total = 0
    for ch in token:
        total = total * 10 + _CN_NUM.get(ch, 0)
    return total


@dataclass
class PromptResult:
    text: str = ""
    form_id: str | None = None
    concept_id: str | None = None
    seed: int | None = None
    res: float | None = None
    quality: str | None = None
    size_mm: float | None = None
    palette: str | None = None
    hollow: bool | None = None
    style: dict[str, float] = field(default_factory=dict)
    param_delta: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    random_seed: bool = False

    @property
    def understood(self) -> bool:
        return bool(
            self.form_id or self.concept_id or self.palette or self.param_delta
            or self.seed is not None or self.style or self.quality or self.size_mm
            or self.random_seed
        )

    def summary(self) -> str:
        if not self.understood:
            return "没有识别到可用的关键词，已把这段话写进创意卡的「灵感来源」。"
        return " · ".join(self.notes) if self.notes else "已解析"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.lower().replace("，", ",").replace("。", ".").strip()


def _find_form(text: str) -> tuple[str | None, str | None]:
    best: tuple[int, str, str] | None = None
    for kw, form_id in registry.keyword_index():
        pos = text.find(kw)
        if pos < 0:
            continue
        if best is None or len(kw) > best[0]:
            best = (len(kw), form_id, kw)
    if best:
        return best[1], best[2]
    return None, None


def _find_palette(text: str) -> tuple[str | None, str | None]:
    best: tuple[int, str, str] | None = None
    for palette_id, words in _PALETTE_WORDS.items():
        for kw in words:
            pos = text.find(kw)
            if pos < 0:
                continue
            if best is None or len(kw) > best[0]:
                best = (len(kw), palette_id, kw)
    if best:
        return best[1], best[2]
    return None, None


def parse(text: str) -> PromptResult:
    raw = (text or "").strip()
    norm = _normalize(raw)
    result = PromptResult(text=raw)
    if not norm:
        return result

    # 1) 概念模型（能生成任意物体的那套配方）优先
    concept_id, concept_kw = concepts.match(norm)
    if concept_id:
        result.concept_id = concept_id
        result.matched.append(concept_kw or "")
        result.notes.append(f"模型「{concepts.get(concept_id).label}」")

    # 2) 经典形态（保留旧入口，图片类形态还要靠它）
    form_id, form_kw = _find_form(norm)
    if form_id:
        result.form_id = form_id
        if not concept_id:
            result.matched.append(form_kw or "")
            result.notes.append(f"形态「{registry.get(form_id).label}」")
        result.concept_id = result.concept_id or concept_for_form(form_id)

    palette_id, palette_kw = _find_palette(norm)
    if palette_id:
        result.palette = palette_id
        result.matched.append(palette_kw or "")
        result.notes.append(f"配色「{_palette_label(palette_id)}」")

    # 尺寸（毫米优先，其次厘米，最后允许裸写「8 厘米」「90mm」）
    size_span: tuple[int, int] | None = None
    m = _NUM_SIZE_MM.search(norm)
    if m:
        result.size_mm = float(max(20.0, min(400.0, float(m.group(1)))))
        size_span = m.span(1)
    else:
        m = _NUM_SIZE_CM.search(norm)
        if m:
            result.size_mm = float(max(20.0, min(400.0, float(m.group(1)) * 10.0)))
            size_span = m.span(1)
        else:
            m = _BARE_SIZE.search(norm)
            if m:
                value = float(m.group(1))
                result.size_mm = float(max(20.0, min(400.0, value * 10.0 if m.group(2)[0] in "cC厘" else value)))
                size_span = m.span(1)
    if result.size_mm is not None:
        result.notes.append(f"尺寸 {result.size_mm:.0f}mm")

    m = _NUM_SEED.search(norm)
    if m:
        result.seed = int(m.group(1))
        result.notes.append(f"种子 {result.seed}")
    else:
        m2 = _STANDALONE_NUM.search(norm)
        if m2 and not (size_span and m2.start(1) >= size_span[0] and m2.end(1) <= size_span[1]):
            result.seed = int(m2.group(1))
            result.notes.append(f"种子 {result.seed}")

    m = _NUM_RES.search(norm)
    if m:
        result.res = float(max(0, min(100, int(m.group(1)))))
        result.notes.append(f"细节密度 {int(result.res)}")
    else:
        m_cn = _NUM_RES_CN.search(norm)
        if m_cn:
            value = _cn_to_int(m_cn.group(1))
            if value is not None:
                result.res = float(max(0, min(100, value)))
                result.notes.append(f"细节密度 {int(result.res)}")

    # 精度档位
    for words, qid in _QUALITY_WORDS:
        hit = next((w for w in words if w in norm), None)
        if hit:
            result.quality = qid
            result.matched.append(hit)
            result.notes.append(f"精度「{quality.get(qid).label}」")
            break

    # 中空
    if any(w in norm for w in _HOLLOW_WORDS):
        result.hollow = True
        result.notes.append("中空壁厚")

    # 风格旋钮（概念模式用）
    for words, key, amount in _STYLE_FLAGS:
        hit = next((w for w in words if w in norm), None)
        if not hit:
            continue
        result.matched.append(hit)
        result.style[key] = result.style.get(key, 0.0) + amount
        label = {"organic": "更圆润", "sharp": "更硬朗", "rough": "更粗糙",
                 "twist": "更扭曲", "detail": "细节更强"}.get(key, key)
        result.notes.append(label)

    for words, desc, delta, extra in _STYLE_RULES:
        hit = next((w for w in words if w in norm), None)
        if not hit:
            continue
        result.matched.append(hit)
        if desc not in result.notes:
            result.notes.append(desc)
        for key, value in delta.items():
            result.param_delta[key] = result.param_delta.get(key, 0.0) + value
        # 用户写了「细节 80」就不要再被风格词悄悄改成 96：显式数字优先
        if extra == "res+16" and result.res is None:
            result.res = min(100.0, 62.0 + 16)
        elif extra == "res-14" and result.res is None:
            result.res = max(6.0, 62.0 - 14)
        elif extra == "random":
            result.random_seed = True

    return result


def concept_for_form(form_id: str) -> str | None:
    """经典形态 → 概念模型 id（概念库里同名的直接对应）。"""
    alias = {"shell": "seashell"}
    cid = alias.get(form_id, form_id)
    return cid if cid in concepts.CONCEPT_MAP else None


def _palette_label(palette_id: str) -> str:
    from . import config

    pal = config.PALETTES.get(palette_id)
    return pal.label if pal else palette_id


def apply_to(form_id: str, params: dict[str, float], result: PromptResult) -> dict[str, float]:
    """把解析出的增量作用到当前参数上，并夹紧到合法范围。"""
    merged = dict(params)
    for key, delta in result.param_delta.items():
        merged[key] = merged.get(key, 50.0) + delta
    return registry.clamp_params(form_id, merged)


def keywords_help() -> str:
    """给界面用的关键词提示。"""
    forms = "、".join(c.label for c in concepts.CONCEPTS[:10])
    return (
        "试着这样写：\n"
        f"· 想要什么：{forms}…… 共 {len(concepts.CONCEPTS)} 个模型，写别的词也行\n"
        "· 风格词：圆润 / 尖锐 / 粗糙 / 扭曲 / 极简 / 繁复 / 中空\n"
        "· 尺寸精度：高 120mm、打印级、精细\n"
        "· 配色词：冰蓝 / 熔岩 / 苔绿 / 月白 / 紫晶 / 琥珀 / 钢铁 / 珊瑚\n"
        "· 数字：种子 2026"
    )
