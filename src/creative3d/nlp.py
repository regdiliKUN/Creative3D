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

from . import registry

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

# 分隔符要包含 "="：很多人会照着命令行习惯写 seed=42
_NUM_SEED = re.compile(r"(?:种子|seed|编号)\s*[:：#=]?\s*(\d{1,5})", re.I)
_NUM_RES = re.compile(r"(?:细节|密度|精度|面数|分辨率|detail|quality)\s*[:：=]?\s*(\d{1,3})", re.I)
# 中文数字：「细节八十」「密度 三十五」
_NUM_RES_CN = re.compile(
    r"(?:细节|密度|精度|面数|分辨率|detail|quality)\s*[:：=]?\s*"
    r"([零一二三四五六七八九十百]+)"
)
_STANDALONE_NUM = re.compile(r"(?<![\d.])(\d{3,4})(?![\d.])")
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
    seed: int | None = None
    res: float | None = None
    palette: str | None = None
    param_delta: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    random_seed: bool = False

    @property
    def understood(self) -> bool:
        return bool(self.form_id or self.palette or self.param_delta or self.seed is not None)

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

    form_id, form_kw = _find_form(norm)
    if form_id:
        result.form_id = form_id
        result.matched.append(form_kw or "")
        result.notes.append(f"形态「{registry.get(form_id).label}」")

    palette_id, palette_kw = _find_palette(norm)
    if palette_id:
        result.palette = palette_id
        result.matched.append(palette_kw or "")
        result.notes.append(f"配色「{_palette_label(palette_id)}」")

    m = _NUM_SEED.search(norm)
    if m:
        result.seed = int(m.group(1))
        result.notes.append(f"种子 {result.seed}")
    else:
        m2 = _STANDALONE_NUM.search(norm)
        if m2:
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

    for words, desc, delta, extra in _STYLE_RULES:
        hit = next((w for w in words if w in norm), None)
        if not hit:
            continue
        result.matched.append(hit)
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
    forms = "、".join(f.keywords[0] for f in registry.FORMS)
    return (
        "试着这样写：\n"
        f"· 形态词：{forms}\n"
        "· 风格词：尖锐 / 柔和 / 扭曲 / 极简 / 繁复 / 修长 / 敦实\n"
        "· 配色词：冰蓝 / 熔岩 / 苔绿 / 月白 / 紫晶 / 琥珀 / 钢铁 / 珊瑚\n"
        "· 数字：种子 2026、细节 80"
    )
