"""session.py —— 单一状态源。

界面上的每个控件都只读写 Session，绝不自持状态：
    「控件 → Session → 事件 → 视口 / 创意卡 / 导出」只有一条数据流，
    因此不会出现「界面显示 30、导出用的是 50」这类错位。

v3 新增两条并行通道：
    concept 模式 —— 提示词/模型库驱动，SDF 建模，精度档位可选（主力路径）
    classic 模式 —— v2 的 8 个参数化形态（保留，图片浮雕/轮廓只能走这里）
"""

from __future__ import annotations

import random
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import concepts, config, generators, ideas, imaging, quality, registry, sdfmesh
from .generators import GenContext
from .mesh import Mesh

PREVIEW_RES_CAP = 44.0  # 经典形态交互时的最高精度
LOCAL_EXTENT = 1.12     # 局部单位下模型的典型最长边（用于毫米 ↔ 局部单位换算）

MODE_CONCEPT = "concept"
MODE_CLASSIC = "classic"


@dataclass
class Snapshot:
    mode: str
    concept: str
    form: str
    seed: int
    res: float
    a: float
    b: float
    palette: str
    quality: str
    size_mm: float
    hollow: bool
    wall_mm: float
    style: str
    image_path: str = ""
    image_mode: str = "silhouette"

    def key(self) -> str:
        return "|".join(
            [
                self.mode, self.concept, self.form, str(self.seed), f"{self.res:.2f}",
                f"{self.a:.2f}", f"{self.b:.2f}", self.palette, self.quality,
                f"{self.size_mm:.2f}", "1" if self.hollow else "0", f"{self.wall_mm:.2f}",
                self.style, self.image_path, self.image_mode,
            ]
        )


class Session:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.mode = settings.get("mode", MODE_CONCEPT)
        if self.mode not in (MODE_CONCEPT, MODE_CLASSIC):
            self.mode = MODE_CONCEPT
        self.concept = settings.get("concept", concepts.DEFAULT_CONCEPT)
        if self.concept not in concepts.CONCEPT_MAP:
            self.concept = concepts.DEFAULT_CONCEPT
        self.form = registry.get(settings.get("form", "vase")).id
        self.seed = int(settings.get("seed", 2026))
        self.res = float(settings.get("res", 62))
        self.params = registry.clamp_params(self.form, settings.get("params", {}) or {})
        self.palette_id = settings.get("palette", config.DEFAULT_PALETTE)
        self.quality_id = quality.get(settings.get("quality", quality.DEFAULT_QUALITY)).id
        self.size_mm = float(settings.get("size_mm", 100.0))
        self.wall_mm = float(settings.get("wall_mm", 1.8))
        hollow_pref = settings.get("hollow", None)
        self.hollow = (concepts.get(self.concept).hollow_default
                       if hollow_pref is None else bool(hollow_pref))
        self.style: dict[str, float] = dict(settings.get("style", {}) or {})
        self.image_path: str = settings.get("image_path", "") or ""
        self.image_mode: str = settings.get("image_mode", "silhouette")
        self.image: imaging.ImageFeatures | None = None
        self.image_error: str = ""

        self._listeners: dict[str, list[Callable[..., None]]] = {}
        self._mesh_cache: "OrderedDict[tuple, Mesh]" = OrderedDict()
        self._history: list[Snapshot] = []
        self._future: list[Snapshot] = []
        self._idea_source = ""
        self._prompt_text = ""
        self._image_palette: config.ModelPalette | None = None

        if self.image_path:
            self._load_image_silently(self.image_path)

    # ------------------------------------------------------------------ 事件
    def subscribe(self, topic: str, callback: Callable[..., None]) -> None:
        self._listeners.setdefault(topic, []).append(callback)

    def emit(self, topic: str, **payload: Any) -> None:
        for cb in self._listeners.get(topic, []):
            try:
                cb(**payload)
            except Exception:  # noqa: BLE001 - 单个订阅者出错不应影响其它
                import logging

                logging.getLogger("creative3d.session").exception("订阅者回调失败：%s", topic)

    # ------------------------------------------------------------------ 派生
    @property
    def spec(self) -> registry.FormSpec:
        return registry.get(self.form)

    @property
    def concept_spec(self) -> concepts.Concept:
        return concepts.get(self.concept)

    @property
    def quality(self) -> quality.Quality:
        return quality.get(self.quality_id)

    def is_concept_mode(self) -> bool:
        return self.mode == MODE_CONCEPT

    def title(self) -> str:
        if self.is_concept_mode():
            return self.concept_spec.label
        return self.spec.label

    def glyph(self) -> str:
        if self.is_concept_mode():
            return {"器物": "◍", "家具": "⌸", "建筑": "▲", "生物": "✿",
                    "机械": "⚙", "食物": "◕", "抽象": "◆"}.get(self.concept_spec.category, "◆")
        return self.spec.glyph

    def subtitle(self) -> str:
        if self.is_concept_mode():
            q = self.quality
            return (f"{self.concept_spec.category} · {self.concept_spec.note}"
                    f" · {q.label} {q.grid} · {self.size_mm:.0f}mm")
        params = self.params
        return (f"{self.spec.desc} · A {params.get('a', 0):.0f} / B {params.get('b', 0):.0f}"
                f" · 密度 {self.res:.0f}")

    def param_labels(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """(A 的名称, A 的提示), (B 的名称, B 的提示)"""
        if self.is_concept_mode():
            return self.concept_spec.pa, self.concept_spec.pb
        a, b = self.spec.param("a"), self.spec.param("b")
        return (a.label, a.hint), (b.label, b.hint)

    def snapshot(self) -> Snapshot:
        style = ",".join(f"{k}={v:.2f}" for k, v in sorted(self.style.items()) if abs(v) > 1e-6)
        return Snapshot(
            mode=self.mode,
            concept=self.concept,
            form=self.form,
            seed=self.seed,
            res=self.res,
            a=float(self.params.get("a", 50)),
            b=float(self.params.get("b", 50)),
            palette=self.palette_id,
            quality=self.quality_id,
            size_mm=self.size_mm,
            hollow=self.hollow,
            wall_mm=self.wall_mm,
            style=style,
            image_path=self.image_path,
            image_mode=self.image_mode,
        )

    def palette(self) -> config.ModelPalette:
        if self.palette_id == "image" and self._image_palette is not None:
            return self._image_palette
        return config.PALETTES.get(self.palette_id, config.PALETTES[config.DEFAULT_PALETTE])

    def palette_label(self) -> str:
        if self.palette_id == "image":
            return "取自图片" if self._image_palette else "默认配色"
        return self.palette().label

    def source_text(self) -> str:
        parts = []
        if self._prompt_text:
            parts.append(f"文本灵感「{self._prompt_text[:40]}」")
        if self.image is not None:
            parts.append(f"参考图片「{self.image.name}」")
        return " + ".join(parts)

    def image_info(self) -> dict | None:
        if self.image is None:
            return None
        return {
            "name": self.image.name,
            "colors": self.image.colors[:4],
            "summary": self.image.summary(),
            "path": self.image_path,
        }

    def idea(self, stats: str = "") -> ideas.Idea:
        return ideas.generate_idea(
            self.form,
            self.seed,
            res=int(self.res),
            params=dict(self.params),
            palette_id=self.palette_id,
            palette_label=self.palette_label(),
            stats=stats,
            source=self.source_text(),
            image_notes=self.image.notes if self.image else None,
            concept_id=self.concept if self.is_concept_mode() else None,
            size_mm=self.size_mm,
            quality_label=self.quality.label,
        )

    def favorite_key(self) -> str:
        return zlib.crc32(self.snapshot().key().encode("utf-8")).__str__()

    # ------------------------------------------------------------------ 建模参数
    def style_value(self, key: str, base: float = 0.0) -> float:
        return max(0.0, min(1.0, base + float(self.style.get(key, 0.0))))

    def build_params(self, grid: int | None = None) -> concepts.BuildParams:
        """把会话状态翻译成 SDF 建模参数。"""
        return concepts.make_params(
            self.seed,
            a=float(self.params.get("a", 50)) / 100.0,
            b=float(self.params.get("b", 50)) / 100.0,
            detail=self.style_value("detail", 0.5),
            organic=self.style_value("organic", 0.5),
            sharp=self.style_value("sharp"),
            rough=self.style_value("rough"),
            twist=self.style_value("twist"),
            hollow=self.hollow,
            wall=self.wall_local(grid),
            salt=self.concept,
        )

    def wall_local(self, grid: int | None = None) -> float:
        """毫米壁厚 → 局部单位；并按网格分辨率自动加厚到「能采样出来」的程度。

        壁厚只有 1.8mm 时，在 128 网格下只占 1.7 个体素 —— 等值面会把内外两层
        粘在一起，看起来像破边。与其生成一个破模型，不如在低精度下自动加厚，
        导出到打印级时再回到用户设定的真实壁厚。
        """
        scale = max(self.size_mm, 20.0) / LOCAL_EXTENT
        wall = max(0.006, min(0.08, self.wall_mm / scale))
        if grid:
            step = LOCAL_EXTENT * 1.35 / max(int(grid), 16)
            wall = max(wall, 2.6 * step)
        return min(wall, 0.16)

    # ------------------------------------------------------------------ 生成
    def _cache_key(self, q: quality.Quality, preview: bool) -> tuple:
        return (
            self.mode,
            self.concept,
            self.form,
            self.seed,
            q.id,
            round(float(self.params.get("a", 50)), 2),
            round(float(self.params.get("b", 50)), 2),
            round(self.size_mm, 1),
            self.hollow,
            round(self.wall_mm, 2),
            tuple(sorted((k, round(v, 2)) for k, v in self.style.items())),
            self.image_path,
            preview,
        )

    def generate_mesh(self, *, preview: bool = False) -> Mesh:
        q = quality.PREVIEW if preview else self.quality
        key = self._cache_key(q, preview)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            self._mesh_cache.move_to_end(key)
            return cached

        if self.is_concept_mode():
            bp = self.build_params(q.grid)
            fn = concepts.build(self.concept, bp)
            mesh = sdfmesh.mesh_from_sdf(fn, q)
            mesh.meta["concept"] = self.concept
            mesh.meta["hollow"] = self.hollow
        else:
            res = min(PREVIEW_RES_CAP, self.res) if preview else self.res
            ctx = GenContext(
                seed=self.seed,
                res=res,
                a=float(self.params.get("a", 50)),
                b=float(self.params.get("b", 50)),
                image=self.image if self.spec.needs_image or self.image is not None else None,
            )
            mesh = generators.build(self.form, ctx)

        mesh.normalize(self.size_mm)
        mesh.meta["preview"] = preview
        mesh.meta["form"] = self.form
        mesh.meta["quality"] = q.label
        self._mesh_cache[key] = mesh
        while len(self._mesh_cache) > 8:
            self._mesh_cache.popitem(last=False)
        return mesh

    def mesh_key(self) -> tuple:
        return self._cache_key(self.quality, False)

    # ------------------------------------------------------------------ 变更
    def _record(self) -> None:
        snap = self.snapshot()
        if self._history and self._history[-1].key() == snap.key():
            return
        self._history.append(snap)
        if len(self._history) > 60:
            self._history.pop(0)
        self._future.clear()

    def _changed(self, *, mesh: bool = True, idea: bool = True) -> None:
        self.emit("state", session=self)
        if idea:
            self.emit("idea", session=self)
        if mesh:
            self.emit("mesh", session=self)

    def set_mode(self, mode: str, *, record: bool = True) -> None:
        if mode not in (MODE_CONCEPT, MODE_CLASSIC) or mode == self.mode:
            return
        if record:
            self._record()
        self.mode = mode
        self._changed()

    def set_concept(self, concept_id: str, *, record: bool = True) -> None:
        if concept_id not in concepts.CONCEPT_MAP or concept_id == self.concept:
            if concept_id in concepts.CONCEPT_MAP:
                self.mode = MODE_CONCEPT
            return
        if record:
            self._record()
        self.concept = concept_id
        self.mode = MODE_CONCEPT
        self.params = registry.default_params(self.form)
        self.hollow = concepts.get(concept_id).hollow_default
        self.palette_id = concepts.palette_for(concept_id)
        self._changed()
        self.emit("palette", session=self)

    def set_form(self, form_id: str, *, record: bool = True) -> None:
        if form_id not in registry.FORM_MAP or form_id == self.form:
            return
        if record:
            self._record()
        self.form = form_id
        self.params = registry.default_params(form_id)
        self._changed()

    def set_param(self, key: str, value: float, *, record: bool = True) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        spec = self.spec.param(key)
        value = max(spec.lo, min(spec.hi, value))
        if abs(value - float(self.params.get(key, spec.default))) < 1e-6:
            return
        if record:
            self._record()
        self.params[key] = value
        self._changed()

    def set_seed(self, seed: int, *, record: bool = True) -> None:
        seed = int(max(0, min(999999, int(seed))))
        if seed == self.seed:
            return
        if record:
            self._record()
        self.seed = seed
        self._changed()

    def set_res(self, value: float, *, record: bool = True) -> None:
        value = max(0.0, min(100.0, float(value)))
        if abs(value - self.res) < 0.5:
            return
        if record:
            self._record()
        self.res = value
        self._changed()

    def set_quality(self, quality_id: str, *, record: bool = True) -> None:
        q = quality.get(quality_id)
        if q.id == self.quality_id:
            return
        if record:
            self._record()
        self.quality_id = q.id
        self._changed(mesh=False)
        self.emit("quality", session=self)

    def set_size(self, mm: float, *, record: bool = True) -> None:
        mm = max(20.0, min(400.0, float(mm)))
        if abs(mm - self.size_mm) < 0.5:
            return
        if record:
            self._record()
        self.size_mm = mm
        self._changed()

    def set_wall(self, mm: float, *, record: bool = True) -> None:
        mm = max(0.6, min(8.0, float(mm)))
        if abs(mm - self.wall_mm) < 0.05:
            return
        if record:
            self._record()
        self.wall_mm = mm
        self._changed()

    def set_hollow(self, hollow: bool, *, record: bool = True) -> None:
        hollow = bool(hollow)
        if hollow == self.hollow:
            return
        if record:
            self._record()
        self.hollow = hollow
        self._changed()

    def toggle_style(self, key: str, *, record: bool = True) -> None:
        """风格开关：0 ↔ 一个明显但不夸张的值。"""
        current = float(self.style.get(key, 0.0))
        if record:
            self._record()
        if abs(current) > 1e-6:
            self.style.pop(key, None)
        else:
            self.style[key] = {"organic": 0.5, "sharp": 0.55, "rough": 0.5,
                               "twist": 0.4, "detail": 0.4}.get(key, 0.4)
        self._changed()

    def set_style(self, key: str, value: float, *, record: bool = True) -> None:
        if record:
            self._record()
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) < 1e-6:
            self.style.pop(key, None)
        else:
            self.style[key] = value
        self._changed()

    def has_style(self, key: str) -> bool:
        return abs(float(self.style.get(key, 0.0))) > 1e-6

    def set_palette(self, palette_id: str, *, record: bool = True) -> None:
        if palette_id == self.palette_id:
            return
        if palette_id != "image" and palette_id not in config.PALETTES:
            return
        if record:
            self._record()
        self.palette_id = palette_id
        self.emit("state", session=self)
        self.emit("palette", session=self)
        self.emit("idea", session=self)

    def set_prompt_text(self, text: str) -> None:
        self._prompt_text = (text or "").strip()
        self.emit("idea", session=self)

    def randomize(self, *, form: bool = True) -> None:
        self._record()
        rng = random.Random()
        if self.is_concept_mode():
            pool = [c for c in concepts.CONCEPTS if c.id != "abstract"]
            self.concept = rng.choice(pool).id
            self.hollow = concepts.get(self.concept).hollow_default
            self.style = {}
            if rng.random() < 0.45:
                self.style[rng.choice(["organic", "sharp", "rough", "twist"])] = rng.uniform(0.25, 0.6)
        else:
            if form:
                self.form = rng.choice(registry.FORMS).id
            self.params = registry.random_params(self.form, rng)
        self.seed = rng.randint(0, 9999)
        self.palette_id = rng.choice(list(config.PALETTES.keys()))
        self._changed()

    def apply_prompt_result(self, result) -> None:
        if result.concept_id:
            self.concept = result.concept_id
            self.mode = MODE_CONCEPT
            self.hollow = concepts.get(result.concept_id).hollow_default
            if result.palette is None:
                self.palette_id = concepts.palette_for(result.concept_id)
        elif result.form_id:
            self.mode = MODE_CLASSIC
        if result.form_id:
            self.form = result.form_id
            if result.concept_id is None:
                self.params = registry.default_params(result.form_id)
        merged = dict(self.params)
        for key, delta in result.param_delta.items():
            merged[key] = merged.get(key, 50.0) + delta
        self.params = registry.clamp_params(self.form, merged)
        for key, delta in (result.style or {}).items():
            value = float(self.style.get(key, 0.0)) + delta
            if abs(value) < 1e-6:
                self.style.pop(key, None)
            else:
                self.style[key] = max(-1.0, min(1.0, value))
        if result.seed is not None:
            self.seed = int(max(0, min(999999, result.seed)))
        elif result.random_seed:
            self.seed = random.randint(0, 9999)
        if result.res is not None:
            self.res = max(0.0, min(100.0, float(result.res)))
        if result.quality:
            self.quality_id = quality.get(result.quality).id
        if result.size_mm:
            self.size_mm = float(result.size_mm)
        if result.hollow is not None:
            self.hollow = bool(result.hollow)
        if result.palette:
            self.palette_id = result.palette
        self._record()
        self._changed()

    # ------------------------------------------------------------------ 图片
    def _load_image_silently(self, path: str) -> None:
        try:
            feats = imaging.analyze_path(path)
        except Exception as exc:  # noqa: BLE001
            self.image_error = str(exc)
            self.image = None
            return
        self._use_features(feats, path)

    def _use_features(self, feats: imaging.ImageFeatures, path: str) -> None:
        self.image = feats
        self.image_path = path
        self.image_error = ""
        self._image_palette = imaging.palette_from_features(feats)
        if self.palette_id == "image" or self.palette_id not in config.PALETTES:
            self.palette_id = "image"

    def set_image(self, path: str | Path, *, analyze: bool = True) -> None:
        p = Path(path)
        feats = imaging.analyze_path(p) if analyze else None
        self._record()
        if feats is not None:
            self._use_features(feats, str(p))
        self._changed()

    def set_image_from_pil(self, img, name: str = "剪贴板图片") -> None:
        feats = imaging.analyze(img, name=name, path="")
        self._record()
        self._use_features(feats, "")
        self._image_palette = imaging.palette_from_features(feats)
        self._changed()

    def clear_image(self) -> None:
        if self.image is None and not self.image_path:
            return
        self._record()
        self.image = None
        self.image_path = ""
        self._image_palette = None
        self.image_error = ""
        if self.palette_id == "image":
            self.palette_id = config.DEFAULT_PALETTE
        if self.spec.needs_image:
            self.mode = MODE_CLASSIC
            self.form = "vase"
        self._changed()

    # ------------------------------------------------------------------ 撤销
    def can_undo(self) -> bool:
        return bool(self._history)

    def can_redo(self) -> bool:
        return bool(self._future)

    def undo(self) -> bool:
        if not self._history:
            return False
        snap = self._history.pop()
        self._future.append(self.snapshot())
        self._apply_snapshot(snap)
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        snap = self._future.pop()
        self._history.append(self.snapshot())
        self._apply_snapshot(snap)
        return True

    def _apply_snapshot(self, snap: Snapshot) -> None:
        self.mode = snap.mode
        self.concept = snap.concept
        self.form = snap.form
        self.seed = snap.seed
        self.res = snap.res
        self.params = registry.clamp_params(snap.form, {"a": snap.a, "b": snap.b})
        self.palette_id = snap.palette
        self.quality_id = quality.get(snap.quality).id
        self.size_mm = snap.size_mm
        self.hollow = snap.hollow
        self.wall_mm = snap.wall_mm
        self.style = {}
        for item in (snap.style or "").split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    self.style[k] = float(v)
                except ValueError:
                    pass
        if snap.image_path != self.image_path:
            if snap.image_path:
                self._load_image_silently(snap.image_path)
            else:
                self.image = None
                self.image_path = ""
        self._changed()

    # ------------------------------------------------------------------ 持久化
    def to_settings(self) -> dict:
        return {
            "mode": self.mode,
            "concept": self.concept,
            "form": self.form,
            "seed": self.seed,
            "res": self.res,
            "params": dict(self.params),
            "palette": self.palette_id,
            "quality": self.quality_id,
            "size_mm": self.size_mm,
            "wall_mm": self.wall_mm,
            "hollow": self.hollow,
            "style": dict(self.style),
            "image_path": self.image_path,
            "image_mode": self.image_mode,
        }

    def save(self) -> None:
        self.settings.update(self.to_settings(), save=True)

    def export_options_defaults(self) -> dict:
        return {
            "stl": True,
            "obj": False,
            "three_mf": False,
            "png": True,
            "card": True,
            "report": True,
            "png_width": 1400,
            "png_height": 1050,
            "transparent": False,
            "supersample": 2,
        }
