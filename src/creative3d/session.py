"""session.py —— 单一状态源。

界面上的每个控件都只读写 Session，绝不自持状态：
    「滑杆 → Session → 事件 → 视口 / 创意卡 / 导出」只有一条数据流，
    因此不会出现「界面显示 30、导出用的是 50」这类错位。
"""

from __future__ import annotations

import random
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import config, generators, ideas, imaging, registry
from .generators import GenContext
from .mesh import Mesh

PREVIEW_RES_CAP = 44.0  # 交互时最高精度，超过就靠静止后的高精度补帧


@dataclass
class Snapshot:
    form: str
    seed: int
    res: float
    a: float
    b: float
    palette: str
    image_path: str = ""
    image_mode: str = "silhouette"

    def key(self) -> str:
        return f"{self.form}|{self.seed}|{self.res}|{self.a}|{self.b}|{self.palette}|{self.image_path}|{self.image_mode}"


class Session:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.form = registry.get(settings.get("form", "vase")).id
        self.seed = int(settings.get("seed", 2026))
        self.res = float(settings.get("res", 62))
        self.params = registry.clamp_params(self.form, settings.get("params", {}) or {})
        self.palette_id = settings.get("palette", config.DEFAULT_PALETTE)
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

    def snapshot(self) -> Snapshot:
        return Snapshot(
            form=self.form,
            seed=self.seed,
            res=self.res,
            a=float(self.params.get("a", 50)),
            b=float(self.params.get("b", 50)),
            palette=self.palette_id,
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
        )

    def favorite_key(self) -> str:
        return zlib.crc32(self.snapshot().key().encode("utf-8")).__str__()

    # ------------------------------------------------------------------ 生成
    def _cache_key(self, res: float) -> tuple:
        return (
            self.form,
            self.seed,
            round(res, 1),
            round(float(self.params.get("a", 50)), 2),
            round(float(self.params.get("b", 50)), 2),
            self.image_path,
        )

    def generate_mesh(self, *, preview: bool = False) -> Mesh:
        res = min(PREVIEW_RES_CAP, self.res) if preview else self.res
        key = self._cache_key(res)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            self._mesh_cache.move_to_end(key)
            return cached

        ctx = GenContext(
            seed=self.seed,
            res=res,
            a=float(self.params.get("a", 50)),
            b=float(self.params.get("b", 50)),
            image=self.image if self.spec.needs_image or self.image is not None else None,
        )
        mesh = generators.build(self.form, ctx)
        mesh.normalize(config.MODEL_TARGET_MM)
        mesh.meta["preview"] = preview
        self._mesh_cache[key] = mesh
        while len(self._mesh_cache) > 8:
            self._mesh_cache.popitem(last=False)
        return mesh

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
        if form:
            self.form = rng.choice(registry.FORMS).id
        self.params = registry.random_params(self.form, rng)
        self.seed = rng.randint(0, 9999)
        self.palette_id = rng.choice(list(config.PALETTES.keys()))
        self._changed()

    def apply_prompt_result(self, result) -> None:
        before = self.snapshot()
        if result.form_id:
            self.form = result.form_id
            if before.form != result.form_id:
                self.params = registry.default_params(result.form_id)
        merged = dict(self.params)
        for key, delta in result.param_delta.items():
            merged[key] = merged.get(key, 50.0) + delta
        self.params = registry.clamp_params(self.form, merged)
        if result.seed is not None:
            self.seed = int(max(0, min(999999, result.seed)))
        elif result.random_seed:
            self.seed = random.randint(0, 9999)
        if result.res is not None:
            self.res = max(0.0, min(100.0, float(result.res)))
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
        self.form = snap.form
        self.seed = snap.seed
        self.res = snap.res
        self.params = registry.clamp_params(snap.form, {"a": snap.a, "b": snap.b})
        self.palette_id = snap.palette
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
            "form": self.form,
            "seed": self.seed,
            "res": self.res,
            "params": dict(self.params),
            "palette": self.palette_id,
            "image_path": self.image_path,
            "image_mode": self.image_mode,
        }

    def save(self) -> None:
        self.settings.update(self.to_settings(), save=True)

    def export_options_defaults(self) -> dict:
        return {
            "stl": True,
            "obj": False,
            "png": True,
            "card": True,
            "png_width": 1400,
            "png_height": 1050,
            "transparent": False,
            "supersample": 2,
        }
