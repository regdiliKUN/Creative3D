"""settings.py —— 设置持久化（带版本迁移与损坏自愈）。

任何一次读写失败都不应该让程序崩溃：读失败用默认值，写失败静默记录日志。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from . import config, logsetup

SCHEMA = 3

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    "mode": "concept",
    "concept": "vase",
    "form": "vase",
    "seed": 2026,
    "res": 62.0,
    "params": {"a": 45.0, "b": 45.0},
    "palette": "frost",
    "quality": "fine",
    "size_mm": 100.0,
    "wall_mm": 1.8,
    "hollow": None,
    "style": {},
    "auto_rotate": True,
    "wire": False,
    "shadow": True,
    "smooth_export": True,
    "export_dir": "",
    "image_path": "",
    "image_mode": "silhouette",
    "window_size": [1500, 940],
    "window_pos": None,
    "welcome_seen": False,
    "recent_exports": [],
    "favorites": [],
    "user_presets": {},
}

log = logsetup.get("settings")


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.settings_path()
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    # ------------------------------------------------------------------ 读写
    def load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("读取设置失败：%s", exc)
            return
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("设置文件不是对象")
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("设置文件损坏，已备份并重置：%s", exc)
            self._quarantine()
            return
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in data.items() if k in DEFAULTS or k == "schema"})
        merged["schema"] = data.get("schema", 1)
        self.data = self._migrate(merged)

    def _migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        version = int(data.get("schema", 1) or 1)
        out = dict(data)
        if version < 3:
            # v2 只有形态 + 参数，补上 v3 的「概念模型 / 精度 / 尺寸」默认项
            out.setdefault("mode", DEFAULTS["mode"])
            out.setdefault("concept", DEFAULTS["concept"])
            out.setdefault("quality", DEFAULTS["quality"])
            out.setdefault("size_mm", DEFAULTS["size_mm"])
            out.setdefault("wall_mm", DEFAULTS["wall_mm"])
            out.setdefault("hollow", DEFAULTS["hollow"])
            out.setdefault("style", {})
        if version < 2:
            # v1 只有 form/seed/res/params，补上 v2 的默认项
            out.setdefault("palette", DEFAULTS["palette"])
            out.setdefault("image_mode", DEFAULTS["image_mode"])
            out.setdefault("favorites", [])
            out.setdefault("recent_exports", [])
        out["schema"] = SCHEMA
        for key, value in DEFAULTS.items():
            out.setdefault(key, value)
        return out

    def _quarantine(self) -> None:
        try:
            if self.path.exists():
                shutil.copy2(self.path, self.path.with_suffix(".bad.json"))
        except OSError:
            pass

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            return True
        except OSError as exc:
            log.warning("保存设置失败：%s", exc)
            return False

    # ------------------------------------------------------------------ 访问
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any, *, save: bool = False) -> None:
        self.data[key] = value
        if save:
            self.save()

    def update(self, mapping: dict[str, Any], *, save: bool = False) -> None:
        self.data.update(mapping)
        if save:
            self.save()

    # ------------------------------------------------------------------ 便捷
    def add_recent_export(self, entry: dict) -> None:
        recent = [e for e in self.get("recent_exports", []) if e.get("path") != entry.get("path")]
        recent.insert(0, entry)
        self.data["recent_exports"] = recent[:12]

    def toggle_favorite(self, entry: dict) -> bool:
        favs = list(self.get("favorites", []))
        key = entry.get("key")
        existing = next((f for f in favs if f.get("key") == key), None)
        if existing:
            favs.remove(existing)
            self.data["favorites"] = favs
            return False
        favs.insert(0, entry)
        self.data["favorites"] = favs[:60]
        return True

    def is_favorite(self, key: str) -> bool:
        return any(f.get("key") == key for f in self.get("favorites", []))
