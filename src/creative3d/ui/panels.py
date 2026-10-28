"""ui.panels —— 左侧（输入 / 形态 / 参数）与右侧（创意卡 / 配色 / 导出）面板。

面板只负责「显示 + 收集用户意图」，所有状态都来自 Session，
所有副作用都通过 app 的方法执行（打开对话框、起后台任务等）。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

from .. import config, nlp, registry
from . import theme
from .widgets import (
    auto_wrap,
    Button,
    Chip,
    IconButton,
    ScrollFrame,
    Section,
    Segmented,
    Slider,
    Toggle,
    Tooltip,
    hr,
)

T = config.THEME


class SliderRow:
    """一根滑杆的三件套（容器 / 名称 / 滑杆 / 数值），便于刷新时只改数值。"""

    __slots__ = ("frame", "label", "slider", "value_label")

    def __init__(self, frame, label, slider, value_label) -> None:
        self.frame = frame
        self.label = label
        self.slider = slider
        self.value_label = value_label


class LeftPanel(tk.Frame):
    def __init__(self, master, app) -> None:
        super().__init__(master, bg=T.panel, width=368, bd=0, highlightthickness=0)
        self.pack_propagate(False)
        self.app = app
        self.scroll = ScrollFrame(self, bg=T.panel)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner

        self._building_text = False
        self._form_chips: dict[str, Chip] = {}
        self._preset_widgets: list[tk.Widget] = []

        self._build_inspiration()
        hr(self.body)
        self._build_forms()
        hr(self.body)
        self._build_params()
        hr(self.body)
        self._build_presets()
        tk.Frame(self.body, bg=T.panel, height=16).pack()

    # ================================================================== 灵感
    def _build_inspiration(self) -> None:
        Section(
            self.body,
            "灵感输入",
            subtitle="直接描述想法，或丢一张图片进来 —— 两种输入都会转成可调参数",
        ).pack(fill="x")

        seg_holder = tk.Frame(self.body, bg=T.panel)
        seg_holder.pack(fill="x", padx=16)
        self.seg = Segmented(seg_holder, ["文本描述", "图片参考"], self._on_tab)
        self.seg.pack(anchor="w")

        self.text_frame = tk.Frame(self.body, bg=T.panel)
        self.text_frame.pack(fill="x", padx=16, pady=(10, 0))

        box = tk.Frame(self.text_frame, bg=T.line, bd=0)
        box.pack(fill="x")
        # 必须显式给 width：Text 默认 80 字符宽，会把整个滚动容器撑到 1100px 外
        self.text_box = tk.Text(
            box, height=3, width=30, bg=T.panel_alt, fg=T.text, insertbackground=T.accent,
            relief="flat", bd=0, highlightthickness=0, wrap="word",
            font=theme.f(9), padx=10, pady=8, spacing1=2, spacing3=2,
        )
        self.text_box.pack(fill="x", padx=1, pady=1)
        self._placeholder = "例如：冷色的尖锐晶簇，细节 80"
        self._set_placeholder()
        self.text_box.bind("<FocusIn>", self._on_text_focus)
        self.text_box.bind("<FocusOut>", self._on_text_blur)
        self.text_box.bind("<Control-Return>", lambda e: (self.app.parse_prompt(), "break")[1])

        row = tk.Frame(self.text_frame, bg=T.panel)
        row.pack(fill="x", pady=(6, 0))
        Button(row, "解析并应用", self.app.parse_prompt, kind="primary", padx=12, pady=6,
               font_size=9).pack(side="left")
        tk.Label(row, text="Ctrl+Enter", bg=T.panel, fg=T.text_mute,
                 font=theme.f(8)).pack(side="left", padx=8)

        self.prompt_note = auto_wrap(
            tk.Label(self.text_frame, text="", bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     justify="left", anchor="w"), 2)
        self.prompt_note.pack(fill="x", pady=(6, 0))

        # ---------------- 图片
        self.image_frame = tk.Frame(self.body, bg=T.panel)

        drop = tk.Frame(self.image_frame, bg=T.panel_alt, highlightthickness=1,
                        highlightbackground=T.line)
        drop.pack(fill="x")
        self.thumb = tk.Label(drop, bg=T.panel_alt, fg=T.text_mute, text="未选择图片",
                              font=theme.f(9), height=6, cursor="hand2")
        self.thumb.pack(fill="x", padx=1, pady=1)
        self.thumb.bind("<Button-1>", lambda e: self.app.pick_image())

        btns = tk.Frame(self.image_frame, bg=T.panel)
        btns.pack(fill="x", pady=(8, 0))
        Button(btns, "选择图片…", self.app.pick_image, kind="secondary", padx=10, pady=6,
               font_size=9).pack(side="left")
        Button(btns, "粘贴", self.app.paste_image, kind="ghost", padx=10, pady=6,
               font_size=9).pack(side="left", padx=4)
        Button(btns, "清除", self.app.clear_image, kind="ghost", padx=10, pady=6,
               font_size=9).pack(side="left")

        self.img_summary = auto_wrap(
            tk.Label(self.image_frame, text="", bg=T.panel, fg=T.text_mute,
                     font=theme.f(8), justify="left", anchor="w"), 2)
        self.img_summary.pack(fill="x", pady=(8, 0))

        self.palette_strip = tk.Frame(self.image_frame, bg=T.panel)
        self.palette_strip.pack(fill="x", pady=(8, 0))

        tk.Label(self.image_frame, text="用这张图做什么", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(anchor="w", pady=(10, 4))
        self.mode_row = tk.Frame(self.image_frame, bg=T.panel)
        self.mode_row.pack(fill="x")
        self._mode_chips: dict[str, Chip] = {}
        modes = (
            ("silhouette", "轮廓 → 旋转体", "提取主体轮廓，绕轴旋转成器皿"),
            ("relief", "亮度 → 浮雕", "把明暗转成高度，做成可打印浮雕牌"),
            ("palette", "仅取配色", "只把配色套到当前形态上"),
        )
        for key, label, tip in modes:
            chip = Chip(self.mode_row, label, lambda k=key: self.app.set_image_mode(k),
                        font_size=8, padx=8, pady=5)
            chip.pack(fill="x", pady=2)
            Tooltip(chip, tip)
            self._mode_chips[key] = chip

        self.image_frame.pack_forget()

    def _set_placeholder(self) -> None:
        self._building_text = True
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", self._placeholder)
        self.text_box.configure(fg=T.text_mute)
        self._building_text = False

    def _on_text_focus(self, _e=None) -> None:
        if self.text_box.get("1.0", "end").strip() == self._placeholder:
            self.text_box.delete("1.0", "end")
            self.text_box.configure(fg=T.text)

    def _on_text_blur(self, _e=None) -> None:
        if not self.text_box.get("1.0", "end").strip():
            self._set_placeholder()

    def prompt_text(self) -> str:
        raw = self.text_box.get("1.0", "end").strip()
        return "" if raw == self._placeholder else raw

    def _on_tab(self, index: int) -> None:
        if index == 0:
            self.image_frame.pack_forget()
            self.text_frame.pack(fill="x", padx=16, pady=(10, 0), after=self.seg.master)
        else:
            self.text_frame.pack_forget()
            self.image_frame.pack(fill="x", padx=16, pady=(10, 0), after=self.seg.master)

    def show_image_tab(self) -> None:
        self.seg.select(1)
        self._on_tab(1)

    # ================================================================== 形态
    def _build_forms(self) -> None:
        Section(self.body, "形态", subtitle="8 个参数化形态族，点击即切换").pack(fill="x")
        grid = tk.Frame(self.body, bg=T.panel)
        grid.pack(fill="x", padx=12)
        # 两列布局：中文形态名 + 图标横向需要约 135px，四列会被裁掉半个字
        for i, spec in enumerate(registry.FORMS):
            chip = Chip(
                grid,
                spec.label,
                lambda fid=spec.id: self.app.select_form(fid),
                glyph=spec.glyph,
                font_size=9,
                padx=9,
                pady=7,
                anchor="w",
            )
            chip.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="nsew")
            Tooltip(chip, f"{spec.label} · {spec.desc}\n{spec.detail}")
            self._form_chips[spec.id] = chip
        for col in range(2):
            grid.columnconfigure(col, weight=1, uniform="form")

    # ================================================================== 参数
    def _build_params(self) -> None:
        Section(self.body, "参数", subtitle="拖动即实时预览；停止拖动后自动切到高清").pack(fill="x")

        self.res_row = self._make_slider_row("细节密度", "影响面数与导出体积", 0, 100, 62, "res")
        self.param_rows: dict[str, "SliderRow"] = {}
        for key in ("a", "b"):
            spec = registry.get(self.app.session.form).param(key)
            self.param_rows[key] = self._make_slider_row(
                spec.label, spec.hint, spec.lo, spec.hi, spec.default, key
            )

        seed_row = tk.Frame(self.body, bg=T.panel)
        seed_row.pack(fill="x", padx=16, pady=(6, 0))
        tk.Label(seed_row, text="随机种子", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.seed_var = tk.StringVar(value="2026")
        self.seed_entry = ttk.Entry(seed_row, textvariable=self.seed_var, width=8,
                                    style="C3D.TEntry", justify="center")
        self.seed_entry.pack(side="right")
        self.seed_entry.bind("<Return>", lambda e: self._commit_seed())
        self.seed_entry.bind("<FocusOut>", lambda e: self._commit_seed())
        IconButton(seed_row, "⟳", self.app.randomize, tip="换一组随机灵感（N）").pack(
            side="right", padx=4)

        note = auto_wrap(
            tk.Label(self.body,
                     text="同一个种子 + 同一组参数 = 完全相同的模型与文案，方便复现。",
                     bg=T.panel, fg=T.text_mute, font=theme.f(8), justify="left", anchor="w"), 2)
        note.pack(fill="x", padx=16, pady=(6, 0))

    def _make_slider_row(self, label: str, hint: str, lo: float, hi: float,
                         value: float, key: str) -> "SliderRow":
        holder = tk.Frame(self.body, bg=T.panel)
        holder.pack(fill="x", padx=16, pady=(6, 0))
        head = tk.Frame(holder, bg=T.panel)
        head.pack(fill="x")
        lbl = tk.Label(head, text=label, bg=T.panel, fg=T.text_dim, font=theme.f(9))
        lbl.pack(side="left")
        val = tk.Label(head, text=f"{value:.0f}", bg=T.panel, fg=T.accent, font=theme.mono(9))
        val.pack(side="right")
        Tooltip(head, hint)
        slider = Slider(
            holder, lo=lo, hi=hi, value=value, step=1,
            command=lambda v, k=key: self._on_slider(k, v),
            release=lambda v, k=key: self._on_slider_commit(k, v),
            bg=T.panel,
        )
        slider.pack(fill="x", pady=(2, 0))
        return SliderRow(holder, lbl, slider, val)

    def _on_slider(self, key: str, value: float) -> None:
        if key == "res":
            self.res_row.value_label.configure(text=f"{value:.0f}")
            self.app.preview_res(value)
        else:
            row = self.param_rows.get(key)
            if row:
                row.value_label.configure(text=f"{value:.0f}")
            self.app.preview_param(key, value)

    def _on_slider_commit(self, key: str, value: float) -> None:
        if key == "res":
            self.app.commit_res(value)
        else:
            self.app.commit_param(key, value)

    def _commit_seed(self) -> None:
        raw = self.seed_var.get().strip()
        try:
            value = int(float(raw))
        except ValueError:
            self.seed_var.set(str(self.app.session.seed))
            self.app.show_toast("种子只能是 0–999999 的整数", kind="warn")
            return
        self.app.commit_seed(value)

    # ================================================================== 预设
    def _build_presets(self) -> None:
        Section(self.body, "预设", subtitle="每个形态都有一组调好的起点").pack(fill="x")
        self.preset_holder = tk.Frame(self.body, bg=T.panel)
        self.preset_holder.pack(fill="x", padx=16)
        self._rebuild_presets(self.app.session.form)

    def _rebuild_presets(self, form_id: str) -> None:
        for widget in self._preset_widgets:
            widget.destroy()
        self._preset_widgets.clear()
        spec = registry.get(form_id)
        for preset in spec.presets:
            chip = Chip(self.preset_holder, preset.name,
                        lambda p=preset: self.app.apply_preset(p), font_size=8, padx=10, pady=5)
            chip.pack(side="left", padx=(0, 4), pady=2)
            Tooltip(chip, f"{preset.note}\n预设参数 {preset.a:.0f} / {preset.b:.0f}")
            self._preset_widgets.append(chip)

    # ================================================================== 刷新
    def refresh(self, session) -> None:
        for fid, chip in self._form_chips.items():
            chip.set_selected(fid == session.form)

        spec = session.spec
        # 参数名随形态变化
        for key in ("a", "b"):
            row = self.param_rows.get(key)
            if not row:
                continue
            p = spec.param(key)
            row.label.configure(text=p.label)
            value = float(session.params.get(key, p.default))
            row.slider.set_value(value)
            row.value_label.configure(text=f"{value:.0f}")

        self.res_row.slider.set_value(session.res)
        self.res_row.value_label.configure(text=f"{session.res:.0f}")
        self.seed_var.set(str(session.seed))

        if self._preset_widgets and getattr(self, "_preset_form", None) != session.form:
            self._rebuild_presets(session.form)
        self._preset_form = session.form

        for key, chip in self._mode_chips.items():
            chip.set_selected(key == session.image_mode)

        if session.image is not None:
            self._render_thumb(session)
            if self._thumb_photo is not None:
                self.thumb.configure(text="", image=self._thumb_photo, height=0)
            self.img_summary.configure(text=f"{session.image.name}\n{session.image.summary()}")
            self._render_palette(session)
        else:
            self.thumb.configure(image="", text="未选择图片 —— 点击选择，或直接 Ctrl+V 粘贴",
                                 height=7, fg=T.text_mute)
            self.img_summary.configure(text="")
            for widget in self.palette_strip.winfo_children():
                widget.destroy()

    def _render_thumb(self, session) -> None:
        path = session.image_path
        try:
            if path:
                img = Image.open(path)
            else:
                img = None
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            self._thumb_photo = None
            return
        img = img.convert("RGBA")
        img.thumbnail((310, 120), Image.LANCZOS)
        # 贴到固定底色上，避免透明图片变成黑块
        canvas = Image.new("RGB", img.size, (24, 30, 42))
        canvas.paste(img, mask=img.split()[3])
        self._thumb_photo = ImageTk.PhotoImage(canvas, master=self)

    def _render_palette(self, session) -> None:
        for widget in self.palette_strip.winfo_children():
            widget.destroy()
        feats = session.image
        if feats is None:
            return
        row = tk.Frame(self.palette_strip, bg=T.panel)
        row.pack(fill="x")
        for sw in feats.swatches[:5]:
            cell = tk.Frame(row, bg=T.panel)
            cell.pack(side="left", expand=True, fill="x", padx=1)
            swatch = tk.Frame(cell, bg=sw.hex, height=22)
            swatch.pack(fill="x")
            tk.Label(cell, text=sw.name, bg=T.panel, fg=T.text_mute,
                     font=theme.f(7)).pack(pady=(2, 0))
            Tooltip(swatch, f"{sw.name} {sw.hex.upper()}\n占比 {sw.ratio * 100:.0f}%")

    def set_prompt_note(self, text: str, *, kind: str = "muted") -> None:
        color = {"ok": T.ok, "warn": T.warn, "muted": T.text_mute, "err": T.err}.get(kind, T.text_mute)
        self.prompt_note.configure(text=text, fg=color)


class RightPanel(tk.Frame):
    def __init__(self, master, app) -> None:
        super().__init__(master, bg=T.panel, width=340, bd=0, highlightthickness=0)
        self.pack_propagate(False)
        self.app = app
        self.scroll = ScrollFrame(self, bg=T.panel)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner

        self._build_card()
        hr(self.body)
        self._build_palette()
        hr(self.body)
        self._build_export()

    # ------------------------------------------------------------------ 创意卡
    def _build_card(self) -> None:
        header = tk.Frame(self.body, bg=T.panel)
        header.pack(fill="x", padx=16, pady=(14, 4))
        # 先 pack 右侧图标，再让标题占满剩余宽度，否则标题会被图标挤掉一截
        self.fav_btn = IconButton(header, "☆", self.app.toggle_favorite, tip="收藏这一件")
        self.fav_btn.pack(side="right")
        tk.Label(header, text="创意卡", bg=T.panel, fg=T.text,
                 font=theme.f(10, True)).pack(side="left", fill="x", expand=True)

        card = tk.Frame(self.body, bg=T.card, highlightthickness=1, highlightbackground=T.line)
        card.pack(fill="x", padx=16, pady=(4, 0))
        self.card_title = auto_wrap(
            tk.Label(card, text="", bg=T.card, fg=T.text, font=theme.f(13, True),
                     justify="left", anchor="w"), 2)
        self.card_title.pack(fill="x", padx=14, pady=(12, 6))
        self.card_story = auto_wrap(
            tk.Label(card, text="", bg=T.card, fg=T.text_dim, font=theme.f(9),
                     justify="left", anchor="w"), 2)
        self.card_story.pack(fill="x", padx=14)

        self.tag_row = tk.Frame(card, bg=T.card)
        self.tag_row.pack(fill="x", padx=14, pady=(8, 0))
        self._tag_labels: list[tk.Label] = []

        self.meta_holder = tk.Frame(card, bg=T.card)
        self.meta_holder.pack(fill="x", padx=14, pady=(8, 12))
        self._meta_labels: list[tuple[tk.Label, tk.Label]] = []
        for name in ("材质", "用途", "打印"):
            row = tk.Frame(self.meta_holder, bg=T.card)
            row.pack(fill="x", pady=1)
            key = tk.Label(row, text=name, bg=T.card, fg=T.text_mute, font=theme.f(8), width=4,
                           anchor="w")
            key.pack(side="left")
            val = auto_wrap(
                tk.Label(row, text="", bg=T.card, fg=T.text_dim, font=theme.f(8),
                         anchor="w", justify="left"), 2)
            val.pack(side="left", fill="x", expand=True)
            self._meta_labels.append((key, val))

        self.card_footer = auto_wrap(
            tk.Label(card, text="", bg=T.card, fg=T.text_mute, font=theme.f(8),
                     anchor="w", justify="left"), 2)
        self.card_footer.pack(fill="x", padx=14, pady=(0, 10))

    def refresh_idea(self, idea, *, stats: str = "") -> None:
        self.card_title.configure(text=idea.title)
        self.card_story.configure(text=idea.story)

        for lbl in self._tag_labels:
            lbl.destroy()
        self._tag_labels.clear()
        for i, tag in enumerate(idea.tags[:6]):
            lbl = tk.Label(self.tag_row, text=f"#{tag}", bg=T.panel_alt, fg=T.text_mute,
                           font=theme.f(8), padx=6, pady=2)
            lbl.grid(row=i // 3, column=i % 3, padx=(0, 4), pady=2, sticky="w")
            self._tag_labels.append(lbl)

        values = (idea.material, idea.usage, idea.tip)
        for (_key, val), text in zip(self._meta_labels, values):
            val.configure(text=text)
        self.card_footer.configure(text=f"{stats} · seed {idea.extra.get('seed', '')}" if stats
                                   else f"seed {idea.extra.get('seed', '')}")

    def set_favorite(self, is_fav: bool) -> None:
        self.fav_btn.configure(text="★" if is_fav else "☆",
                               fg=T.warn if is_fav else T.text_dim)

    # ------------------------------------------------------------------ 配色
    def _build_palette(self) -> None:
        Section(self.body, "配色", subtitle="点选即换；有图片时可套用图片主色").pack(fill="x")
        holder = tk.Frame(self.body, bg=T.panel)
        holder.pack(fill="x", padx=12, pady=(0, 4))
        self._palette_cells: dict[str, tk.Frame] = {}
        items = list(config.PALETTES.values())
        for i, pal in enumerate(items):
            cell = tk.Frame(holder, bg=T.panel, cursor="hand2")
            cell.grid(row=i // 4, column=i % 4, padx=3, pady=3, sticky="nsew")
            swatch = tk.Frame(cell, bg=pal.hex("base"), height=26, cursor="hand2")
            swatch.pack(fill="x")
            label = tk.Label(cell, text=pal.label, bg=T.panel, fg=T.text_mute, font=theme.f(7),
                             cursor="hand2")
            label.pack(pady=(2, 0))
            for widget in (cell, swatch, label):
                widget.bind("<Button-1>", lambda e, pid=pal.id: self.app.select_palette(pid))
            Tooltip(swatch, f"{pal.label} · {pal.hex('base').upper()}")
            self._palette_cells[pal.id] = cell
            cell.bind("<Enter>", lambda e, c=cell: c.configure(highlightthickness=0))
        for col in range(4):
            holder.columnconfigure(col, weight=1, uniform="pal")

        self.image_palette_btn = Chip(
            self.body, "套用图片主色", lambda: self.app.select_palette("image"),
            font_size=8, padx=10, pady=5,
        )

    def refresh_palette(self, session) -> None:
        for pid, cell in self._palette_cells.items():
            selected = pid == session.palette_id
            cell.configure(bg=T.accent_soft if selected else T.panel)
            for child in cell.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=T.accent_soft if selected else T.panel,
                                    fg=T.accent if selected else T.text_mute)
        if session.image is not None:
            self.image_palette_btn.pack(fill="x", padx=16, pady=(4, 0))
            self.image_palette_btn.set_selected(session.palette_id == "image")
        else:
            self.image_palette_btn.pack_forget()

    # ------------------------------------------------------------------ 导出
    def _build_export(self) -> None:
        Section(self.body, "导出", subtitle="STL 可直接进切片软件；创意卡方便归档").pack(fill="x")

        formats = tk.Frame(self.body, bg=T.panel)
        formats.pack(fill="x", padx=16)
        self._format_chips: dict[str, Chip] = {}
        # 2×2：四个芯片横向排需要 315px，而栏内只有 304px，最后一个会被挤掉字
        for i, (key, label) in enumerate(
            (("stl", "STL"), ("obj", "OBJ"), ("png", "预览图"), ("card", "创意卡"))
        ):
            chip = Chip(formats, label, lambda k=key: self.app.toggle_export_format(k),
                        selected=True, font_size=8, padx=9, pady=5, anchor="w")
            chip.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="nsew")
            self._format_chips[key] = chip
        for col in range(2):
            formats.columnconfigure(col, weight=1, uniform="fmt")

        self.export_btn = Button(self.body, "导出当前作品", self.app.open_export_dialog,
                                 kind="primary", padx=14, pady=10, font_size=10, bold=True)
        self.export_btn.pack(fill="x", padx=16, pady=(10, 4))

        row = tk.Frame(self.body, bg=T.panel)
        row.pack(fill="x", padx=16)
        Button(row, "打开导出目录", self.app.open_export_dir, kind="ghost", padx=10, pady=6,
               font_size=8).pack(side="left")
        Button(row, "保存设置", self.app.save_settings, kind="ghost", padx=10, pady=6,
               font_size=8).pack(side="right")

        tk.Label(self.body, text="最近导出", bg=T.panel, fg=T.text_mute,
                 font=theme.f(8)).pack(anchor="w", padx=16, pady=(12, 2))
        self.recent_holder = tk.Frame(self.body, bg=T.panel)
        self.recent_holder.pack(fill="x", padx=16, pady=(0, 12))

    def refresh_formats(self, enabled: dict[str, bool]) -> None:
        for key, chip in self._format_chips.items():
            chip.set_selected(enabled.get(key, True))

    def refresh_recent(self, entries: list[dict]) -> None:
        for widget in self.recent_holder.winfo_children():
            widget.destroy()
        if not entries:
            tk.Label(self.recent_holder, text="还没有导出记录", bg=T.panel, fg=T.text_mute,
                     font=theme.f(8)).pack(anchor="w")
            return
        for entry in entries[:4]:
            row = tk.Frame(self.recent_holder, bg=T.panel, cursor="hand2")
            row.pack(fill="x", pady=1)
            name = entry.get("name", "")
            lbl = tk.Label(row, text=f"· {name}", bg=T.panel, fg=T.text_dim, font=theme.f(8),
                           anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True)
            for widget in (row, lbl):
                widget.bind("<Button-1>",
                            lambda e, p=entry.get("path"): self.app.open_path(p))
            lbl.bind("<Enter>", lambda e, w=lbl: w.configure(fg=T.accent))
            lbl.bind("<Leave>", lambda e, w=lbl: w.configure(fg=T.text_dim))
