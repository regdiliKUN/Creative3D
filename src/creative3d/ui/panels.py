"""ui.panels —— 左侧（提示词 / 模型库 / 精度 / 工艺 / 风格）与右侧（创意卡 / 打印体检 / 配色 / 导出）。

面板只负责「显示 + 收集用户意图」，所有状态都来自 Session，
所有副作用都通过 app 的方法执行（打开对话框、起后台任务等）。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .. import concepts, config, quality
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
        super().__init__(master, bg=T.panel, width=config.sp(372), bd=0, highlightthickness=0)
        self.pack_propagate(False)
        self.app = app
        self.scroll = ScrollFrame(self, bg=T.panel)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner

        self._building_text = False
        self._form_chips: dict[str, Chip] = {}
        self._concept_chips: dict[str, Chip] = {}
        self._category = concepts.CATEGORIES[0]
        self._category_chips: dict[str, Chip] = {}
        self._quality_chips: dict[str, Chip] = {}
        self._style_chips: dict[str, Chip] = {}
        self._preset_widgets: list[tk.Widget] = []

        self._build_inspiration()
        hr(self.body)
        self._build_library()
        hr(self.body)
        self._build_quality()
        hr(self.body)
        self._build_craft()
        hr(self.body)
        self._build_style()
        hr(self.body)
        self._build_params()
        hr(self.body)
        self._build_classic()
        tk.Frame(self.body, bg=T.panel, height=16).pack()

    # ================================================================== 灵感
    def _build_inspiration(self) -> None:
        Section(
            self.body,
            "一句话建模",
            subtitle="描述任何东西 —— 猫、齿轮、城堡、汉堡…… 没见过的词也能生成抽象构成",
        ).pack(fill="x")

        box = tk.Frame(self.body, bg=T.line, bd=0)
        box.pack(fill="x", padx=16)
        self.text_box = tk.Text(
            box, height=3, width=30, bg=T.panel_alt, fg=T.text, insertbackground=T.accent,
            relief="flat", bd=0, highlightthickness=0, wrap="word",
            font=theme.f(9), padx=10, pady=8, spacing1=2, spacing3=2,
        )
        self.text_box.pack(fill="x", padx=1, pady=1)
        self._placeholder = "例如：一只坐着的猫，圆润一点，高 120mm，打印级"
        self._set_placeholder()
        self.text_box.bind("<FocusIn>", self._on_text_focus)
        self.text_box.bind("<FocusOut>", self._on_text_blur)
        self.text_box.bind("<Control-Return>", lambda e: (self.app.parse_prompt(), "break")[1])

        row = tk.Frame(self.body, bg=T.panel)
        row.pack(fill="x", padx=16, pady=(6, 0))
        Button(row, "生成模型", self.app.parse_prompt, kind="primary", padx=14, pady=7,
               font_size=9, bold=True).pack(side="left")
        tk.Label(row, text="Ctrl+Enter", bg=T.panel, fg=T.text_mute,
                 font=theme.f(8)).pack(side="left", padx=8)

        self.prompt_note = auto_wrap(
            tk.Label(self.body, text="", bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     justify="left", anchor="w"), 2)
        self.prompt_note.pack(fill="x", padx=16, pady=(6, 0))

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

    def set_prompt_note(self, text: str, *, kind: str = "muted") -> None:
        color = {"ok": T.ok, "warn": T.warn, "muted": T.text_mute, "err": T.err}.get(kind, T.text_mute)
        self.prompt_note.configure(text=text, fg=color)

    # ================================================================== 模型库
    def _build_library(self) -> None:
        Section(self.body, "模型库", subtitle=f"{len(concepts.CONCEPTS)} 个可直接生成的模型 · 点选即切换"
                ).pack(fill="x")

        cats = tk.Frame(self.body, bg=T.panel)
        cats.pack(fill="x", padx=12)
        for i, cat in enumerate(concepts.CATEGORIES):
            chip = Chip(cats, cat, lambda c=cat: self._select_category(c), font_size=8,
                        padx=6, pady=5, anchor="center")
            chip.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="nsew")
            self._category_chips[cat] = chip
        for col in range(4):
            cats.columnconfigure(col, weight=1, uniform="cat")

        self.concept_holder = tk.Frame(self.body, bg=T.panel)
        self.concept_holder.pack(fill="x", padx=12, pady=(6, 0))
        self._rebuild_concepts(self._category)

    def _select_category(self, category: str) -> None:
        self._category = category
        self._rebuild_concepts(category)
        self.refresh_categories()

    def _rebuild_concepts(self, category: str) -> None:
        for widget in self.concept_holder.winfo_children():
            widget.destroy()
        self._concept_chips.clear()
        items = concepts.by_category(category)
        for i, c in enumerate(items):
            chip = Chip(
                self.concept_holder, c.label,
                lambda cid=c.id: self.app.select_concept(cid),
                font_size=8, padx=6, pady=6, anchor="w",
            )
            chip.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="nsew")
            Tooltip(chip, f"{c.label} · {c.note}\n{c.hint or ''}")
            self._concept_chips[c.id] = chip
        for col in range(2):
            self.concept_holder.columnconfigure(col, weight=1, uniform="con")

    def refresh_categories(self) -> None:
        for cat, chip in self._category_chips.items():
            chip.set_selected(cat == self._category)

    # ================================================================== 精度
    def _build_quality(self) -> None:
        Section(self.body, "精度", subtitle="精度越高面越多、导出越慢；预览始终用轻量档").pack(fill="x")
        grid = tk.Frame(self.body, bg=T.panel)
        grid.pack(fill="x", padx=12)
        for i, q in enumerate(quality.QUALITIES):
            chip = Chip(grid, f"{q.label} · {q.face_text()}",
                        lambda qid=q.id: self.app.set_quality(qid), font_size=8,
                        padx=6, pady=6, anchor="w")
            chip.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="nsew")
            Tooltip(chip, f"{q.label}：网格 {q.grid} · {q.note}")
            self._quality_chips[q.id] = chip
        for col in range(2):
            grid.columnconfigure(col, weight=1, uniform="q")

        self.quality_note = auto_wrap(
            tk.Label(self.body, text="", bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     justify="left", anchor="w"), 2)
        self.quality_note.pack(fill="x", padx=16, pady=(6, 0))

    # ================================================================== 工艺
    def _build_craft(self) -> None:
        Section(self.body, "尺寸与工艺", subtitle="导出就是真实尺寸，直接进切片软件").pack(fill="x")

        row = tk.Frame(self.body, bg=T.panel)
        row.pack(fill="x", padx=16)
        tk.Label(row, text="最长边", bg=T.panel, fg=T.text_dim, font=theme.f(9)).pack(side="left")
        self.size_var = tk.StringVar(value="100")
        self.size_entry = ttk.Entry(row, textvariable=self.size_var, width=6,
                                    style="C3D.TEntry", justify="center")
        self.size_entry.pack(side="right")
        tk.Label(row, text="mm", bg=T.panel, fg=T.text_mute, font=theme.f(8)).pack(side="right", padx=4)
        self.size_entry.bind("<Return>", lambda e: self._commit_size())
        self.size_entry.bind("<FocusOut>", lambda e: self._commit_size())

        quick = tk.Frame(self.body, bg=T.panel)
        quick.pack(fill="x", padx=16, pady=(6, 0))
        for mm in (60, 100, 150, 200):
            chip = Chip(quick, f"{mm}", lambda v=mm: self.app.set_size(v), font_size=8,
                        padx=8, pady=4)
            chip.pack(side="left", padx=(0, 4))

        hollow_row = tk.Frame(self.body, bg=T.panel)
        hollow_row.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(hollow_row, text="中空（省料、可透光）", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.hollow_toggle = Toggle(hollow_row, value=False, command=self.app.set_hollow)
        self.hollow_toggle.pack(side="right")

        self.wall_row = self._make_slider_row("壁厚", "中空时的壳体厚度，1.6mm 起更好打印",
                                              0.8, 5.0, 1.8, "wall")

    def _make_slider_row(self, label: str, hint: str, lo: float, hi: float,
                         value: float, key: str) -> SliderRow:
        holder = tk.Frame(self.body, bg=T.panel)
        holder.pack(fill="x", padx=16, pady=(6, 0))
        head = tk.Frame(holder, bg=T.panel)
        head.pack(fill="x")
        lbl = tk.Label(head, text=label, bg=T.panel, fg=T.text_dim, font=theme.f(9))
        lbl.pack(side="left")
        val = tk.Label(head, text=f"{value:.1f}", bg=T.panel, fg=T.accent, font=theme.mono(9))
        val.pack(side="right")
        Tooltip(head, hint)
        slider = Slider(
            holder, lo=lo, hi=hi, value=value, step=0.1,
            command=lambda v, k=key: self._on_slider(k, v),
            release=lambda v, k=key: self._on_slider_commit(k, v),
            bg=T.panel,
        )
        slider.pack(fill="x", pady=(2, 0))
        return SliderRow(holder, lbl, slider, val)

    # ================================================================== 风格
    def _build_style(self) -> None:
        Section(self.body, "风格", subtitle="点一下叠加，再点一次取消").pack(fill="x")
        grid = tk.Frame(self.body, bg=T.panel)
        grid.pack(fill="x", padx=12)
        styles = (
            ("organic", "圆润", "把棱角都揉圆，像手捏的黏土"),
            ("sharp", "硬朗", "切出平面，像宝石或低多边形"),
            ("rough", "粗糙", "表面加噪声肌理，像风化岩石"),
            ("twist", "扭曲", "整体拧一下，更有动势"),
            ("detail", "繁复", "增加装饰细节"),
            ("minimal", "极简", "减少细节，只留大形"),
        )
        for i, (key, label, tip) in enumerate(styles):
            chip = Chip(grid, label, lambda k=key: self.app.toggle_style(k), font_size=8,
                        padx=6, pady=5)
            chip.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky="nsew")
            Tooltip(chip, tip)
            self._style_chips[key] = chip
        for col in range(3):
            grid.columnconfigure(col, weight=1, uniform="st")

        seed_row = tk.Frame(self.body, bg=T.panel)
        seed_row.pack(fill="x", padx=16, pady=(8, 0))
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

    # ================================================================== 参数
    def _build_params(self) -> None:
        Section(self.body, "参数", subtitle="拖动即实时预览；松手后自动切到高清").pack(fill="x")
        self.param_rows: dict[str, SliderRow] = {}
        for key in ("a", "b"):
            spec = self.app.session.spec.param(key)
            self.param_rows[key] = self._make_slider_row(
                spec.label, spec.hint, spec.lo, spec.hi, spec.default, key
            )

    # ================================================================== 经典
    def _build_classic(self) -> None:
        Section(self.body, "经典形态", subtitle="v2 的 8 个参数化形态；图片浮雕 / 轮廓只能在这里用"
                ).pack(fill="x")

        img_row = tk.Frame(self.body, bg=T.panel)
        img_row.pack(fill="x", padx=16)
        Button(img_row, "选择图片…", self.app.pick_image, kind="secondary", padx=8, pady=5,
               font_size=8).pack(side="left")
        Button(img_row, "粘贴", self.app.paste_image, kind="ghost", padx=8, pady=5,
               font_size=8).pack(side="left", padx=4)
        Button(img_row, "清除", self.app.clear_image, kind="ghost", padx=8, pady=5,
               font_size=8).pack(side="left")

        self.img_summary = auto_wrap(
            tk.Label(self.body, text="", bg=T.panel, fg=T.text_mute,
                     font=theme.f(8), justify="left", anchor="w"), 2)
        self.img_summary.pack(fill="x", padx=16, pady=(6, 0))

        self._mode_chips: dict[str, Chip] = {}
        mode_row = tk.Frame(self.body, bg=T.panel)
        mode_row.pack(fill="x", padx=16, pady=(6, 0))
        for key, label, tip in (("silhouette", "轮廓→旋转体", "提取主体轮廓绕轴旋转"),
                                ("relief", "亮度→浮雕", "把明暗转成高度"),
                                ("palette", "仅取配色", "只套用图片主色")):
            chip = Chip(mode_row, label, lambda k=key: self.app.set_image_mode(k),
                        font_size=7, padx=5, pady=4)
            chip.pack(side="left", padx=(0, 3))
            Tooltip(chip, tip)
            self._mode_chips[key] = chip

        grid = tk.Frame(self.body, bg=T.panel)
        grid.pack(fill="x", padx=12)
        from .. import registry

        for i, spec in enumerate(registry.FORMS):
            chip = Chip(
                grid,
                spec.label,
                lambda fid=spec.id: self.app.select_form(fid),
                glyph=spec.glyph,
                font_size=8,
                padx=6,
                pady=6,
                anchor="w",
            )
            chip.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="nsew")
            Tooltip(chip, f"{spec.label} · {spec.desc}\n{spec.detail}")
            self._form_chips[spec.id] = chip
        for col in range(2):
            grid.columnconfigure(col, weight=1, uniform="form")

    # ================================================================== 交互
    def _on_slider(self, key: str, value: float) -> None:
        if key == "wall":
            self.wall_row.value_label.configure(text=f"{value:.1f}")
            self.app.preview_wall(value)
        elif key == "res":
            self.app.preview_res(value)
        else:
            row = self.param_rows.get(key)
            if row:
                row.value_label.configure(text=f"{value:.0f}")
            self.app.preview_param(key, value)

    def _on_slider_commit(self, key: str, value: float) -> None:
        if key == "wall":
            self.app.commit_wall(value)
        elif key == "res":
            self.app.commit_res(value)
        else:
            self.app.commit_param(key, value)

    def show_image_tab(self) -> None:
        """定位到图片区块（经典形态里）。"""
        try:
            self.scroll.ensure_visible(self.img_summary)
        except Exception:  # noqa: BLE001
            pass

    def _commit_seed(self) -> None:
        raw = self.seed_var.get().strip()
        try:
            value = int(float(raw))
        except ValueError:
            self.seed_var.set(str(self.app.session.seed))
            self.app.show_toast("种子只能是 0–999999 的整数", kind="warn")
            return
        self.app.commit_seed(value)

    def _commit_size(self) -> None:
        raw = self.size_var.get().strip().replace("mm", "").replace("毫米", "")
        try:
            value = float(raw)
        except ValueError:
            self.size_var.set(f"{self.app.session.size_mm:.0f}")
            self.app.show_toast("尺寸要填 20–400 之间的毫米数", kind="warn")
            return
        self.app.set_size(value)

    # ================================================================== 刷新
    def refresh(self, session) -> None:
        for fid, chip in self._form_chips.items():
            chip.set_selected(not session.is_concept_mode() and fid == session.form)
        for key, chip in self._mode_chips.items():
            chip.set_selected(key == session.image_mode)
        if session.image is not None:
            self.img_summary.configure(
                text=f"{session.image.name} · {session.image.summary()}", fg=T.text_dim)
        else:
            self.img_summary.configure(text="未选择图片（图片浮雕 / 轮廓需要它）", fg=T.text_mute)
        self.refresh_categories()
        for cid, chip in self._concept_chips.items():
            chip.set_selected(session.is_concept_mode() and cid == session.concept)

        for qid, chip in self._quality_chips.items():
            chip.set_selected(qid == session.quality_id)
        q = session.quality
        self.quality_note.configure(
            text=f"{q.label}：网格 {q.grid}，{q.note}；{q.face_text()}，STL 约 "
                 f"{q.approx_faces * 50 / 1024 / 1024:.0f} MB"
        )

        self.size_var.set(f"{session.size_mm:.0f}")
        self.hollow_toggle.set_value(session.hollow)
        if session.hollow:
            self.wall_row.frame.pack(fill="x", padx=16, pady=(6, 0))
        else:
            self.wall_row.frame.pack_forget()
        self.wall_row.slider.set_value(session.wall_mm)
        self.wall_row.value_label.configure(text=f"{session.wall_mm:.1f}")

        for key, chip in self._style_chips.items():
            active = session.has_style("detail" if key == "minimal" else key)
            if key == "minimal":
                active = float(session.style.get("detail", 0.0)) < -1e-6
            chip.set_selected(active)

        for key in ("a", "b"):
            row = self.param_rows.get(key)
            if not row:
                continue
            label, hint = session.param_labels()[0 if key == "a" else 1]
            row.label.configure(text=label)
            row.slider.set_value(float(session.params.get(key, 50)))
            row.value_label.configure(text=f"{float(session.params.get(key, 50)):.0f}")
        self.seed_var.set(str(session.seed))

        if getattr(self, "_concept_form", None) != session.concept:
            self._concept_form = session.concept
            cat = session.concept_spec.category
            if cat != self._category:
                self._category = cat
                self._rebuild_concepts(cat)
                self.refresh_categories()
                for cid, chip in self._concept_chips.items():
                    chip.set_selected(cid == session.concept)


class RightPanel(tk.Frame):
    def __init__(self, master, app) -> None:
        super().__init__(master, bg=T.panel, width=config.sp(344), bd=0, highlightthickness=0)
        self.pack_propagate(False)
        self.app = app
        self.scroll = ScrollFrame(self, bg=T.panel)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner

        self._build_card()
        hr(self.body)
        self._build_check()
        hr(self.body)
        self._build_palette()
        hr(self.body)
        self._build_export()

    # ------------------------------------------------------------------ 创意卡
    def _build_card(self) -> None:
        header = tk.Frame(self.body, bg=T.panel)
        header.pack(fill="x", padx=16, pady=(14, 4))
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
        self.card_footer.configure(
            text=f"{stats} · seed {idea.extra.get('seed', '')}" if stats
            else f"seed {idea.extra.get('seed', '')}")

    def set_favorite(self, is_fav: bool) -> None:
        self.fav_btn.configure(text="★" if is_fav else "☆",
                               fg=T.warn if is_fav else T.text_dim)

    # ------------------------------------------------------------------ 体检
    def _build_check(self) -> None:
        header = tk.Frame(self.body, bg=T.panel)
        header.pack(fill="x", padx=16, pady=(14, 4))
        self.grade_label = tk.Label(header, text="—", bg=T.panel_alt, fg=T.text_dim,
                                    font=theme.f(8, True), padx=8, pady=2)
        self.grade_label.pack(side="right")
        tk.Label(header, text="打印体检", bg=T.panel, fg=T.text,
                 font=theme.f(10, True)).pack(side="left")
        Button(header, "摆正", self.app.auto_orient, kind="ghost", padx=6, pady=2,
               font_size=8).pack(side="right", padx=6)

        self.check_holder = tk.Frame(self.body, bg=T.panel)
        self.check_holder.pack(fill="x", padx=16)
        self._check_rows: dict[str, tk.Label] = {}
        for key, label in (("water", "水密"), ("size", "尺寸"), ("volume", "体积"),
                           ("overhang", "悬垂"), ("foot", "底部")):
            row = tk.Frame(self.check_holder, bg=T.panel)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     width=4, anchor="w").pack(side="left")
            val = tk.Label(row, text="—", bg=T.panel, fg=T.text_dim, font=theme.f(8), anchor="w")
            val.pack(side="left", fill="x", expand=True)
            self._check_rows[key] = val

        self.check_note = auto_wrap(
            tk.Label(self.body, text="", bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     justify="left", anchor="w"), 2)
        self.check_note.pack(fill="x", padx=16, pady=(6, 0))

    def refresh_check(self, report) -> None:
        if report is None:
            for lbl in self._check_rows.values():
                lbl.configure(text="—", fg=T.text_dim)
            self.grade_label.configure(text="—", bg=T.panel_alt, fg=T.text_dim)
            self.check_note.configure(text="")
            return
        ok_color = T.ok if report.watertight else T.warn
        self._check_rows["water"].configure(
            text=f"{'封闭 ✓' if report.watertight else '有缺陷 ✗'}"
                 f"（非流形 {report.nonmanifold_edges}）", fg=ok_color)
        self._check_rows["size"].configure(text=report.bbox_text(), fg=T.text_dim)
        self._check_rows["volume"].configure(
            text=f"{report.volume_cm3:.1f} cm³ · PLA {report.weight_pla_g:.0f} g", fg=T.text_dim)
        self._check_rows["overhang"].configure(
            text=f"{report.overhang_ratio * 100:.0f}%",
            fg=T.ok if report.overhang_ratio < 0.18 else T.warn)
        self._check_rows["foot"].configure(
            text=f"{report.footprint_ratio * 100:.0f}% 接触",
            fg=T.ok if report.footprint_ratio >= 0.04 else T.warn)
        color = {"优": T.ok, "良": T.accent, "需注意": T.warn}.get(report.grade, T.text_dim)
        self.grade_label.configure(text=report.grade, bg=T.panel_alt, fg=color)
        note = next((n for n in report.notes if "建议" in n or "需要" in n),
                    report.notes[0] if report.notes else "")
        self.check_note.configure(text=note)

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
        Section(self.body, "导出", subtitle="STL/3MF 直接进切片软件；创意卡方便归档").pack(fill="x")

        formats = tk.Frame(self.body, bg=T.panel)
        formats.pack(fill="x", padx=16)
        self._format_chips: dict[str, Chip] = {}
        for i, (key, label) in enumerate(
            (("stl", "STL"), ("three_mf", "3MF"), ("obj", "OBJ"),
             ("png", "预览图"), ("card", "创意卡"), ("report", "报告"))
        ):
            chip = Chip(formats, label, lambda k=key: self.app.toggle_export_format(k),
                        selected=True, font_size=8, padx=6, pady=5, anchor="w")
            chip.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky="nsew")
            self._format_chips[key] = chip
        for col in range(3):
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
            lbl.bind("<Leave>", lambda e, w=lbl: w.configure(fg=T.text_mute))
