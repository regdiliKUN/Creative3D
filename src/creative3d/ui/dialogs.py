"""ui.dialogs —— 模态对话框：导出选项 / 快捷键帮助 / 错误详情 / 首次引导。

统一风格：深色、无系统标题栏装饰、居中于父窗口、Esc 关闭、Enter 确认。
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .. import config, shortcut
from . import theme
from .widgets import Button, Chip, Segmented, Toggle, Tooltip

T = config.THEME


class Modal(tk.Toplevel):
    def __init__(self, parent, title: str, *, width: int = 520, height: int = 0) -> None:
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        self.title(title)
        self.configure(bg=T.panel)
        self.resizable(False, False)
        self._parent = parent
        self._result = None
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda e: self.cancel())

        self.header = tk.Frame(self, bg=T.panel)
        self.header.pack(fill="x", padx=22, pady=(18, 6))
        tk.Label(self.header, text=title, bg=T.panel, fg=T.text,
                 font=theme.f(13, True)).pack(anchor="w")
        self.subtitle = tk.Label(self.header, text="", bg=T.panel, fg=T.text_mute,
                                 font=theme.f(9), justify="left", wraplength=width - 60)
        self.subtitle.pack(anchor="w", pady=(4, 0))

        self.body = tk.Frame(self, bg=T.panel)
        self.body.pack(fill="both", expand=True, padx=22, pady=6)

        self.footer = tk.Frame(self, bg=T.panel)
        self.footer.pack(fill="x", padx=22, pady=(10, 18))

        self._width = width
        self._height = height

    def set_subtitle(self, text: str) -> None:
        self.subtitle.configure(text=text)

    def show_modal(self):
        self.update_idletasks()
        w = max(self._width, self.winfo_reqwidth())
        h = self._height or self.winfo_reqheight()
        px = self._parent.winfo_rootx()
        py = self._parent.winfo_rooty()
        pw = self._parent.winfo_width()
        ph = self._parent.winfo_height()
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.deiconify()
        self.grab_set()
        self.focus_force()
        self.wait_window()
        return self._result

    def cancel(self) -> None:
        self._result = None
        self.destroy()

    def confirm(self, result=None) -> None:
        self._result = result
        self.destroy()


class ExportDialog(Modal):
    def __init__(self, parent, *, defaults: dict, out_dir: Path, title: str, stats: str,
                 report=None) -> None:
        super().__init__(parent, "导出", width=660)
        self.set_subtitle(f"《{title}》 · {stats}")

        self.options = dict(defaults)
        self.out_dir = out_dir
        self.report = report

        columns = tk.Frame(self.body, bg=T.panel)
        columns.pack(fill="both", expand=True)
        left = tk.Frame(columns, bg=T.panel)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(columns, bg=T.panel)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0))

        # ---------------- 左：导出内容
        tk.Label(left, text="导出内容", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(anchor="w", pady=(4, 6))
        row = tk.Frame(left, bg=T.panel)
        row.pack(fill="x")
        self._chips: dict[str, Chip] = {}
        for i, (key, label, tip) in enumerate(
            (("stl", "STL 模型", "最通用的 3D 打印格式，二进制、单位毫米"),
             ("three_mf", "3MF 模型", "带单位与对象名，切片软件导入不用猜尺寸"),
             ("obj", "OBJ 模型", "导入 Blender / Rhino 二次加工"),
             ("png", "预览图", "高分辨率渲染图，2× 超采样抗锯齿"),
             ("card", "创意卡", "Markdown 文案，含参数与打印建议"),
             ("report", "体检报告", "纯文本，含体积/重量/悬垂与摆放建议"))
        ):
            chip = Chip(row, label, lambda k=key: self._toggle(k),
                        selected=bool(self.options.get(key)), font_size=9, padx=8, pady=6,
                        anchor="w")
            chip.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="nsew")
            Tooltip(chip, tip)
            self._chips[key] = chip
        for col in range(2):
            row.columnconfigure(col, weight=1, uniform="fmt")

        tk.Label(left, text="预览图尺寸", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(anchor="w", pady=(14, 6))
        self.size_seg = Segmented(left, ["1200×900", "1400×1050", "2000×1500"], self._on_size)
        self.size_seg.pack(anchor="w")
        current = (int(self.options.get("png_width", 1400)), int(self.options.get("png_height", 1050)))
        index = {1200: 0, 1400: 1, 2000: 2}.get(current[0], 1)
        self.size_seg.select(index, notify=False)
        self._size_map = [(1200, 900), (1400, 1050), (2000, 1500)]

        transparent_row = tk.Frame(left, bg=T.panel)
        transparent_row.pack(fill="x", pady=(14, 0))
        tk.Label(transparent_row, text="透明背景", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.transparent = Toggle(transparent_row, value=bool(self.options.get("transparent")),
                                  command=self._on_transparent)
        self.transparent.pack(side="right")

        # ---------------- 右：精度与工艺
        tk.Label(right, text="精度", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(anchor="w", pady=(4, 6))
        qrow = tk.Frame(right, bg=T.panel)
        qrow.pack(fill="x")
        self._quality_chips: dict[str, Chip] = {}
        from .. import quality as _quality

        for i, q in enumerate(_quality.QUALITIES):
            chip = Chip(qrow, f"{q.label} · {q.face_text()}",
                        lambda qid=q.id: self._set_quality(qid), font_size=8, padx=8, pady=5,
                        selected=q.id == self.options.get("quality"), anchor="w")
            chip.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="nsew")
            Tooltip(chip, f"{q.label}：网格 {q.grid} · {q.note}")
            self._quality_chips[q.id] = chip
        for col in range(2):
            qrow.columnconfigure(col, weight=1, uniform="q")

        size_row = tk.Frame(right, bg=T.panel)
        size_row.pack(fill="x", pady=(14, 0))
        tk.Label(size_row, text="目标尺寸", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.size_var = tk.StringVar(value=f"{float(self.options.get('size_mm', 100)):.0f}")
        self.size_entry = ttk.Entry(size_row, textvariable=self.size_var, width=6,
                                    style="C3D.TEntry", justify="center")
        self.size_entry.pack(side="right")
        tk.Label(size_row, text="mm", bg=T.panel, fg=T.text_mute,
                 font=theme.f(8)).pack(side="right", padx=4)

        hollow_row = tk.Frame(right, bg=T.panel)
        hollow_row.pack(fill="x", pady=(10, 0))
        tk.Label(hollow_row, text="中空（省料）", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.hollow = Toggle(hollow_row, value=bool(self.options.get("hollow")),
                             command=self._on_hollow)
        self.hollow.pack(side="right")

        orient_row = tk.Frame(right, bg=T.panel)
        orient_row.pack(fill="x", pady=(10, 0))
        tk.Label(orient_row, text="自动摆正（少支撑）", bg=T.panel, fg=T.text_dim,
                 font=theme.f(9)).pack(side="left")
        self.orient = Toggle(orient_row, value=bool(self.options.get("auto_orient")),
                             command=self._on_orient)
        self.orient.pack(side="right")

        if report is not None:
            box = tk.Frame(right, bg=T.panel_alt, highlightthickness=1, highlightbackground=T.line)
            box.pack(fill="x", pady=(14, 0))
            tk.Label(box, text=f"打印体检 · {report.grade}", bg=T.panel_alt,
                     fg=T.ok if report.grade == "优" else T.text, font=theme.f(9, True),
                     anchor="w").pack(fill="x", padx=10, pady=(8, 2))
            tk.Label(box, text=report.summary(), bg=T.panel_alt, fg=T.text_dim,
                     font=theme.f(8), anchor="w", justify="left",
                     wraplength=250).pack(fill="x", padx=10, pady=(0, 8))

        # ---------------- 底部：目录
        dir_row = tk.Frame(self.footer, bg=T.panel)
        dir_row.pack(side="left", fill="x", expand=True)
        self.dir_label = tk.Label(dir_row, text=str(out_dir), bg=T.panel_alt, fg=T.text_dim,
                                  font=theme.f(8), anchor="w", padx=10, pady=7)
        self.dir_label.pack(side="left", fill="x", expand=True)
        Button(dir_row, "更改…", self._pick_dir, kind="ghost", padx=10, pady=6,
               font_size=8).pack(side="left", padx=(6, 0))

        Button(self.footer, "取消", self.cancel, kind="ghost", padx=16, pady=9).pack(side="right")
        Button(self.footer, "开始导出", self._ok, kind="primary", padx=18, pady=9,
               font_size=10, bold=True).pack(side="right", padx=8)
        self.bind("<Return>", lambda e: self._ok())

    # ------------------------------------------------------------------
    def _toggle(self, key: str) -> None:
        self.options[key] = not self.options.get(key)
        self._chips[key].set_selected(bool(self.options[key]))

    def _set_quality(self, quality_id: str) -> None:
        self.options["quality"] = quality_id
        for qid, chip in self._quality_chips.items():
            chip.set_selected(qid == quality_id)

    def _on_size(self, index: int) -> None:
        w, h = self._size_map[max(0, min(len(self._size_map) - 1, index))]
        self.options["png_width"] = w
        self.options["png_height"] = h

    def _on_transparent(self, value: bool) -> None:
        self.options["transparent"] = bool(value)

    def _on_hollow(self, value: bool) -> None:
        self.options["hollow"] = bool(value)

    def _on_orient(self, value: bool) -> None:
        self.options["auto_orient"] = bool(value)

    def _pick_dir(self) -> None:
        folder = filedialog.askdirectory(parent=self, initialdir=str(self.out_dir),
                                         title="选择导出目录")
        if folder:
            self.out_dir = Path(folder)
            self.dir_label.configure(text=str(self.out_dir))

    def _ok(self) -> None:
        if not any(self.options.get(k) for k in
                   ("stl", "three_mf", "obj", "png", "card", "report")):
            self.set_subtitle("至少选择一种导出内容。")
            return
        raw = self.size_var.get().strip().replace("mm", "").replace("毫米", "")
        try:
            size = float(raw)
        except ValueError:
            self.set_subtitle("目标尺寸要填 20–400 之间的毫米数。")
            return
        self.options["size_mm"] = max(20.0, min(400.0, size))
        self.options["supersample"] = 2
        self.confirm({"options": self.options, "out_dir": self.out_dir})


class HelpDialog(Modal):
    SHORTCUTS = (
        ("拖动", "旋转视角"),
        ("滚轮", "缩放"),
        ("双击 / R", "复位视角"),
        ("空格", "暂停 / 继续自转"),
        ("N", "换一组随机灵感"),
        ("W", "切换线框"),
        ("Ctrl+Z / Ctrl+Y", "撤销 / 重做"),
        ("Ctrl+V", "粘贴剪贴板图片"),
        ("Ctrl+E", "打开导出窗口"),
        ("Ctrl+S", "保存设置"),
        ("F1", "打开本帮助"),
        ("Esc", "关闭对话框"),
    )

    def __init__(self, parent) -> None:
        super().__init__(parent, "使用帮助", width=580)
        self.set_subtitle("全部功能都在本地离线运行，不会联网。")

        columns = tk.Frame(self.body, bg=T.panel)
        columns.pack(fill="both", expand=True)

        left = tk.Frame(columns, bg=T.panel)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="快捷键", bg=T.panel, fg=T.text, font=theme.f(10, True)).pack(anchor="w")
        for key, desc in self.SHORTCUTS:
            row = tk.Frame(left, bg=T.panel)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key, bg=T.panel_alt, fg=T.text_dim, font=theme.mono(8),
                     padx=6, pady=2, width=13, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=T.panel, fg=T.text_mute, font=theme.f(8)).pack(
                side="left", padx=8)

        right = tk.Frame(columns, bg=T.panel)
        right.pack(side="left", fill="both", expand=True, padx=(18, 0))
        tk.Label(right, text="三步上手", bg=T.panel, fg=T.text,
                 font=theme.f(10, True)).pack(anchor="w")
        steps = (
            ("1", "写一句话（猫 / 齿轮 / 城堡 / 花瓶…），或从模型库点一个。"),
            ("2", "选精度档位、填目标尺寸；风格开关随手叠。"),
            ("3", "看右下角打印体检，必要时点「摆正」，然后导出。"),
        )
        for num, text in steps:
            row = tk.Frame(right, bg=T.panel)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=num, bg=T.accent_soft, fg=T.accent, font=theme.f(8, True),
                     width=2).pack(side="left", anchor="n")
            tk.Label(row, text=text, bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     wraplength=230, justify="left").pack(side="left", padx=8, anchor="w")

        tip = tk.Label(
            self.body,
            text="小技巧：同一个种子永远生成同样的模型与文案，记下 seed 就能复现。\n"
                 "精度档位只影响面数与导出体积，不改变造型本身。",
            bg=T.panel_alt, fg=T.text_dim, font=theme.f(8), justify="left",
            padx=12, pady=10, wraplength=500,
        )
        tip.pack(fill="x", pady=(14, 0))

        Button(self.footer, "知道了", lambda: self.confirm(True), kind="primary",
               padx=18, pady=9).pack(side="right")


class ErrorDialog(Modal):
    def __init__(self, parent, *, title: str, message: str, hint: str = "", detail: str = "") -> None:
        super().__init__(parent, title, width=520)
        tk.Label(self.body, text=message, bg=T.panel, fg=T.text, font=theme.f(10),
                 wraplength=470, justify="left", anchor="w").pack(fill="x", pady=(4, 0))
        if hint:
            tk.Label(self.body, text=hint, bg=T.panel, fg=T.text_dim, font=theme.f(9),
                     wraplength=470, justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        self._detail = detail
        self.detail_frame = tk.Frame(self.body, bg=T.panel)
        self.detail_text = tk.Text(self.detail_frame, height=5, width=44, bg=T.panel_alt, fg=T.text_mute,
                                   relief="flat", bd=0, highlightthickness=0, font=theme.mono(8),
                                   wrap="word", padx=8, pady=6)
        self.detail_text.insert("1.0", detail[:2000] if detail else "（无更多信息）")
        self.detail_text.configure(state="disabled")
        self.detail_text.pack(fill="both", expand=True)

        self.toggle_btn = Button(self.body, "查看技术细节", self._toggle_detail, kind="ghost",
                                 padx=0, pady=4, font_size=8)
        self.toggle_btn.pack(anchor="w", pady=(8, 0))

        Button(self.footer, "复制详情", self._copy, kind="ghost", padx=14, pady=8).pack(side="left")
        Button(self.footer, "关闭", lambda: self.confirm(False), kind="primary",
               padx=18, pady=9).pack(side="right")

    def _toggle_detail(self) -> None:
        if self.detail_frame.winfo_ismapped():
            self.detail_frame.pack_forget()
            self.toggle_btn.set_text("查看技术细节")
        else:
            self.detail_frame.pack(fill="both", expand=True, pady=(8, 0))
            self.toggle_btn.set_text("收起技术细节")

    def _copy(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(self._detail or "")
            self.toggle_btn.set_text("已复制")
        except tk.TclError:
            pass


class WelcomeDialog(Modal):
    def __init__(self, parent, *, on_shortcut) -> None:
        super().__init__(parent, "欢迎", width=560)
        self.set_subtitle("一个完全离线的参数化 3D 创意工具：形态、文案、导出，一条龙。")
        self._on_shortcut = on_shortcut

        items = (
            ("◍", "8 个形态族", "花瓶 / 晶体 / 纽结 / 塔 / 地形 / 海螺 / 浮雕 / 轮廓"),
            ("✦", "多模态输入", "一句话描述，或拖入一张图片，自动转成参数与配色"),
            ("⤓", "一键导出", "STL 直接切片打印，另附高分辨率预览图与创意卡"),
        )
        for glyph, title, desc in items:
            row = tk.Frame(self.body, bg=T.panel)
            row.pack(fill="x", pady=6)
            tk.Label(row, text=glyph, bg=T.accent_soft, fg=T.accent, font=theme.f(14),
                     width=3, pady=6).pack(side="left", anchor="n")
            col = tk.Frame(row, bg=T.panel)
            col.pack(side="left", fill="x", expand=True, padx=10)
            tk.Label(col, text=title, bg=T.panel, fg=T.text, font=theme.f(10, True)).pack(anchor="w")
            tk.Label(col, text=desc, bg=T.panel, fg=T.text_mute, font=theme.f(8),
                     wraplength=420, justify="left").pack(anchor="w")

        self.shortcut_btn = Button(self.body, "在桌面创建快捷方式", self._make_shortcut,
                                   kind="ghost", padx=0, pady=4, font_size=8)
        self.shortcut_btn.pack(anchor="w", pady=(10, 0))

        Button(self.footer, "开始创作", lambda: self.confirm("start"), kind="primary",
               padx=20, pady=10, font_size=10, bold=True).pack(side="right")

    def _make_shortcut(self) -> None:
        try:
            path = shortcut.create_desktop_shortcut(force=True)
            self.shortcut_btn.set_text(f"已创建：{path.parent}")
        except Exception as exc:  # noqa: BLE001
            self.shortcut_btn.set_text(f"创建失败：{exc}")
