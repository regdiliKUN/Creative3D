"""ui.mainwindow —— 主窗口：把 Session / Viewport / 面板 / 导出 / 快捷键串起来。

事件流只走一条线：
    用户操作 → Session 方法 → Session 事件 → 本类的 _on_* 回调 → 刷新界面
因此任何入口（滑杆、随机、提示词、图片、撤销）都得到完全一致的行为。
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image

from .. import (config, concepts, exporter, generators, imaging, nlp, printcheck,
                quality, registry, render, shortcut)
from ..errors import AppError, ExportError
from ..jobs import Jobs
from ..mesh import Mesh
from ..session import Session
from ..settings import Settings
from . import dialogs, theme
from .panels import LeftPanel, RightPanel
from .viewport import Viewport
from .widgets import Button, IconButton, ProgressBar, Toast, Tooltip

T = config.THEME
log = logging.getLogger("creative3d.ui")


class MainWindow(tk.Tk):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.session = Session(settings)
        self.jobs = Jobs(self)
        self.export_formats = dict(self.session.export_options_defaults())
        self._mesh_token = None
        self._final_after: str | None = None
        self._idea_after: str | None = None
        self._mesh_key = ""
        self._export_token = None
        self._toast: Toast | None = None

        self.title(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.configure(bg=T.bg)
        self.minsize(config.sp(config.MIN_WINDOW_SIZE[0]), config.sp(config.MIN_WINDOW_SIZE[1]))
        self._restore_geometry()
        theme.init_root(self)
        self._set_icon()
        self.report_callback_exception = self._on_tk_exception  # type: ignore[assignment]

        self._build_topbar()
        self._build_statusbar()
        self._build_content()
        self._bind_keys()
        self._wire_session()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.refresh_all()
        self._initial_mesh()

        if not settings.get("welcome_seen") and not os.environ.get("C3D_NO_WELCOME"):
            self.after(500, self._show_welcome)

    # ================================================================== 布局
    def _restore_geometry(self) -> None:
        size = self.settings.get("window_size") or list(config.DEFAULT_WINDOW_SIZE)
        w, h = int(size[0]), int(size[1])
        if w < config.sp(1100) or h < config.sp(700):
            # 旧版本（低 DPI）存下来的尺寸在高分屏上会小得没法用
            w, h = config.sp(config.DEFAULT_WINDOW_SIZE[0]), config.sp(config.DEFAULT_WINDOW_SIZE[1])
        w = min(w, self.winfo_screenwidth() - 40)
        h = min(h, self.winfo_screenheight() - 80)
        pos = self.settings.get("window_pos")
        if pos and len(pos) == 2:
            self.geometry(f"{w}x{h}+{int(pos[0])}+{int(pos[1])}")
        else:
            x = max(0, (self.winfo_screenwidth() - w) // 2)
            y = max(0, (self.winfo_screenheight() - h) // 2 - 20)
            self.geometry(f"{w}x{h}+{x}+{y}")

    def _set_icon(self) -> None:
        ico = config.resource_path("assets", "icon.ico")
        try:
            if ico.exists():
                self.iconbitmap(str(ico))
        except tk.TclError:
            pass

    def _build_topbar(self) -> None:
        # 不用固定高度 + pack_propagate(False)：在高倍屏（tk scaling 2.0+）下
        # 固定像素高度会把标题和图标压扁。改成由内容自然撑开高度。
        bar = tk.Frame(self, bg=T.bg_alt)
        bar.pack(fill="x", side="top")
        inner = tk.Frame(bar, bg=T.bg_alt)
        inner.pack(fill="x", pady=11)

        mark = tk.Label(inner, text="◈", bg=T.accent_soft, fg=T.accent, font=theme.f(17),
                        width=3, pady=4)
        mark.pack(side="left", padx=(16, 10))
        title_box = tk.Frame(inner, bg=T.bg_alt)
        title_box.pack(side="left")
        tk.Label(title_box, text=config.APP_NAME, bg=T.bg_alt, fg=T.text,
                 font=theme.f(12, True)).pack(anchor="w")
        tk.Label(title_box, text=f"{config.APP_TAGLINE} · v{config.APP_VERSION}",
                 bg=T.bg_alt, fg=T.text_mute, font=theme.f(8)).pack(anchor="w")

        right = tk.Frame(inner, bg=T.bg_alt)
        right.pack(side="right", padx=16)
        Button(right, "导出", self.open_export_dialog, kind="primary", padx=16, pady=8,
               font_size=9, bold=True).pack(side="right", padx=(8, 0))
        Button(right, "帮助", self.show_help, kind="ghost", padx=10, pady=8,
               font_size=9).pack(side="right", padx=4)
        Button(right, "更多", self._show_more_menu, kind="ghost", padx=10, pady=8,
               font_size=9).pack(side="right", padx=4)
        Button(right, "撤销", self.undo, kind="ghost", padx=8, pady=8,
               font_size=9).pack(side="right", padx=2)
        self.redo_btn = Button(right, "重做", self.redo, kind="ghost", padx=8, pady=8,
                               font_size=9)
        self.redo_btn.pack(side="right", padx=2)

        tk.Frame(self, bg=T.line, height=1).pack(fill="x", side="top")

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=T.bg_alt)
        bar.pack(fill="x", side="bottom")
        inner = tk.Frame(bar, bg=T.bg_alt)
        inner.pack(fill="x", pady=6)
        tk.Frame(self, bg=T.line, height=1).pack(fill="x", side="bottom")

        self.status_label = tk.Label(inner, text="就绪", bg=T.bg_alt, fg=T.text_mute,
                                     font=theme.f(8), anchor="w")
        self.status_label.pack(side="left", padx=14)

        right = tk.Frame(inner, bg=T.bg_alt)
        right.pack(side="right", padx=12)
        self.cancel_btn = tk.Label(right, text="取消", bg=T.bg_alt, fg=T.err, font=theme.f(8),
                                   cursor="hand2", padx=8)
        self.cancel_btn.bind("<Button-1>", lambda e: self._cancel_export())
        self.progress = ProgressBar(right, width=150, height=4, bg=T.bg_alt)
        self.progress.pack(side="right", padx=(0, 8))
        self.progress_label = tk.Label(right, text="", bg=T.bg_alt, fg=T.text_dim,
                                       font=theme.f(8))
        self.progress_label.pack(side="right", padx=6)
        self.hint_label = tk.Label(right, text="", bg=T.bg_alt, fg=T.text_mute, font=theme.f(8))
        self.hint_label.pack(side="right", padx=10)
        self._set_busy_ui(False)

    def _build_content(self) -> None:
        main = tk.Frame(self, bg=T.bg)
        main.pack(fill="both", expand=True)

        self.left = LeftPanel(main, self)
        self.left.pack(side="left", fill="y")
        tk.Frame(main, bg=T.line, width=1).pack(side="left", fill="y")

        self.right = RightPanel(main, self)
        self.right.pack(side="right", fill="y")
        tk.Frame(main, bg=T.line, width=1).pack(side="right", fill="y")

        center = tk.Frame(main, bg=T.bg)
        center.pack(side="left", fill="both", expand=True)
        self.viewport = Viewport(center, on_state=self.refresh_all)
        self.viewport.pack(fill="both", expand=True)
        self.viewport.set_image_request_handler(self.pick_image)
        self.viewport.chip_rotate.set_selected(bool(self.settings.get("auto_rotate", True)))
        self.viewport.auto_rotate = bool(self.settings.get("auto_rotate", True))
        self.viewport.wire = bool(self.settings.get("wire", False))
        self.viewport.chip_wire.set_selected(self.viewport.wire)
        self.viewport.shadow = bool(self.settings.get("shadow", True))
        self.viewport.chip_shadow.set_selected(self.viewport.shadow)

        self._toast = Toast(center)

    def _bind_keys(self) -> None:
        self.bind_all("<space>", self._key_space)
        self.bind_all("<Key-r>", lambda e: self.viewport.reset_camera())
        self.bind_all("<Key-n>", lambda e: self.randomize())
        self.bind_all("<Key-w>", lambda e: self.viewport.toggle_wire())
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())
        self.bind_all("<Control-e>", lambda e: self.open_export_dialog())
        self.bind_all("<Control-s>", lambda e: self.save_settings())
        self.bind_all("<Control-v>", lambda e: self.paste_image())
        self.bind_all("<F1>", lambda e: self.show_help())

    # ================================================================== 订阅
    def _wire_session(self) -> None:
        self.session.subscribe("state", self._on_state)
        self.session.subscribe("mesh", self._on_mesh)
        self.session.subscribe("palette", self._on_palette)
        self.session.subscribe("idea", self._on_idea)
        self.session.subscribe("quality", self._on_quality)

    def _on_state(self, **_kw) -> None:
        self.refresh_all()

    def _on_quality(self, **_kw) -> None:
        self._on_mesh()
        self.refresh_all()

    def _on_palette(self, **_kw) -> None:
        self.viewport.set_palette(self.session.palette())
        self.right.refresh_palette(self.session)

    def _on_idea(self, **_kw) -> None:
        if self._idea_after:
            try:
                self.after_cancel(self._idea_after)
            except tk.TclError:
                pass
        self._idea_after = self.after(160, self._refresh_idea)

    def _refresh_idea(self) -> None:
        self._idea_after = None
        try:
            mesh = self.session.generate_mesh(preview=True)
            stats = mesh.stats_text()
        except Exception:  # noqa: BLE001
            stats = ""
        idea = self.session.idea(stats)
        self.right.refresh_idea(idea, stats=stats)
        self.right.set_favorite(self.settings.is_favorite(self.session.favorite_key()))

    def _on_mesh(self, **_kw) -> None:
        key = self.session.snapshot().key()
        self._mesh_key = key
        try:
            mesh = self.session.generate_mesh(preview=True)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc, title="生成失败")
            return
        self.viewport.set_mesh(mesh)
        self.viewport.set_palette(self.session.palette())
        self._update_titles()
        self._schedule_final(key)

    def _schedule_final(self, key: str) -> None:
        if self._final_after:
            try:
                self.after_cancel(self._final_after)
            except tk.TclError:
                pass
        self._final_after = self.after(520, lambda: self._build_final(key))

    def _build_final(self, key: str) -> None:
        self._final_after = None
        if key != self._mesh_key:
            return
        self.viewport.set_busy(True)

        def work(token=None) -> Mesh:
            return self.session.generate_mesh(preview=False)

        def done(mesh: Mesh) -> None:
            self.viewport.set_busy(False)
            if self.session.snapshot().key() != key:
                return
            self.viewport.set_mesh(mesh)
            self._update_titles()
            stats = mesh.stats_text()
            self.right.refresh_idea(self.session.idea(stats), stats=stats)
            self._refresh_check(mesh)

        def failed(_exc) -> None:
            self.viewport.set_busy(False)

        self.jobs.submit(work, on_done=done, on_error=failed, name="final-mesh")

    def _refresh_check(self, mesh: Mesh | None) -> None:
        if mesh is None:
            self.right.refresh_check(None)
            return
        try:
            self.right.refresh_check(printcheck.analyze(mesh))
        except Exception:  # noqa: BLE001 - 体检失败不该影响主流程
            log.exception("打印体检失败")
            self.right.refresh_check(None)

    # ================================================================== 刷新
    def refresh_all(self) -> None:
        session = self.session
        self.left.refresh(session)
        self.right.refresh_palette(session)
        self.right.refresh_formats(self.export_formats)
        self.viewport.set_palette(session.palette())
        self._update_titles()
        self.redo_btn.set_enabled(session.can_redo())
        if session.is_concept_mode():
            self.status_label.configure(
                text=f"{session.title()} · {session.quality.label} · {session.size_mm:.0f}mm"
                     f" · seed {session.seed}"
            )
        else:
            self.status_label.configure(
                text=f"{session.title()} · seed {session.seed} · {session.palette_label()}"
            )

    def _update_titles(self) -> None:
        session = self.session
        self.viewport.set_titles(f"{session.glyph()}  {session.title()}", session.subtitle())
        if not session.is_concept_mode() and session.spec.needs_image and session.image is None:
            self.hint_label.configure(text="提示：这个形态用图片效果最好")
        elif session.is_concept_mode() and session.hollow and session.wall_mm < 1.2:
            self.hint_label.configure(text="提示：壁厚低于 1.2mm 打印容易破")
        else:
            self.hint_label.configure(text="")

    def _initial_mesh(self) -> None:
        self._on_mesh()

    # ================================================================== 操作
    def select_concept(self, concept_id: str) -> None:
        concept = concepts.get(concept_id)
        self.session.set_concept(concept_id)
        self.show_toast(f"已切到「{concept.label}」· {concept.note}", kind="ok")

    def set_quality(self, quality_id: str) -> None:
        q = quality.get(quality_id)
        self.session.set_quality(quality_id)
        self.show_toast(f"精度：{q.label} · {q.face_text()}", kind="ok")

    def set_size(self, mm: float) -> None:
        self.session.set_size(mm)
        self.show_toast(f"目标尺寸：{self.session.size_mm:.0f} mm", kind="ok")

    def set_hollow(self, hollow: bool) -> None:
        self.session.set_hollow(bool(hollow))
        if self.session.hollow:
            self.show_toast(f"已中空：壁厚 {self.session.wall_mm:.1f} mm", kind="ok")
        else:
            self.show_toast("已改为实心")

    def toggle_style(self, key: str) -> None:
        if key == "minimal":
            self.session.set_style("detail", -0.5 if not self.session.has_style("detail")
                                   or self.session.style.get("detail", 0) >= 0 else 0.0)
        else:
            self.session.toggle_style(key)
        self.left.refresh(self.session)

    def preview_wall(self, mm: float) -> None:
        self.session.set_wall(mm, record=False)

    def commit_wall(self, mm: float) -> None:
        self.session.set_wall(mm, record=True)

    def auto_orient(self) -> None:
        try:
            mesh = self.session.generate_mesh(preview=False)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc, title="生成失败")
            return
        self.viewport.set_busy(True)
        self.update_idletasks()

        def work(token=None):
            return printcheck.auto_orient(mesh)

        def done(result) -> None:
            self.viewport.set_busy(False)
            oriented, report, label = result
            self.viewport.set_mesh(oriented)
            self.right.refresh_check(report)
            self.show_toast(f"已摆正：{label} · 悬垂降到 {report.overhang_ratio * 100:.0f}%",
                            kind="ok", duration=3200)

        def failed(_exc) -> None:
            self.viewport.set_busy(False)

        self.jobs.submit(work, on_done=done, on_error=failed, name="orient")

    def select_form(self, form_id: str) -> None:
        self.session.set_mode("classic")
        self.session.set_form(form_id)
        if registry.get(form_id).needs_image and self.session.image is None:
            self.left.show_image_tab()
            self.show_toast("这个形态更适合用图片驱动：选一张图，或直接用当前的程序化版本", kind="warn")

    def preview_param(self, key: str, value: float) -> None:
        self.session.set_param(key, value, record=False)

    def commit_param(self, key: str, value: float) -> None:
        self.session.set_param(key, value, record=True)

    def preview_res(self, value: float) -> None:
        self.session.set_res(value, record=False)

    def commit_res(self, value: float) -> None:
        self.session.set_res(value, record=True)

    def commit_seed(self, value: int) -> None:
        self.session.set_seed(value)

    def apply_preset(self, preset) -> None:
        self.session.set_param("a", preset.a)
        self.session.set_param("b", preset.b)
        self.show_toast(f"已应用预设「{preset.name}」：{preset.note}")

    def randomize(self) -> None:
        self.session.randomize()
        self.show_toast("换了一组随机灵感 · 按 Ctrl+Z 可以回到上一组", kind="ok")

    def undo(self) -> None:
        if self.session.undo():
            self.show_toast("已撤销")
        else:
            self.show_toast("没有可撤销的操作", kind="warn")

    def redo(self) -> None:
        if self.session.redo():
            self.show_toast("已重做")
        else:
            self.show_toast("没有可重做的操作", kind="warn")

    def select_palette(self, palette_id: str) -> None:
        if palette_id == "image" and self.session.image is None:
            self.show_toast("先选一张图片，才能套用图片主色", kind="warn")
            return
        self.session.set_palette(palette_id)

    def toggle_favorite(self) -> None:
        key = self.session.favorite_key()
        entry = {
            "key": key,
            "form": self.session.form,
            "seed": self.session.seed,
            "params": dict(self.session.params),
            "res": self.session.res,
            "palette": self.session.palette_id,
            "title": self.session.idea().title,
        }
        added = self.settings.toggle_favorite(entry)
        self.settings.save()
        self.right.set_favorite(added)
        self.show_toast("已收藏" if added else "已取消收藏", kind="ok")

    # ---------------------------------------------------------------- 文本
    def parse_prompt(self) -> None:
        text = self.left.prompt_text()
        if not text:
            self.left.set_prompt_note(nlp.keywords_help(), kind="muted")
            return
        result = nlp.parse(text)
        if not result.understood:
            self.left.set_prompt_note(
                "没读懂这句话，已把它写进创意卡的「灵感来源」。\n" + nlp.keywords_help(), kind="warn"
            )
            self.session.set_prompt_text(text)
            return
        self.session.set_prompt_text(text)
        self.session.apply_prompt_result(result)
        note = result.summary()
        if result.seed is None:
            note += " · 种子沿用当前值"
        self.left.set_prompt_note(note, kind="ok")
        self.show_toast(f"已生成「{self.session.title()}」· 可继续用滑杆微调", kind="ok")

    # ---------------------------------------------------------------- 图片
    def pick_image(self) -> None:
        types = [
            ("图片", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
            ("所有文件", "*.*"),
        ]
        path = filedialog.askopenfilename(parent=self, title="选择参考图片", filetypes=types)
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str) -> None:
        try:
            self.session.set_image(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc, title="图片无法使用")
            return
        self.left.show_image_tab()
        self._after_image_loaded(Path(path).name)

    def paste_image(self) -> None:
        img = imaging.from_clipboard()
        if img is None:
            self.show_toast("剪贴板里没有图片", kind="warn")
            return
        try:
            self.session.set_image_from_pil(img)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc, title="图片无法使用")
            return
        self.left.show_image_tab()
        self._after_image_loaded("剪贴板图片")

    def _after_image_loaded(self, name: str) -> None:
        if self.session.image is None:
            return
        feats = self.session.image
        self.show_toast(f"已读取 {name}：{feats.summary()}", kind="ok", duration=3200)
        # 依当前模式决定用图片做什么
        if self.session.image_mode == "relief":
            self.session.set_form("relief")
        elif self.session.image_mode == "silhouette":
            self.session.set_form("profile")
        self.left.refresh(self.session)

    def clear_image(self) -> None:
        self.session.clear_image()
        self.show_toast("已清除参考图片")

    def set_image_mode(self, mode: str) -> None:
        self.session.image_mode = mode
        if self.session.image is None:
            self.left.show_image_tab()
            self.show_toast("先选一张图片，再选择用途", kind="warn")
            self.left.refresh(self.session)
            return
        if mode == "silhouette":
            self.session.set_form("profile")
            self.show_toast("已用图片轮廓生成旋转体", kind="ok")
        elif mode == "relief":
            self.session.set_form("relief")
            self.show_toast("已用图片亮度生成浮雕", kind="ok")
        else:
            self.session.set_palette("image")
            self.show_toast("已套用图片主色", kind="ok")
        self.left.refresh(self.session)

    # ---------------------------------------------------------------- 导出
    def toggle_export_format(self, key: str) -> None:
        self.export_formats[key] = not self.export_formats.get(key, True)
        if not any(self.export_formats.get(k) for k in
                   ("stl", "three_mf", "obj", "png", "card", "report")):
            self.export_formats[key] = True
            self.show_toast("至少保留一种导出内容", kind="warn")
        self.right.refresh_formats(self.export_formats)

    def open_export_dialog(self) -> None:
        try:
            mesh = self.session.generate_mesh(preview=False)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc, title="生成失败")
            return
        session = self.session
        options = {
            **{k: bool(self.export_formats.get(k)) for k in
               ("stl", "three_mf", "obj", "png", "card", "report")},
            "png_width": 1400,
            "png_height": 1050,
            "transparent": False,
            "supersample": 2,
            "quality": session.quality_id,
            "size_mm": session.size_mm,
            "hollow": session.hollow,
            "auto_orient": False,
        }
        idea = session.idea(mesh.stats_text())
        report = printcheck.analyze(mesh)
        dialog = dialogs.ExportDialog(
            self,
            defaults=options,
            out_dir=config.exports_dir(self.settings.get("export_dir")),
            title=idea.title,
            stats=mesh.stats_text(),
            report=report,
        )
        result = dialog.show_modal()
        if not result:
            return
        opts = result["options"]
        self.export_formats.update({
            k: bool(opts.get(k)) for k in ("stl", "three_mf", "obj", "png", "card", "report")
        })
        self.right.refresh_formats(self.export_formats)

        # 导出窗口里改的精度 / 尺寸 / 中空要真正生效
        if opts.get("quality") and opts["quality"] != session.quality_id:
            session.set_quality(opts["quality"])
        if opts.get("size_mm") and abs(float(opts["size_mm"]) - session.size_mm) > 0.5:
            session.set_size(float(opts["size_mm"]))
        if bool(opts.get("hollow")) != session.hollow:
            session.set_hollow(bool(opts["hollow"]))
        self._run_export(opts, Path(result["out_dir"]))

    def _run_export(self, options: dict, out_dir: Path) -> None:
        if self._export_token is not None:
            self.show_toast("正在导出中，请稍候", kind="warn")
            return
        snapshot = self.session.snapshot()
        session = self.session
        palette = session.palette()
        palette_id = session.palette_id
        idea = session.idea()
        image_info = session.image_info()
        camera = self.viewport.camera
        settings = self.settings
        auto_orient = bool(options.get("auto_orient"))

        def work(token=None):
            mesh = session.generate_mesh(preview=False)
            token.raise_if_cancelled()
            report = printcheck.analyze(mesh)
            if auto_orient:
                mesh, report, _label = printcheck.auto_orient(mesh)
            token.raise_if_cancelled()

            def progress(stage: str, frac: float) -> None:
                self.jobs.post(lambda: self._set_progress(stage, frac))

            result = exporter.export_bundle(
                mesh,
                form_id=snapshot.form,
                seed=snapshot.seed,
                res=snapshot.res,
                params={"a": snapshot.a, "b": snapshot.b},
                palette_id=palette_id,
                palette=palette,
                idea=idea,
                image_info=image_info,
                out_dir=out_dir,
                options=exporter.ExportOptions(
                    stl=bool(options.get("stl")),
                    three_mf=bool(options.get("three_mf")),
                    obj=bool(options.get("obj")),
                    png=bool(options.get("png")),
                    card=bool(options.get("card")),
                    report=bool(options.get("report")),
                    png_width=int(options.get("png_width", 1400)),
                    png_height=int(options.get("png_height", 1050)),
                    supersample=int(options.get("supersample", 2)),
                    transparent=bool(options.get("transparent")),
                ),
                camera=camera,
                concept_id=session.concept if session.is_concept_mode() else None,
                size_mm=session.size_mm,
                quality_label=session.quality.label,
                print_report=report,
                on_progress=progress,
                token=token,
            )
            return result

        self._set_busy_ui(True)
        self._set_progress("准备导出", 0.05)

        def done(result) -> None:
            self._export_token = None
            self._set_busy_ui(False)
            self._set_progress("", 0.0)
            self.show_toast(result.summary(), kind="ok", duration=3600)
            settings.add_recent_export({
                "path": str(result.files[0]) if result.files else str(result.folder),
                "name": result.basename,
                "folder": str(result.folder),
            })
            if str(out_dir) != str(settings.get("export_dir", "")):
                settings.set("export_dir", str(out_dir))
            settings.save()
            self.right.refresh_recent(settings.get("recent_exports", []))

        def failed(exc) -> None:
            self._export_token = None
            self._set_busy_ui(False)
            self._set_progress("", 0.0)
            self.show_error(exc, title="导出失败")

        self._export_token = self.jobs.submit(work, on_done=done, on_error=failed, name="export")

    def _set_progress(self, stage: str, frac: float) -> None:
        if stage:
            self.progress_label.configure(text=stage)
            self.progress.set(frac)
            self.cancel_btn.pack(side="right", padx=(6, 0))
        else:
            self.progress_label.configure(text="")
            self.progress.set(0.0)
            self.cancel_btn.pack_forget()

    def _cancel_export(self) -> None:
        if self._export_token is not None:
            self._export_token.cancel()
            self.progress_label.configure(text="正在取消…")

    def _set_busy_ui(self, busy: bool) -> None:
        self.cancel_btn.pack_forget()
        if not busy:
            self.progress_label.configure(text="")

    def open_export_dir(self) -> None:
        folder = config.exports_dir(self.settings.get("export_dir"))
        exporter.open_folder(folder)

    def open_path(self, path: str | None) -> None:
        if not path:
            return
        p = Path(path)
        if p.exists():
            exporter.open_folder(p)
        else:
            self.show_toast("文件已被移动或删除", kind="warn")

    # ---------------------------------------------------------------- 其它
    def save_settings(self) -> None:
        self.settings.update({
            "auto_rotate": self.viewport.auto_rotate,
            "wire": self.viewport.wire,
            "shadow": self.viewport.shadow,
            "window_size": [self.winfo_width(), self.winfo_height()],
            "window_pos": [self.winfo_x(), self.winfo_y()],
        })
        self.session.save()
        self.show_toast("设置已保存", kind="ok")

    def show_help(self) -> None:
        dialogs.HelpDialog(self).show_modal()

    def show_toast(self, text: str, *, kind: str = "info", duration: int = 2600) -> None:
        if self._toast:
            self._toast.show(text, kind=kind, duration=duration)

    def show_error(self, exc: BaseException, *, title: str = "出错了") -> None:
        if isinstance(exc, AppError):
            dialogs.ErrorDialog(
                self, title=exc.title, message=exc.message, hint=exc.hint, detail=exc.detail
            ).show_modal()
        else:
            log.exception("未处理的异常", exc_info=exc)
            dialogs.ErrorDialog(
                self,
                title=title,
                message="发生了意料之外的问题，操作没有完成。",
                hint="可以重试一次；若反复出现，请把技术细节反馈给开发者。",
                detail=f"{type(exc).__name__}: {exc}",
            ).show_modal()

    def _on_tk_exception(self, exc_type, exc_value, exc_tb) -> None:
        log.error("界面回调异常", exc_info=(exc_type, exc_value, exc_tb))
        try:
            self.show_error(exc_value if exc_value else RuntimeError("未知错误"))
        except Exception:  # noqa: BLE001
            pass

    def _show_more_menu(self) -> None:
        menu = tk.Menu(self, tearoff=0, bg=T.panel_alt, fg=T.text,
                       activebackground=T.accent_soft, activeforeground=T.text, bd=0)
        menu.add_command(label="创建桌面快捷方式", command=self._make_shortcut)
        menu.add_command(label="打开导出目录", command=self.open_export_dir)
        menu.add_command(label="打开日志文件", command=self._open_log)
        menu.add_separator()
        menu.add_command(label="使用帮助 (F1)", command=self.show_help)
        menu.add_command(label="关于", command=self._show_about)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def _make_shortcut(self) -> None:
        try:
            path = shortcut.create_desktop_shortcut(force=True)
            self.show_toast(f"桌面快捷方式已创建：{path}", kind="ok", duration=3600)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)

    def _open_log(self) -> None:
        from .. import logsetup

        path = logsetup.log_path()
        if path.exists():
            exporter.open_folder(path)
        else:
            self.show_toast("还没有日志文件", kind="warn")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "关于",
            f"{config.APP_NAME} v{config.APP_VERSION}\n"
            f"{config.APP_TAGLINE}\n\n"
            "完全离线运行：不联网、不写注册表、不装服务。\n"
            f"日志与设置：{config.user_data_dir()}",
            parent=self,
        )

    def _show_welcome(self) -> None:
        result = dialogs.WelcomeDialog(self, on_shortcut=self._make_shortcut).show_modal()
        self.settings.set("welcome_seen", True, save=True)
        if result:
            self.show_toast("提示：左侧写一句话，或直接拖入一张图片", kind="ok", duration=3600)

    def _key_space(self, _event=None):
        self.viewport.toggle_auto()
        return "break"

    # ================================================================== 关闭
    def on_close(self) -> None:
        try:
            self.save_settings()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.viewport.shutdown()
            self.jobs.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()
