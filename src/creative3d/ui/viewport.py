"""ui.viewport —— 交互式 3D 视口。

交互 → 立即用低精度帧（保证跟手）；停手 ~260ms → 后台补一张 2× 超采样高清帧。
这样「手感」和「画质」不再互相牺牲。
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Callable

from PIL import Image, ImageTk

from .. import config, render
from ..mesh import Mesh
from . import theme

T = config.THEME


class Viewport(tk.Frame):
    def __init__(self, master, *, on_state: Callable[[], None] | None = None) -> None:
        super().__init__(master, bg=T.bg, bd=0, highlightthickness=0)
        self.mesh: Mesh | None = None
        self.palette: config.ModelPalette = config.PALETTES[config.DEFAULT_PALETTE]
        self.camera = render.Camera()
        self.auto_rotate = True
        self.wire = False
        self.shadow = True
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._drag_from: tuple[int, int] | None = None
        self._interactive_scale = 0.82   # 拖拽时的渲染缩放，按实测帧耗时自适应
        self._frame_pending = False      # 把连续的鼠标事件合并成一帧，避免重绘风暴
        self._yaw_hold = 0.0
        self._high_after: str | None = None
        self._anim_after: str | None = None
        self._resize_after: str | None = None
        self._last_frame = 0.0
        self._last_stats = render.RenderStats()
        self._quality = "–"
        self._on_state = on_state
        self._busy = False
        self._empty_reason = ""
        self._on_request_image: Callable | None = None

        self.canvas = tk.Canvas(self, bg=T.bg, highlightthickness=0, bd=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_camera())

        self.toolbar = tk.Frame(self, bg=T.bg_alt, height=40)
        self.toolbar.pack(fill="x", side="bottom")
        self._build_toolbar()
        self._build_hud()
        self._tick()

    # ------------------------------------------------------------------ 构建
    def _build_toolbar(self) -> None:
        from .widgets import Chip, Tooltip

        left = tk.Frame(self.toolbar, bg=T.bg_alt)
        left.pack(side="left", padx=10, pady=6)
        self.chip_rotate = Chip(left, "自转", lambda: self.toggle_auto(), selected=True,
                                glyph="◐", font_size=9)
        self.chip_rotate.pack(side="left", padx=(0, 6))
        Tooltip(self.chip_rotate, "自动旋转（空格）")

        self.chip_wire = Chip(left, "线框", lambda: self.toggle_wire(), glyph="▦", font_size=9)
        self.chip_wire.pack(side="left", padx=(0, 6))
        Tooltip(self.chip_wire, "叠一层三角形线框，方便检查网格密度（W）")

        self.chip_shadow = Chip(left, "投影", lambda: self.toggle_shadow(), selected=True,
                                glyph="◍", font_size=9)
        self.chip_shadow.pack(side="left", padx=(0, 6))
        Tooltip(self.chip_shadow, "地面柔和阴影，可关掉换取更高帧率")

        right = tk.Frame(self.toolbar, bg=T.bg_alt)
        right.pack(side="right", padx=10, pady=6)
        for text, yaw, pitch, tip in (
            ("正面", 0.0, 0.12, "正面视角"),
            ("斜视", -34.0, 22.0, "默认斜视角"),
            ("侧面", 90.0, 0.12, "侧面视角"),
            ("俯视", -34.0, 72.0, "俯视视角"),
        ):
            chip = Chip(right, text, lambda y=yaw, p=pitch: self.set_view(y, p), font_size=8,
                        padx=8, pady=5)
            chip.pack(side="left", padx=(0, 4))
            Tooltip(chip, tip)

    def _build_hud(self) -> None:
        self.hud_left = tk.Frame(self.canvas, bg=T.panel, bd=0, highlightthickness=1,
                                 highlightbackground=T.line_soft)
        self.hud_title = tk.Label(self.hud_left, text="", bg=T.panel, fg=T.text,
                                  font=theme.f(10, True), anchor="w")
        self.hud_title.pack(anchor="w", padx=10, pady=(6, 0))
        self.hud_sub = tk.Label(self.hud_left, text="", bg=T.panel, fg=T.text_mute,
                                font=theme.f(8), anchor="w")
        self.hud_sub.pack(anchor="w", padx=10, pady=(0, 6))

        self.hud_right = tk.Frame(self.canvas, bg=T.panel, bd=0, highlightthickness=1,
                                  highlightbackground=T.line_soft)
        self.hud_stats = tk.Label(self.hud_right, text="", bg=T.panel, fg=T.text_mute,
                                  font=theme.f(8), anchor="e", justify="right")
        self.hud_stats.pack(anchor="e", padx=10, pady=(6, 0))
        self.hud_quality = tk.Label(self.hud_right, text="", bg=T.accent_soft, fg=T.accent,
                                    font=theme.f(8), padx=7, pady=2)
        self.hud_quality.pack(anchor="e", padx=10, pady=(4, 6))

        self.empty = tk.Frame(self.canvas, bg=T.panel, highlightthickness=1,
                              highlightbackground=T.line)
        self.empty_icon = tk.Label(self.empty, text="▣", bg=T.panel, fg=T.accent,
                                   font=theme.f(28))
        self.empty_icon.pack(pady=(18, 4))
        self.empty_title = tk.Label(self.empty, text="", bg=T.panel, fg=T.text,
                                    font=theme.f(11, True))
        self.empty_title.pack()
        self.empty_text = tk.Label(self.empty, text="", bg=T.panel, fg=T.text_mute,
                                   font=theme.f(9), wraplength=300, justify="center")
        self.empty_text.pack(pady=(4, 10), padx=18)
        self.empty_btn = tk.Label(self.empty, text="选择图片", bg=T.accent, fg="#ffffff",
                                  font=theme.f(9, True), padx=16, pady=8, cursor="hand2")
        self.empty_btn.pack(pady=(0, 18))
        self.empty_btn.bind("<Button-1>", lambda e: self._request_image())
        self.empty_btn.bind("<Enter>", lambda e: self.empty_btn.configure(bg=T.accent_hover))
        self.empty_btn.bind("<Leave>", lambda e: self.empty_btn.configure(bg=T.accent))

    def set_image_request_handler(self, fn: Callable) -> None:
        self._on_request_image = fn

    def _request_image(self) -> None:
        if self._on_request_image:
            self._on_request_image()

    # ------------------------------------------------------------------ 数据
    def set_mesh(self, mesh: Mesh | None) -> None:
        self.mesh = mesh
        self.request_render(high=True)

    def set_palette(self, palette: config.ModelPalette) -> None:
        self.palette = palette
        self.request_render(high=False)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_hud()

    def set_empty_state(self, title: str = "", text: str = "", button: str = "") -> None:
        self._empty_reason = title
        if not title:
            self.empty.place_forget()
            return
        self.empty_title.configure(text=title)
        self.empty_text.configure(text=text)
        self.empty_btn.configure(text=button or "选择图片")
        self.empty.place(relx=0.5, rely=0.46, anchor="center")
        self.empty.lift()

    # ------------------------------------------------------------------ 相机
    def set_view(self, yaw_deg: float, pitch_deg: float) -> None:
        self.camera.yaw = math.radians(yaw_deg)
        self.camera.pitch = math.radians(pitch_deg)
        self.request_render(high=True)

    def reset_camera(self) -> None:
        self.camera = render.Camera()
        self.request_render(high=True)

    def toggle_auto(self) -> None:
        self.auto_rotate = not self.auto_rotate
        self.chip_rotate.set_selected(self.auto_rotate)
        if not self.auto_rotate:
            self.request_render(high=True)
        self._notify()

    def toggle_wire(self) -> None:
        self.wire = not self.wire
        self.chip_wire.set_selected(self.wire)
        self.request_render(high=True)

    def toggle_shadow(self) -> None:
        self.shadow = not self.shadow
        self.chip_shadow.set_selected(self.shadow)
        self.request_render(high=True)

    def _notify(self) -> None:
        if self._on_state:
            try:
                self._on_state()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 事件
    def _on_resize(self, _event=None) -> None:
        if self._resize_after:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(60, lambda: self.request_render(high=True))

    def _on_press(self, event) -> None:
        self._drag_from = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _on_drag(self, event) -> None:
        if self._drag_from is None:
            return
        dx = event.x - self._drag_from[0]
        dy = event.y - self._drag_from[1]
        self._drag_from = (event.x, event.y)
        self.camera.yaw += dx * 0.010
        self.camera.pitch = max(
            -render.PITCH_LIMIT, min(render.PITCH_LIMIT, self.camera.pitch + dy * 0.008)
        )
        self._schedule_high()
        self._render_interactive()

    def _render_interactive(self) -> None:
        """把一秒内可能收到的几十个鼠标事件合并成一次重绘。"""
        if self._frame_pending:
            return
        self._frame_pending = True
        self.after_idle(self._flush_interactive)

    def _flush_interactive(self) -> None:
        self._frame_pending = False
        self._render_now(high=False)

    def _on_release(self, _event=None) -> None:
        self._drag_from = None
        self._schedule_high()

    def _on_wheel(self, event) -> None:
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        self.camera.zoom = max(0.5, min(4.0, self.camera.zoom * factor))
        self.request_render(high=False)
        self._schedule_high()
        return "break"

    # ------------------------------------------------------------------ 渲染
    def request_render(self, *, high: bool = False) -> None:
        self._render_now(high=high)
        if high:
            self._schedule_high()

    def _schedule_high(self) -> None:
        if self._high_after:
            try:
                self.after_cancel(self._high_after)
            except tk.TclError:
                pass
        self._high_after = self.after(260, self._render_high)

    def _render_high(self) -> None:
        self._high_after = None
        if self._drag_from is None and not self.auto_rotate:
            self._render_now(high=True)

    def _render_now(self, *, high: bool) -> None:
        if self.mesh is None:
            self._draw_placeholder()
            return
        w = max(64, self.canvas.winfo_width())
        h = max(64, self.canvas.winfo_height())
        if w < 80 or h < 80:
            return

        # 交互期间限制单帧成本：分辨率自适应 + 关闭超采样/阴影。
        # 自适应策略：上一帧慢就再降一点，快就慢慢加回来，目标 12–24ms 一帧。
        scale = 1.0 if high else self._interactive_scale
        rw, rh = int(w * scale), int(h * scale)
        try:
            result = render.render_scene(
                self.mesh,
                self.camera,
                rw,
                rh,
                self.palette,
                supersample=2 if high else 1,
                shadow=self.shadow and high,
                wire=self.wire and high,
            )
        except Exception:  # noqa: BLE001 - 渲染失败不应崩界面
            import logging

            logging.getLogger("creative3d.viewport").exception("渲染失败")
            return

        img = result.image
        if scale != 1.0:
            img = img.resize((w, h), Image.BILINEAR)
        if self._photo is None or self._photo.width() != w or self._photo.height() != h:
            self._photo = ImageTk.PhotoImage(img, master=self.canvas)
            if self._image_item is not None:
                self.canvas.delete(self._image_item)
            self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self._photo.paste(img)

        self._last_stats = result.stats
        if not high:
            ms = result.stats.ms
            if ms > 26.0:
                self._interactive_scale = max(0.45, self._interactive_scale - 0.10)
            elif ms < 12.0:
                self._interactive_scale = min(1.0, self._interactive_scale + 0.05)
        self._quality = "精细" if high else "快速"
        self._last_frame = time.perf_counter()
        self._update_hud()
        self._position_hud()

    def _draw_placeholder(self) -> None:
        self.canvas.delete("ph")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, w, h, fill=T.bg, outline="", tags="ph")
        self.canvas.create_text(
            w / 2, h / 2, text="正在准备模型…", fill=T.text_mute, font=theme.f(10), tags="ph"
        )

    # ------------------------------------------------------------------ HUD
    def set_titles(self, title: str, subtitle: str) -> None:
        self.hud_title.configure(text=title)
        self.hud_sub.configure(text=subtitle)
        self._position_hud()

    def _position_hud(self) -> None:
        self.hud_left.place(x=14, y=14)
        self.hud_left.lift()
        self.hud_right.place(relx=1.0, x=-14, y=14, anchor="ne")
        self.hud_right.lift()

    def _update_hud(self) -> None:
        if self.mesh is None:
            self.hud_stats.configure(text="")
            self.hud_quality.configure(text="")
            return
        s = self._last_stats
        self.hud_stats.configure(
            text=f"{self.mesh.n_faces:,} 面\n{s.tris:,} 三角 · {s.ms:.0f}ms"
        )
        self.hud_quality.configure(
            text=("生成中…" if self._busy else self._quality),
            fg=T.warn if self._busy else T.accent,
            bg=T.panel_alt if self._busy else T.accent_soft,
        )

    # ------------------------------------------------------------------ 动画
    def _tick(self) -> None:
        try:
            if self.auto_rotate and self.mesh is not None and self._drag_from is None:
                # 目标 25fps：按帧间隔推进角度，快机器不会转太快
                now = time.perf_counter()
                if now - self._last_frame > 0.032:
                    self.camera.yaw += 0.0125
                    self._render_now(high=False)
            self._anim_after = self.after(28, self._tick)
        except tk.TclError:
            pass

    def shutdown(self) -> None:
        for attr in ("_anim_after", "_high_after", "_resize_after"):
            handle = getattr(self, attr, None)
            if handle:
                try:
                    self.after_cancel(handle)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
